import gzip,json,sys,collections,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent;BASE=TOP/'experiment'
def read(cell,table):
    path=ROOT/'raw'/cell/f'{table}-000000000000.json.gz'
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)
def aggregate(events,mapping,names):
    n=len(names);lookup={s:i for i,s in enumerate(names)}
    memberships=collections.defaultdict(set)
    for mid,c,p in mapping:memberships[mid].add((c,p))
    # Ambiguous membership has no effective timestamps; exclude instead of choosing a PDU.
    idx={mid:lookup.get(f'cell{next(iter(v))[0]}_{next(iter(v))[1]}',-1) for mid,v in memberships.items() if len(v)==1}
    total=np.bincount([i for i in idx.values() if i>=0],minlength=n)
    # All timestamps are microseconds relative to the same trace origin.
    # Slots are end-of-interval states: future events never change an earlier slot.
    bins=31*96;result=np.zeros((n,bins,8));state={};counts=np.zeros((n,7));pointer=0
    events=sorted(events,key=lambda x:(int(x['time']),int(x['machine_id'])))
    def contribution(v):
        active,cpu,mem,knowncpu,knownmem,missing=v
        return np.array([1,active,cpu if active and knowncpu else 0,mem if active and knownmem else 0,active*knowncpu,active*knownmem,missing],float)
    for t in range(bins):
        end=(600+(t+1)*900)*1000000
        while pointer<len(events) and int(events[pointer]['time'])<end:
            e=events[pointer];pointer+=1;mid=int(e['machine_id']);i=idx.get(mid,-1)
            if i<0:continue
            old=state.get(mid);typ=int(e.get('type',0))
            if old is not None:counts[i]-=contribution(old)
            if typ==2:
                v=(0,0,0,False,False,int(e.get('missing_data_reason',0))!=0)
            elif typ in (1,3):
                cap=e.get('capacity') or {};old=old or (0,0,0,False,False,False)
                cpu=float(cap.get('cpus',old[1]));mem=float(cap.get('memory',old[2]))
                v=(1,cpu,mem,('cpus' in cap) or old[3],('memory' in cap) or old[4],int(e.get('missing_data_reason',0))!=0)
            else:
                if old is not None:counts[i]+=contribution(old)
                continue
            state[mid]=v;counts[i]+=contribution(v)
        den=np.maximum(total,1);result[:,t]=np.stack([counts[:,0]/den,counts[:,1]/den,counts[:,2]/den,counts[:,3]/den,counts[:,4]/den,counts[:,5]/den,counts[:,6]/den,total>0],-1)
    return result,total,idx
def main():
    d=np.load(BASE/'results/data.npz');names=d['names'];m=pd.read_csv(TOP/'google-powerdata-2019/csv/machine_to_pdu_mapping.csv',dtype={'machine_id':'int64'})
    # Machine IDs are documented as cell-scoped. Use a separate state per cell.
    arrays=[];report={};all_total=np.zeros(57,int)
    final=np.zeros((57,2976,8));features=['mapped_machine_observed_fraction','active_fraction','mean_cpu_capacity_NCU','mean_memory_capacity_normalized','known_cpu_capacity_fraction','known_memory_capacity_fraction','state_missing_reason_fraction','mapping_available']
    for cell in 'abcdefgh':
        mm=m[m.cell==cell];ev=list(read(cell,'machine_events'));arr,total,idx=aggregate(ev,list(zip(mm.machine_id,mm.cell,mm.pdu)),names)
        sel=d['cells']==cell;final[sel]=arr[sel];all_total[sel]=total[sel]
        observed=set(int(e['machine_id']) for e in ev);ambiguous=mm[mm.duplicated('machine_id',keep=False)];report[cell]=dict(mapped_machines=len(mm),mapped_unique=int(mm.machine_id.nunique()),event_rows=len(ev),event_machines=len(observed),matched_machines=len(set(idx)&observed),ambiguous_machine_ids=sorted(set(ambiguous.machine_id.astype(int))),mapped_domains=int(sel.sum()),capacity_missing_event_rows=sum(not e.get('capacity') for e in ev))
    np.savez_compressed(ROOT/'aggregated/resources_15min.npz',resource=final.reshape(57,31,96,8),names=names,features=np.array(features),mapped_machines=all_total)
    rows=[]
    for i,n in enumerate(names):
        for t in range(2976):rows.append([n,d['cells'][i],600+(t+1)*900,*final[i,t]])
    pd.DataFrame(rows,columns=['series','cell','slot_end_trace_seconds']+features).to_csv(ROOT/'aggregated/resources_15min.csv',index=False)
    audits={}
    all_ev=list(read('a','machine_events'));mm=m[m.cell=='a'];mapping=list(zip(mm.machine_id,mm.cell,mm.pdu));original,_,_=aggregate(all_ev,mapping,names)
    for o in [14,22,26,30]:
        cutoff=(600+o*86400)*1000000;modified=[dict(e) for e in all_ev if int(e['time'])<cutoff]
        truncated,_,_=aggregate(modified,mapping,names);audits[f'future_events_removed_origin{o}']=bool(np.array_equal(original[:,:o*96],truncated[:,:o*96]))
    r=dict(cells=report,unsupported_domains=names[~np.isin(d['cells'],list('abcdefgh'))].tolist(),all_domains=57,causality_checks=audits,all_passed=all(audits.values()),resource_definition='Scheduler capacity and machine availability, not actual CPU/memory usage',mean_observed_fraction=float(final[d['cells']=='a',:,-8].mean()),static_mapping_assumption='Official machine-to-PDU mapping treated as fixed physical membership; mapping has no per-row effective timestamps')
    (ROOT/'results/mapping_audit.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2),flush=True)
    # Count separately by format: JSON and Parquet are alternative exports, not extra observations.
    size={};allgroups=collections.defaultdict(lambda:dict(files=0,bytes=0))
    for cell in 'abcdefgh':
        g=collections.defaultdict(lambda:dict(files=0,bytes=0))
        for x in json.loads((ROOT/'docs'/f'gcs_{cell}_objects.json').read_text()):
            key=x['name'].split('-')[0]+'_'+x['name'].split('.')[-2];g[key]['files']+=1;g[key]['bytes']+=int(x['size']);allgroups[key]['files']+=1;allgroups[key]['bytes']+=int(x['size'])
        size[cell]=dict(g)
    (ROOT/'docs/official_sizes_by_format.json').write_text(json.dumps(dict(cells=size,total=dict(allgroups)),indent=2))
if __name__=='__main__':main()
