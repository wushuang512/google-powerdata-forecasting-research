import gzip,json,collections
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent
def main():
    man=json.loads((ROOT/'docs/usage_sample_manifest.json').read_text());p=ROOT/'raw/a'/man['file'];d=np.load(TOP/'experiment/results/data.npz');mapping=pd.read_csv(TOP/'google-powerdata-2019/csv/machine_to_pdu_mapping.csv');mapping=mapping[mapping.cell=='a'];idx={int(x.machine_id):list(d['names']).index('cella_'+x.pdu) for x in mapping.itertuples()}
    sums=np.zeros((57,2976,5));counts=np.zeros((57,2976));machines=[set() for _ in range(2976)];types=collections.Counter();range_=[10**20,0];unique=set();rows=0;kept=0
    with gzip.open(p,'rt') as f:
        for line in f:
            x=json.loads(line);rows+=1;t=int(x['end_time']);start=int(x['start_time']);mid=int(x.get('machine_id',0));typ=int(x['collection_type']);types[typ]+=1;range_[0]=min(range_[0],start);range_[1]=max(range_[1],t);unique.add(mid)
            if typ!=0 or mid not in idx or t<=start or t-start>300000001:continue
            # Record becomes available only at end_time. Assign to containing completion bin.
            slot=int(np.ceil((t/1e6-600)/900)-1)
            if not 0<=slot<2976:continue
            au=x.get('average_usage') or {};cpu=au.get('cpus');mem=au.get('memory')
            if cpu is None or mem is None:continue
            dur=(t-start)/1e6;i=idx[mid];sums[i,slot]+=[float(cpu)*dur,float(mem)*dur,dur,float(x.get('assigned_memory',0) or 0)*dur,float(x.get('sample_rate',0) or 0)*dur];counts[i,slot]+=1;machines[slot].add(mid);kept+=1
            if rows%1000000==0:print('READ',rows,flush=True)
    features=np.zeros((57,2976,6));den=sums[:,:,2];avail=den>0
    features[:,:,0]=np.divide(sums[:,:,0],den,out=np.zeros_like(den),where=avail);features[:,:,1]=np.divide(sums[:,:,1],den,out=np.zeros_like(den),where=avail);features[:,:,2]=np.divide(sums[:,:,3],den,out=np.zeros_like(den),where=avail);features[:,:,3]=np.divide(sums[:,:,4],den,out=np.zeros_like(den),where=avail);features[:,:,4]=np.log1p(counts);features[:,:,5]=avail
    nn=['sample_job_cpu_mean_NCU','sample_job_memory_mean_normalized','sample_assigned_memory_mean','sample_rate_mean','log_observed_job_windows','sample_available']
    np.savez_compressed(ROOT/'aggregated/usage_sample_15min.npz',features=features.reshape(57,31,96,6),counts=counts.reshape(57,31,96),names=d['names'],feature_names=np.array(nn))
    report=dict(file=man['file'],rows=rows,used_job_rows=kept,collection_type_counts=dict(types),start_min_us=range_[0],end_max_us=range_[1],unique_machine_ids=len(unique),domains={n:dict(available_slots=int(avail[i].sum()),rows=int(counts[i].sum()),min_rows_per_available_slot=float(counts[i][avail[i]].min()) if avail[i].any() else 0) for i,n in enumerate(d['names']) if d['cells'][i]=='a'},sample_definition='One smallest complete official JSON export shard selected by compressed size before looking at labels. Export shard sampling mechanism and inclusion probabilities are unknown. Means describe only recorded job windows; not total workload or a population-unbiased estimate.',aggregation='Duration-weighted means in completion bins. No alloc_set records. Partial periods are not treated as observed zero total load.',full_usage_downloaded=False)
    (ROOT/'results/usage_sample_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
