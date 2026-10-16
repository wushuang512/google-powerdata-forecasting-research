"""Complete cell-a official Parquet projection; no BigQuery, no paid cloud job."""
import argparse,collections,concurrent.futures,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from parquet_remote import RemoteFile
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent
COLUMNS=['start_time','end_time','machine_id','collection_type','average_usage.cpus','average_usage.memory','sample_rate']

def prefetch(cell,obj):
    key=obj['name'].removesuffix('.parquet.gz');dest=ROOT/'projected_usage'/cell
    if (dest/(key+'.npz')).exists() and (dest/(key+'.json')).exists():return obj
    if shutil.disk_usage(ROOT).free<8*2**30:raise RuntimeError('Disk reserve reached during projected-column download')
    rf=RemoteFile(cell,obj);pf=pq.ParquetFile(rf,pre_buffer=False);rf.prefetch_columns(pf.metadata,COLUMNS)
    p=rf.dir/'prefetch.json';old=json.loads(p.read_text()) if p.exists() else dict(network_bytes=0,requests=0)
    p.write_text(json.dumps(dict(network_bytes=old['network_bytes']+rf.bytes,requests=old['requests']+rf.requests)),encoding='utf-8')
    return obj
def mapping(cell):
    m=pd.read_csv(TOP/'google-powerdata-2019/csv/machine_to_pdu_mapping.csv');m=m[m.cell==cell];m=m[~m.duplicated('machine_id',keep=False)].sort_values('machine_id');names=np.load(TOP/'experiment/results/data.npz')['names'];return m.machine_id.to_numpy(),np.array([list(names).index(f'cell{cell}_{p}') for p in m.pdu]),names
def process(cell,obj):
    dest=ROOT/'projected_usage'/cell;dest.mkdir(parents=True,exist_ok=True);key=obj['name'].removesuffix('.parquet.gz');npz=dest/(key+'.npz');log=dest/(key+'.json')
    if npz.exists() and log.exists():return npz,json.loads(log.read_text())
    if shutil.disk_usage(ROOT).free<8*2**30:raise RuntimeError('Disk reserve reached; exact resumable extraction paused')
    mids,pdu,names=mapping(cell);rf=RemoteFile(cell,obj);pf=pq.ParquetFile(rf,pre_buffer=False);rf.prefetch_columns(pf.metadata,COLUMNS);agg=np.zeros((57,2976,6));stats=collections.Counter();seen=set();t0=time.perf_counter();minstart=10**20;maxend=0
    for group in range(pf.num_row_groups):
        table=pf.read_row_group(group,columns=COLUMNS,use_threads=False);start=table['start_time'].to_numpy();end=table['end_time'].to_numpy();machine=table['machine_id'].to_numpy();typ=table['collection_type'].to_numpy();usage=table['average_usage'].combine_chunks();cpu=usage.field('cpus').to_numpy(zero_copy_only=False);mem=usage.field('memory').to_numpy(zero_copy_only=False);rate=table['sample_rate'].to_numpy();stats['all_rows']+=len(start);stats['job_rows']+=int((typ==0).sum());stats['alloc_rows']+=int((typ==1).sum());minstart=min(minstart,int(start.min()));maxend=max(maxend,int(end.max()))
        pos=np.searchsorted(mids,machine);safe=np.minimum(pos,len(mids)-1);matched=(pos<len(mids))&(mids[safe]==machine);slot=np.ceil((end-600000000)/900000000).astype('int64')-1;duration=(end-start)/1e6
        ok=(typ==0)&matched&(slot>=0)&(slot<2976)&(duration>0)&(duration<=300.000001)&np.isfinite(cpu)&np.isfinite(mem)&(cpu>=0)&(mem>=0);stats['used_job_rows']+=int(ok.sum());stats['rejected_job_rows']+=int(((typ==0)&~ok).sum());seen.update(machine[ok].tolist())
        flat=pdu[safe[ok]]*2976+slot[ok];dur=duration[ok];val=[cpu[ok]*dur/900,mem[ok]*dur/900,dur/900,np.ones(len(dur)),np.nan_to_num(rate[ok])*dur,dur]
        for k,v in enumerate(val):agg[:,:,k]+=np.bincount(flat,weights=v,minlength=57*2976).reshape(57,2976)
    tmp=npz.with_suffix('.tmp.npz');np.savez_compressed(tmp,aggregate=agg);tmp.replace(npz)
    chunks=[]
    for f in sorted(rf.dir.glob('*.bin')):chunks.append(dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    downloaded=json.loads((rf.dir/'prefetch.json').read_text()) if (rf.dir/'prefetch.json').exists() else dict(network_bytes=0,requests=0)
    row=dict(cell=cell,file=obj['name'],object_bytes=int(obj['size']),object_md5=obj['md5Hash'],columns=COLUMNS,**dict(stats),min_start_us=minstart,max_end_us=maxend,machines_observed=sorted(seen),network_bytes=rf.bytes+downloaded['network_bytes'],requests=rf.requests+downloaded['requests'],elapsed_seconds=time.perf_counter()-t0,raw_range_files=chunks,aggregate_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),complete=True)
    log.write_text(json.dumps(row),encoding='utf-8');return npz,row
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',default='a');ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--download-workers',type=int,default=0);a=ap.parse_args();objs=json.loads((ROOT/'docs'/f'gcs_{a.cell}_objects.json').read_text());items=sorted([x for x in objs if x['name'].startswith('instance_usage-') and x['name'].endswith('.parquet.gz')],key=lambda x:x['name'])
    expected=len(items)
    if a.limit:items=items[:a.limit]
    sums=np.zeros((57,2976,6));rows=[];seen=set();start=time.perf_counter();totalnet=0
    def completed():
        if not a.download_workers:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:yield from ex.map(lambda x:process(a.cell,x),items)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.download_workers) as downloads,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as decoding:
                ds=[downloads.submit(prefetch,a.cell,obj) for obj in items];pending=set()
                for f in concurrent.futures.as_completed(ds):
                    pending.add(decoding.submit(process,a.cell,f.result()))
                    if len(pending)>=32:
                        done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                        for z in done:yield z.result()
                for z in concurrent.futures.as_completed(pending):yield z.result()
    for i,(npz,row) in enumerate(completed(),1):
            sums+=np.load(npz)['aggregate'];seen.update(row.pop('machines_observed'));rows.append(row);totalnet+=row['network_bytes']
            if i%25==0 or i==len(items):
                status=dict(cell=a.cell,completed=i,expected=expected,planned=len(items),all_projected_files_complete=i==expected,network_bytes_total=totalnet,elapsed_seconds=time.perf_counter()-start,rows_total=sum(x['all_rows'] for x in rows),job_rows_total=sum(x['used_job_rows'] for x in rows));(ROOT/'results'/f'projection_progress_{a.cell}.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    _,_,names=mapping(a.cell);np.savez_compressed(ROOT/'aggregated'/f'complete_usage_{a.cell}_15min.npz',aggregate=sums.reshape(57,31,96,6),names=names,features=np.array(['completed_job_cpu_NCU','completed_job_memory_normalized','job_window_equivalent','job_window_count','sample_rate_duration_sum','duration_seconds_sum']),source_files=len(rows),expected_source_files=expected,complete=len(rows)==expected)
    summary=dict(cell=a.cell,complete=len(rows)==expected,files_read=len(rows),expected_files=expected,job_rows=sum(x['used_job_rows'] for x in rows),all_rows=sum(x['all_rows'] for x in rows),network_bytes=sum(x['network_bytes'] for x in rows),object_bytes=sum(x['object_bytes'] for x in rows),machines_observed=len(seen),domains={n:dict(observed_slots=int((sums[j,:,3]>0).sum()),job_windows=int(sums[j,:,3].sum())) for j,n in enumerate(names) if n.startswith('cell'+a.cell+'_')},retained_raw_ranges=True,aggregation='JOB only; entire windows assigned to end_time bins; at most 300 seconds timing delay. Complete projected export, not a guarantee all physical workload monitored.',object_manifest_sha256=hashlib.sha256((ROOT/'docs'/f'gcs_{a.cell}_objects.json').read_bytes()).hexdigest())
    (ROOT/'results'/f'complete_projection_{a.cell}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
