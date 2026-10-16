"""Read-only conservative storage forecast for the active complete projection."""
import json,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    agg_bytes=0;agg_files=0;evidence_bytes=0;aggregate_keys=set()
    for f in os.scandir(ROOT/'projected_usage/a'):
        if f.name.endswith('.npz'):
            agg_files+=1;agg_bytes+=f.stat().st_size;aggregate_keys.add(f.name[:-4])
        elif f.name.endswith('.json'):evidence_bytes+=f.stat().st_size
    completed_objects=0;cache_bytes=0;pending_bytes=0
    for entry in os.scandir(ROOT/'column_cache/a'):
        if not entry.is_dir():continue
        ready=entry.name in aggregate_keys;size=0
        for f in os.scandir(entry.path):
            if f.name=='prefetch.json':ready=True
            if f.is_file() and f.name.endswith('.bin'):size+=f.stat().st_size
        if ready:completed_objects+=1;cache_bytes+=size
        else:pending_bytes+=size
    expected=12085
    free=shutil.disk_usage(ROOT).free
    raw_remaining=max(0,expected-completed_objects)*cache_bytes/max(completed_objects,1)-pending_bytes
    agg_remaining=max(0,expected-agg_files)*(agg_bytes+evidence_bytes)/max(agg_files,1)
    row=dict(cached_objects=completed_objects,aggregated_objects=agg_files,expected_objects=expected,
             retained_range_bytes=cache_bytes+pending_bytes,aggregate_and_evidence_bytes=agg_bytes+evidence_bytes,
             free_bytes=free,estimated_remaining_bytes=raw_remaining+agg_remaining,
             estimated_free_after_projection_bytes=free-raw_remaining-agg_remaining,
             reserve_bytes=8*2**30,method='Measured average over existing objects; approximate, not a hard guarantee')
    (ROOT/'results/projection_storage_forecast.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)

if __name__=='__main__':main()
