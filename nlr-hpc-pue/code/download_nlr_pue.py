from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlsplit
import argparse,hashlib,json,time,datetime,os,shutil
ap=argparse.ArgumentParser();ap.add_argument('research_root');ap.add_argument('--phase',choices=['metadata','data'],required=True);a=ap.parse_args()
T=Path(a.research_root);R=T/'nlr-hpc-pue';assert R.resolve().parent==T.resolve()
for p in [R/'raw',R/'docs',R/'results',R/'code']:p.mkdir(exist_ok=True,parents=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
manifest_path=R/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(dataset='NLR HPC Facility Power Usage Effectiveness (PUE) Data',doi='10.7799/3015212',official_catalog='https://data.nlr.gov/submissions/300',download_started_utc=utc(),files=[],duplicate_csv_formats_not_downloaded=True)
def download(url,dest,kind):
    previous=next((r for r in manifest['files'] if r['path']==str(dest.relative_to(R)) and r.get('complete')),None)
    if previous and dest.exists() and dest.stat().st_size==previous['bytes']:
        h=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
        if h==previous['sha256']:print(json.dumps(dict(cached=True,file=dest.name,bytes=dest.stat().st_size)),flush=True);return
    if shutil.disk_usage(R).free<1024**3:raise RuntimeError('Less than 1 GiB disk reserve')
    part=dest.with_name(dest.name+'.part')
    last=None
    for attempt in range(3):
        fresh=url+('&' if '?' in url else '?')+'download=1&codex_acquire='+str(time.time_ns())
        req=Request(fresh,headers={'User-Agent':'Mozilla/5.0','Cache-Control':'no-cache','Pragma':'no-cache','Accept-Encoding':'identity'})
        try:
            with urlopen(req,timeout=45) as response:
                host=urlsplit(response.url).hostname
                if host not in ['data.nlr.gov','nrel-datacat-public-prod.s3.us-west-2.amazonaws.com','www.nlr.gov']:raise RuntimeError('Unexpected official download redirect host')
                expected=response.headers.get('Content-Length');expected=int(expected) if expected else None
                content_type=response.headers.get('Content-Type');etag=response.headers.get('ETag');modified=response.headers.get('Last-Modified');disposition=response.headers.get('Content-Disposition')
                digest=hashlib.sha256();count=0;started=time.monotonic();last_progress=started
                with part.open('wb') as out:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        out.write(chunk);digest.update(chunk);count+=len(chunk)
                        if time.monotonic()-last_progress>=15:
                            write(R/'results/download_progress.json',dict(file=dest.name,bytes=count,expected_bytes=expected,elapsed_seconds=round(time.monotonic()-started,1),state='downloading',utc=utc()))
                            print(json.dumps(dict(file=dest.name,downloaded_bytes=count,expected_bytes=expected)),flush=True);last_progress=time.monotonic()
                if expected is not None and count!=expected:raise RuntimeError('HTTP length mismatch')
                if kind=='parquet':
                    with part.open('rb') as f:
                        head=f.read(4);f.seek(-4,os.SEEK_END);tail=f.read(4)
                    if head!=b'PAR1' or tail!=b'PAR1':raise RuntimeError('Not a complete Parquet file')
                part.replace(dest)
                entry=dict(path=str(dest.relative_to(R)),source_url=url,redirect_host=host,redirect_path=urlsplit(response.url).path,bytes=count,sha256=digest.hexdigest(),http_content_length=expected,http_content_type=content_type,http_etag=etag,http_last_modified=modified,http_content_disposition=disposition,complete=True,acquired_utc=utc())
                manifest['files']=[r for r in manifest['files'] if r['path']!=entry['path']]+[entry];manifest['updated_utc']=utc();write(manifest_path,manifest)
                print(json.dumps(dict(complete=True,file=dest.name,bytes=count,sha256=entry['sha256'])),flush=True);return
        except Exception as e:
            # Signed redirect credentials are never logged or retained.
            last=dict(type=type(e).__name__,message=str(e).split('?')[0][:400],attempt=attempt+1,utc=utc(),source_url=url)
            write(R/'results/download_error.json',last)
            print(json.dumps(last),flush=True)
    raise RuntimeError('Official download did not complete: '+json.dumps(last))
if a.phase=='metadata':
    download('https://data.nlr.gov/submissions/300',R/'docs/dataset_page.html','html')
    download('https://data.nlr.gov/node/300/license',R/'docs/license_official.html','html')
    download('https://data.nlr.gov/system/files/300/1769711385-README.md',R/'docs/README_official.md','text')
    download('https://www.nlr.gov/computational-science/measuring-efficiency-pue',R/'docs/facility_pue_official.html','html')
    print('OFFICIAL_README_BEGIN\n'+(R/'docs/README_official.md').read_text(encoding='utf-8')+'\nOFFICIAL_README_END',flush=True)
else:
    assert (R/'docs/license_official.html').exists() and (R/'docs/README_official.md').exists()
    download('https://data.nlr.gov/system/files/300/1757103411-esif.influx.buildingData.PUE.combined.parquet',R/'raw/esif.influx.buildingData.PUE.combined.parquet','parquet')
    download('https://data.nlr.gov/system/files/300/1757105566-esif.influx.buildingData.outside.combined_2.parquet',R/'raw/esif.influx.buildingData.outside.combined.parquet','parquet')
    manifest['raw_data_bytes']=sum(f['bytes'] for f in manifest['files'] if f['path'].startswith('raw'))
    manifest['selected_data_files_complete']=all(any(f['path']==name and f['complete'] for f in manifest['files']) for name in ['raw\\esif.influx.buildingData.PUE.combined.parquet','raw\\esif.influx.buildingData.outside.combined.parquet'])
    manifest['license_source']='docs/license_official.html and docs/README_official.md; do not assume a third-party CC BY label'
    write(manifest_path,manifest);write(R/'results/download_progress.json',dict(state='complete',raw_data_bytes=manifest['raw_data_bytes'],utc=utc()))
    print(json.dumps(dict(selected_data_files_complete=manifest['selected_data_files_complete'],raw_data_bytes=manifest['raw_data_bytes'])),flush=True)
