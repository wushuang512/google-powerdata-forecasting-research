import base64,concurrent.futures,gzip,hashlib,json,urllib.request,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def download(item):
    cell,obj=item;out=ROOT/'raw'/cell/obj['name'];out.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://storage.googleapis.com/clusterdata_2019_{cell}/{obj["name"]}'
    for attempt in range(4):
        try:
            if not out.exists() or out.stat().st_size!=int(obj['size']):
                tmp=out.with_suffix(out.suffix+'.part')
                with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
                    while chunk:=r.read(1024*1024):f.write(chunk)
                tmp.replace(out)
            md=hashlib.md5(out.read_bytes()).digest();assert base64.b64encode(md).decode()==obj['md5Hash']
            count=0
            if out.name.endswith('.json.gz'):
                with gzip.open(out,'rb') as f:
                    for line in f:json.loads(line);count+=1
                fmt='gzip JSON lines'
            else:
                with out.open('rb') as f:
                    assert f.read(4)==b'PAR1';f.seek(-4,2);assert f.read(4)==b'PAR1'
                count=None;fmt='Parquet magic checked; row count not decoded; no outer gzip'
            result=dict(cell=cell,file=obj['name'],url=url,bytes=out.stat().st_size,md5=obj['md5Hash'],rows=count,format=fmt,verified=True)
            print(json.dumps(result),flush=True);return result
        except Exception:
            if attempt==3:raise
            time.sleep(2*(attempt+1))
def main():
    items=[]
    for cell in 'abcdefgh':
        objs=json.loads((ROOT/'docs'/f'gcs_{cell}_objects.json').read_text())
        items.extend((cell,x) for x in objs if x['name'].startswith(('machine_events-','machine_attributes-')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:manifest=list(ex.map(download,items))
    (ROOT/'docs/download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # One complete official shard is retained ONLY for schema/coverage inspection.
    objs=json.loads((ROOT/'docs/gcs_a_objects.json').read_text());usage=min((x for x in objs if x['name'].startswith('instance_usage-') and x['name'].endswith('.json.gz')),key=lambda x:int(x['size']))
    sample=download(('a',usage));(ROOT/'docs/usage_sample_manifest.json').write_text(json.dumps(sample,indent=2),encoding='utf-8')
    print('COMPLETE small tables',sum(x['bytes'] for x in manifest),'sample',sample['bytes'],flush=True)
if __name__=='__main__':main()
