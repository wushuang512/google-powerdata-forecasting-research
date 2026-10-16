import json,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def listing(cell):
    bucket=f'clusterdata_2019_{cell}';items=[];token=None
    while True:
        params=dict(maxResults='1000',fields='items(name,size,md5Hash,crc32c),nextPageToken')
        if token:params['pageToken']=token
        url=f'https://storage.googleapis.com/storage/v1/b/{bucket}/o?'+urllib.parse.urlencode(params)
        with urllib.request.urlopen(url,timeout=90) as r:data=json.load(r)
        items.extend(data.get('items',[]));token=data.get('nextPageToken')
        if not token:break
    (ROOT/'docs'/f'gcs_{cell}_objects.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
    groups={}
    for x in items:
        table=x['name'].split('-')[0];a=groups.setdefault(table,dict(files=0,bytes=0));a['files']+=1;a['bytes']+=int(x['size'])
    print(cell,groups,flush=True);return cell,groups
def main():
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for cell,groups in ex.map(listing,'abcdefgh'):results[cell]=groups
    (ROOT/'docs/official_sizes.json').write_text(json.dumps(results,indent=2))
    for table in ['machine_events','machine_attributes','instance_usage','collection_events','instance_events']:
        url=f'https://storage.googleapis.com/clusterdata_2019_schema/{table}.schema.json'
        with urllib.request.urlopen(url,timeout=60) as r:(ROOT/'docs'/f'{table}.schema.json').write_bytes(r.read())
if __name__=='__main__':main()
