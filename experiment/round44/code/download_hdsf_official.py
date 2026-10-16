from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,concurrent.futures,time,sys,re,datetime
R=Path(sys.argv[1])/'experiment/round44';Q=R/'results';C=R/'downloads';C.mkdir(exist_ok=True)
def get(url,headers=None):return urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=40)
def js(url):
 with get(url) as r:return json.load(r)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def download(url,path,total,expected=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size==total and (not expected or digest(path)==expected):return digest(path)
 chunks=path.parent/(path.name+'.parts');chunks.mkdir(exist_ok=True);step=4*1024*1024
 def part(i):
  a=i*step;b=min(total,a+step)-1;p=chunks/f'{i:06d}'
  if p.exists() and p.stat().st_size==b-a+1:return
  for attempt in range(5):
   try:
    u=url
    with get(u,{'Range':f'bytes={a}-{b}'}) as r:
     assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{total}',(r.status,r.headers.get('Content-Range'))
     raw=r.read();assert len(raw)==b-a+1
    tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(p);return
   except Exception:
    if attempt==4:raise
    time.sleep(min(2**attempt,10))
 count=(total+step-1)//step
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for n,_ in enumerate(pool.map(part,range(count)),1):
   if n%20==0 or n==count:print(path.name,n,count,flush=True)
 temp=path.with_suffix(path.suffix+'.tmp');h=hashlib.sha256()
 with temp.open('wb') as f:
  for i in range(count):raw=(chunks/f'{i:06d}').read_bytes();f.write(raw);h.update(raw)
 got=h.hexdigest();assert not expected or expected==got,(expected,got);temp.replace(path)
 for p in chunks.iterdir():p.unlink()
 chunks.rmdir();return got
out=[]
if sys.argv[2]=='torch':
 index='https://download.pytorch.org/whl/cpu/torch/'
 with get(index) as r:html=r.read().decode()
 link=next(x for x in re.findall(r'href="([^"]+)"',html) if 'torch-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl' in x)
 url=urllib.parse.urljoin(index,link.split('#')[0]).replace('download-r2.pytorch.org','download.pytorch.org');expected=link.split('#sha256=')[1];path=C/'torch-2.8.0+cpu-cp312-cp312-win_amd64.whl';got=download(url,path,619352539,expected);out.append(dict(url=url,path=str(path),bytes=path.stat().st_size,sha256=got,official_sha256=expected))
else:
 for mid,rev in [('Datadog/Toto-2.0-313m',None),('ibm-granite/granite-timeseries-patchtst-fm-r2','b125275f9204d37cbb81fe47b9cf5e08a521e829')]:
  info=js('https://huggingface.co/api/models/'+mid+('/revision/'+rev if rev else '')+'?blobs=true');rev=info['sha'];base=R/'model-cache'/('models--'+mid.replace('/','--'));snap=base/'snapshots'/rev;snap.mkdir(parents=True,exist_ok=True);(base/'refs').mkdir(exist_ok=True);(base/'refs/main').write_text(rev)
  for item in info['siblings']:
   name=item['rfilename']
   if name not in ['config.json','model.safetensors','README.md','LICENSE','LICENSE.txt','LICENSE.md','NOTICE','NOTICE.md']:continue
   url='https://huggingface.co/'+mid+'/resolve/'+rev+'/'+name;size=item.get('size') or item.get('lfs',{}).get('size');expected=item.get('lfs',{}).get('sha256');path=snap/name
   if size and size>10*1024*1024:got=download(url,path,size,expected)
   else:
    with get(url) as r:raw=r.read()
    path.write_bytes(raw);got=digest(path)
   out.append(dict(model_id=mid,revision=rev,url=url,path=str(path),bytes=path.stat().st_size,sha256=got,official_lfs_sha256=expected));print(mid,name,path.stat().st_size,flush=True)
(Q/('official_'+sys.argv[2]+'_downloads.json')).write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=out),ensure_ascii=False,indent=2),encoding='utf-8');print('Official downloads verified',sys.argv[2],flush=True)
