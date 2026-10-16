from pathlib import Path
import sys,json,urllib.request,zipfile,io,hashlib
T=Path(sys.argv[1]);R=T/'experiment/round35';rt=R/'runtime'
for name,version in [('jaxtyping','0.2.38'),('typeguard','2.13.3')]:
 meta=json.loads(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json').read());wheel=next(x for x in meta['urls'] if x['filename'].endswith('py3-none-any.whl') or x['filename'].endswith('py2.py3-none-any.whl'));data=urllib.request.urlopen(wheel['url']).read();assert hashlib.sha256(data).hexdigest()==wheel['digests']['sha256']
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  for n in z.namelist():assert (rt/n).resolve().is_relative_to(rt.resolve())
  z.extractall(rt)
 (R/'results'/f'{name}_source.json').write_text(json.dumps(dict(url=wheel['url'],sha256=wheel['digests']['sha256'],version=version),indent=2),encoding='utf-8')
print('minimal typing dependencies installed; native forecast adapter avoids trainer/GUI packages')
