from pathlib import Path
import csv,hashlib,importlib.util,json,os,shutil,sys,urllib.request

REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sha_source(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def source_path(work):
 for p in [work/'experiment/round44/reference/PowerData_Prediction',REPO/'third_party/granite-tsfm',work/'third_party/granite-tsfm',work/'experiment/round35/code/official_uni2ts/src']:
  if p.exists():sys.path.insert(0,str(p))
def initialize(work):
 work=Path(work).resolve()
 if work.is_relative_to(REPO) or work==REPO.parent:raise ValueError('Keep the reproduction work directory outside the Git checkout')
 work.mkdir(parents=True,exist_ok=True)
 marker=work/'reproduction_workspace.json'
 source_digest=hashlib.sha256()
 codes=list((REPO/'repro').glob('*.py'))+[REPO/'reproduce.py',REPO/'google-clusterdata-2019/code/finetune_best_foundation.py']
 codes += [p for p in (REPO/'experiment').rglob('*.py') if '/reference/' not in p.as_posix() and '/official_uni2ts/' not in p.as_posix()]
 for code in sorted(codes):source_digest.update(code.relative_to(REPO).as_posix().encode()+code.read_bytes())
 identity={'mode':'replication_of_published_frozen_parameters_not_new_blind_test','config_sha256':sha(REPO/'repro/google_frozen.json'),'assets_lock_sha256':sha(REPO/'repro/assets.lock.json'),'reproduction_code_sha256':source_digest.hexdigest()}
 if marker.exists():
  old=read(marker)
  if old!=identity:raise RuntimeError('Work directory belongs to different reproduction inputs; choose a new directory')
 elif any(work.iterdir()):raise ValueError('A new work directory must be empty')
 else:save(marker,identity)
 # Only source/configuration files are materialized, never local result arrays.
 for p in (REPO/'experiment').rglob('*.py'):
  if '/reference/' in p.as_posix() or '/official_uni2ts/' in p.as_posix():continue
  rel=p.relative_to(REPO);dest=work/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and sha(dest)!=sha(p):raise RuntimeError('Copied research source changed: '+str(rel))
  if not dest.exists():shutil.copyfile(p,dest)
 for rel in ['experiment/round44/results/selection.json','experiment/round46/results/selection.json','experiment/round47/docs/protocol.json','experiment/round47/deployment/selected_configuration.json','experiment/round48/deployment/selected_configuration.json','experiment/round5/results/selection.json','experiment/round5/docs/calibration_features.json','experiment/round8b/results/selection.json']:
  p=REPO/rel;dest=work/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copyfile(p,dest)
 for rel in ['experiment/results','experiment/round44/results','experiment/round44/models','experiment/round46/results','experiment/round46/models','experiment/round47/results','experiment/round47/models','reproduction/results']:(work/rel).mkdir(parents=True,exist_ok=True)
 # The upstream code remains in its fixed submodule. A directory symlink is
 # optional: copying unchanged tracked code also works on Windows without admin.
 return work
def attach_sources(work,source_root=None):
 source_root=Path(source_root) if source_root else REPO
 for rel in ['experiment/round44/reference/PowerData_Prediction','experiment/round35/code/official_uni2ts','third_party/granite-tsfm']:
  src=source_root/rel
  if rel=='third_party/granite-tsfm' and not src.exists():src=source_root/'experiment/round44/reference/granite-tsfm-3d92535ca4138648b9d1e2f3f1421b5dec4dc3aa'
  if not src.exists() or not any(src.iterdir()):continue
  dest=work/rel
  lock=REPO/'repro/upstream_sources.lock.json'
  if lock.exists():
   for row in read(lock).get(rel,[]):
    if not (src/row['path']).is_file() or sha_source(src/row['path'])!=row['sha256']:raise ValueError('Upstream source differs from recorded version: '+rel+'/'+row['path'])
  if not (dest.exists() and any(dest.iterdir())):shutil.copytree(src,dest,ignore=shutil.ignore_patterns('.git','__pycache__','.venv','.model-cache','raw.zip','*.npz','*.csv','*.parquet'),dirs_exist_ok=True)
  if lock.exists():
   for row in read(lock).get(rel,[]):
    if not (dest/row['path']).is_file() or sha_source(dest/row['path'])!=row['sha256']:raise ValueError('Copied upstream source changed: '+rel+'/'+row['path'])
 source_path(work)
def download(row,dest):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']:return
  raise RuntimeError('Existing asset differs from lock: '+str(dest))
 part=dest.with_suffix(dest.suffix+'.part')
 req=urllib.request.Request(row['url'],headers={'User-Agent':'powerdata-research-reproduction','Accept-Encoding':'identity'})
 with urllib.request.urlopen(req,timeout=90) as r,part.open('wb') as f:shutil.copyfileobj(r,f,1024*1024)
 if part.stat().st_size!=row['bytes'] or sha(part)!=row['sha256']:raise RuntimeError('Asset checksum mismatch: '+dest.name)
 part.replace(dest)
def fetch(work,kind,accept_timesfm=False):
 lock=read(REPO/'repro/assets.lock.json')
 if kind in ['google','nlr']:
  for row in lock[kind+'_data']:download(row,work/row['path'])
 else:
  if not accept_timesfm:raise ValueError('Read official TimesFM 3 license; use --accept-timesfm-license for noncommercial/nonproduction local research only')
  for m in lock['models']:
   for row in m['files']:download(row,work/m['location']/row['name'])
  prepare_model_cache(work)
def prepare_model_cache(work):
 for m in read(REPO/'repro/assets.lock.json')['models']:
  base=work/'experiment/round44/model-cache'/('models--'+m['model_id'].replace('/','--'))
  snap=base/'snapshots'/m['revision'];snap.mkdir(parents=True,exist_ok=True)
  for row in m['files']:
   src=work/m['location']/row['name'];dest=snap/row['name']
   if src==dest or not src.exists() or dest.exists():continue
   try:os.link(src,dest)
   except OSError:shutil.copyfile(src,dest)
  (base/'refs').mkdir(exist_ok=True);(base/'refs/main').write_text(m['revision'],encoding='utf-8')
def linked_local_assets(work,local,kind):
 # Development/validation convenience only. This does not count as a download test.
 lock=read(REPO/'repro/assets.lock.json');rows=lock[kind+'_data'] if kind in ['google','nlr'] else [dict(x,path=m['location']+'/'+x['name']) for m in lock['models'] for x in m['files']]
 for row in rows:
  src=Path(local)/row['path'];dest=work/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():continue
  if src.stat().st_size!=row['bytes'] or sha(src)!=row['sha256']:raise ValueError('Local asset differs from official lock: '+row['path'])
  try:os.link(src,dest)
  except OSError:shutil.copyfile(src,dest)
 if kind=='models':prepare_model_cache(work)
def export_csv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
