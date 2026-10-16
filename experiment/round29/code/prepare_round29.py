from pathlib import Path
import sys,json,hashlib,datetime,urllib.request
T=Path(sys.argv[1]);R=T/'experiment/round29';Q=R/'results';C=R/'code';M=R/'models';M.mkdir(exist_ok=True)
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GooglePowerData2019-audited-research'}),timeout=90) as r:return r.read()
def fetch(url,p):
 if p.exists():return
 p.write_bytes(get(url))
commit=json.loads(get('https://api.github.com/repos/ibm-granite/granite-tsfm/commits/main'))['sha'];P=C/'ttm_official';P.mkdir(exist_ok=True);(P/'__init__.py').write_text('',encoding='utf-8')
sources=[]
for name in ['modeling_tinytimemixer.py','configuration_tinytimemixer.py']:
 url=f'https://raw.githubusercontent.com/ibm-granite/granite-tsfm/{commit}/tsfm_public/models/tinytimemixer/{name}';fetch(url,P/name);sources.append(dict(url=url,file=str(P/name)))
fetch(f'https://raw.githubusercontent.com/ibm-granite/granite-tsfm/{commit}/LICENSE',P/'LICENSE')
for branch in ['512-720-r2','1024-720-r2']:
 rev=json.loads(get('https://huggingface.co/api/models/ibm-granite/granite-timeseries-ttm-r2/revision/'+branch))['sha'];mp=M/branch;mp.mkdir(exist_ok=True)
 for name in ['config.json','model.safetensors','generation_config.json']:
  url=f'https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/{rev}/{name}';fetch(url,mp/name);sources.append(dict(url=url,file=str(mp/name),branch=branch,revision=rev))
fetch('https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/raw/main/README.md',M/'official_model_card.md')
for v in sources:
 p=Path(v['file']);v.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(Q/'official_sources.json').write_text(json.dumps(dict(sources=sources,code_commit=commit,license='Apache-2.0',scope='IBM official model weights and pinned official inference source; no executable pickle downloaded'),ensure_ascii=False,indent=2),encoding='utf-8')
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='registered before DEV scoring',metric_version_sha256=hashlib.sha256((C/'aligned_metrics.py').read_bytes()).hexdigest(),source_reference_sha256=json.loads((Q/'metric_audit/audit.json').read_text(encoding='utf-8'))['reference_sha256'],dev24=[14,16,18,20],dev7=[14,15],test24=list(range(22,31)),test7=[22,23,24],training_dev_exclusive_end=14,training_final_exclusive_end=22,models=['TTM512-720-r2','TTM1024-720-r2','JointFactorFourier rank2 ridge10','JointFactorFourier rank2 ridge100','JointFactorFourier rank4 ridge10','JointFactorFourier rank4 ridge100'],decoder='one direct672 output; first96 and seven96-point means derive from same curve',features='causal power histories only; factor/standardization fit on first14 days for DEV and22 for final, supervised labels no later than cutoff; no true future CPU',selection='min joint cost mean of four normalized DEV error ratios:24curve,24peak,7curve,7daily; choose candidate with cost<=0.98 and no primary ratio above1.02 vs current recommendations; otherwise retain old recommendations. Ties within0.002 choose fewer model parameters. No TEST ranking changes selection.',primary_metrics=['wape24','peak_mape24','wape7','daily_wape7'],test='score only after selection.json frozen SHA; all preregistered family scores are descriptive diagnostics; selected best is DEV choice or original controls',uncertainty='single repeatedly inspected May2019 month exploratory;7day origins overlap; no independent confirmation or universal gain guarantee',existing_frozen_selections_unchanged=True)
p=R/'docs/protocol.json';assert not p.exists(),'Never overwrite frozen protocol';p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(downloaded_bytes=sum(v['bytes'] for v in sources),models=protocol['models'],protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
