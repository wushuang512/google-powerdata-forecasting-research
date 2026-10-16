from pathlib import Path
import sys,json,hashlib,datetime,csv,ast,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round29';sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(B/'code'));import common;from aligned_metrics import evaluate
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=common.load();checks=[];new=[];iv=[];predsources={}
def check(n,v):checks.append(dict(check=n,passed=bool(v)));assert v,n
def independent(p,orig,H):
 y=np.stack([d['y'][:,o:o+H].reshape(57,H*96) for o in orig],1);m=np.stack([d['valid'][:,o:o+H].reshape(57,H*96) for o in orig],1);per=[];daily=[]
 for i in range(57):
  ok=m[i];per.append(100*abs(p[i]-y[i])[ok].sum()/abs(y[i])[ok].sum());ym=y[i].reshape(-1,96);pm=p[i].reshape(-1,96);vm=ok.reshape(-1,96).all(-1);a=ym.mean(-1)[vm];b=pm.mean(-1)[vm];daily.append(100*abs(a-b).sum()/abs(a).sum())
 return np.mean(per),np.mean(daily)
for n in [29,30]:
 rr=B/f'round{n}';s=read(rr/'results/summary.json');sel=s['selection'];check(f'r{n} frozen SHA',sha(rr/'results/selection.json')==(rr/'results/selection.sha256').read_text());check(f'r{n} no TEST selection',sel['no_test_used'] and sel['retain_controls']);check(f'r{n} pipeline complete',(rr/'results/pipeline_complete.json').exists());check(f'r{n} selection predates TEST',(rr/'results/selection.json').stat().st_mtime<(rr/'results/test_predictions.npz').stat().st_mtime)
 with np.load(rr/'results/test_predictions.npz') as z,np.load(rr/'results/dev_predictions.npz') as dz:
  for name,scores in s['scores'].items():
   for H in [1,7]:
    orig=list(range(22,31)) if H==1 else[22,23,24];devorig=[14,16,18,20] if H==1 else[14,15];key=f'{name}_H{H}';p=z[key];de=dz[key];w,dw=independent(p,orig,H);check(f'r{n} {key} independent metrics',abs(w-scores[str(H)]['wape'])<1e-11 and abs(dw-scores[str(H)]['daily_wape'])<1e-11);check(f'r{n} {key} bounded',p.shape==(57,len(orig),H*96) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1)
    e=evaluate(common,d,orig,p,H);dev=evaluate(common,d,devorig,de,H);yp,mp=common.truth(d,devorig,H);loq=np.array([np.quantile((yp-de)[i][mp[i]],.1) for i in range(57)]);hiq=np.array([np.quantile((yp-de)[i][mp[i]],.9) for i in range(57)]);lo=np.clip(p+loq[:,None,None],0,1);hi=np.clip(p+hiq[:,None,None],0,1);y,m=common.truth(d,orig,H);width=hi-lo;wink=width+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0);im=dict(horizon_days=H,model=name,nominal_percent=80,coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),mean_width_normalized=float(width[m].mean()),winkler80=float(wink[m].mean()),kind='DEV-only per-PDU residual empirical interval; overlapping DEV no coverage guarantee')
    iv.append(im);new.append(dict(horizon_days=H,model=f'r{n}:{name}',model_name=name,report_name=name,status='开发未支持 未采用诊断',development_wape_original_grid=dev['wape'],source=dict(path=f'round{n}/results/test_predictions.npz',array=key,sha256=sha(rr/'results/test_predictions.npz')),prediction_digest=hashlib.sha256(p.astype('float64').tobytes()).hexdigest(),**e));predsources[key]=p
    np.savez_compressed(rr/'results'/f'interval_H{H}_{name.replace(" ","_")}.npz',lower=lo,upper=hi,dev_lower_quantile=loq,dev_upper_quantile=hiq)
  # Joint decoder consistency on shared origins22/23/24.
  for name in s['scores']:check(f'r{n} {name} same first96',np.array_equal(z[f'{name}_H1'][:,:3],z[f'{name}_H7'][:,:,:96]))
 # Persist deployment numerical evidence and scalar-daily coherence; each new architecture remains research-only.
 dep={}
 if n==29:
  with np.load(rr/'results/ttm_deploy.npz') as z:
   for key in z.files:dep[key.removesuffix('_o31')]=z[key]
  with np.load(rr/'results/factor_deployment.npz') as z:
   for key in z.files:dep[key]=z[key]
 else:
  with np.load(rr/'results/deployment.npz') as z:dep={key:z[key] for key in z.files}
 for name,p in dep.items():
  check(f'r{n} {name} deploy valid',p.shape==(57,672) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1);save=rr/'results/exports';save.mkdir(exist_ok=True);stem=name.replace(' ','_');ts=np.datetime64('2019-06-01T00:00')+np.arange(672)*np.timedelta64(15,'m');ids=d.get('names',d.get('pdus',np.arange(57)))
  rows=[]
  for i in range(57):
   for j in range(672):rows.append(dict(pdu=str(ids[i]),timestamp_local=str(ts[j]),predicted_power_normalized=float(p[i,j]),model=name,research_only=True))
  def csvsave(path,rows):
   with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
  csvsave(save/f'{stem}_7day_curve.csv',rows);csvsave(save/f'{stem}_1day_curve.csv',[r for r in rows if r['timestamp_local']<'2019-06-02']);csvsave(save/f'{stem}_7day_daily_mean.csv',[dict(pdu=str(ids[i]),date=f'2019-06-{j+1:02d}',predicted_daily_mean_normalized=float(p[i,j*96:(j+1)*96].mean()),model=name,research_only=True) for i in range(57) for j in range(7)])
# Replay low-rank training and forecast with future truth perturbed, using the actual functions from AST.
src=(R/'code/new_architecture_experiment.py').read_text(encoding='utf-8');tree=ast.parse(src);nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['fit_factor','features','fit','predict']];grid=np.arange(96);basis=np.stack([np.ones(96)]+[f(2*np.pi*j*grid/96) for j in range(1,7) for f in [np.cos,np.sin]],1);ns=dict(np=np,d=d,basis=basis,pinv=np.linalg.pinv(basis));exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual factor implementation>','exec'),ns);m1=ns['fit'](14,2,100);p1=ns['predict'](m1,[14]);saved=d['y'].copy();d['y'][:,14:]=.999;p2=ns['predict'](ns['fit'](14,2,100),[14]);d['y'][:]=saved;check('factor future truth perturbation invariance',np.array_equal(p1,p2))
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=all(c['passed'] for c in checks),checks=checks,new_rows=new,new_intervals=iv,independent_confirmation=False,scope='Implementation/numerical/source checks; no proof of no adaptive overfitting across previous rounds.')
(R/'results/new_methods_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=out['all_passed'],checks=len(checks),new_rows=len(new),intervals=len(iv)),ensure_ascii=False))
