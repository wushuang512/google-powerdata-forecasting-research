from pathlib import Path
import sys,json,hashlib,datetime,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round33';Q=R/'results';Q.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(B/'code'));import common;d=common.load()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def z(path,key):
 with np.load(B/path) as a:return a[key].copy()
proto=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=['perPDU asymmetric10/90','perPDU symmetricabs80','cell pooled scaled symmetricabs80'],selection='DEV leave-one-origin-out coverage within3pp of80 first, then minimum pooledWinkler80. If none satisfy choose smallestcoverage distance thenWinkler. No TEST calibration choice.',dev24=[14,16,18,20],dev7=[14,15],point_models_frozen='R32 selected24 andR31 selected7 unchanged',scope='Additional interval calibration only; no point parameter revision. TEST already repeatedly viewed,7DEV overlap, no independent guarantee.')
pp=R/'docs/protocol.json';pp.parent.mkdir(parents=True,exist_ok=True);assert not pp.exists();pp.write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf-8');results={};scale=d['y'][:,:14].mean((1,2)).clip(.05)
def bounds(mode,y,p,m):
 err=y-p;lo=[];hi=[]
 if mode==2:
  radii={c:np.quantile(abs(err[d['cells']==c]/scale[d['cells']==c,None,None])[m[d['cells']==c]],.8) for c in np.unique(d['cells'])}
 for i in range(57):
  a=err[i][m[i]]
  if mode==0:l,h=np.quantile(a,[.1,.9])
  else:r=np.quantile(abs(a),.8) if mode==1 else radii[d['cells'][i]]*scale[i];l,h=-r,r
  lo.append(l);hi.append(h)
 return np.array(lo),np.array(hi)
def metrics(y,p,m,l,h):
 lo=np.clip(p+l[:,None,None],0,1);hi=np.clip(p+h[:,None,None],0,1);w=hi-lo;wink=w+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0);return dict(nominal_percent=80,coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),mean_width_normalized=float(w[m].mean()),winkler80=float(wink[m].mean()))
for H,nr,key in [(1,32,'TTM_head_sparsepeak50_strength0.1_H1'),(7,31,'TTM_head_shape10_H7')]:
 oo=proto[f'dev{24 if H==1 else 7}'];p=z(f'round{nr}/results/dev_predictions.npz',key);y,m=common.truth(d,oo,H);cv=[]
 for mode in range(3):
  all_lo=np.zeros_like(p);all_hi=np.zeros_like(p)
  for j in range(len(oo)):
   train=[k for k in range(len(oo)) if k!=j];lo,hi=bounds(mode,y[:,train],p[:,train],m[:,train]);all_lo[:,j]=np.clip(p[:,j]+lo[:,None],0,1);all_hi[:,j]=np.clip(p[:,j]+hi[:,None],0,1)
  w=all_hi-all_lo;wink=w+10*np.maximum(all_lo-y,0)+10*np.maximum(y-all_hi,0);cov=float(100*((y>=all_lo)&(y<=all_hi))[m].mean());cv.append(dict(mode=mode,name=proto['models'][mode],coverage_percent=cov,winkler80=float(wink[m].mean()),within3pp=abs(cov-80)<=3))
 feasible=[v for v in cv if v['within3pp']];selected=min(feasible,key=lambda r:(r['winkler80'],r['mode'])) if feasible else min(cv,key=lambda r:(abs(r['coverage_percent']-80),r['winkler80'],r['mode']));lo,hi=bounds(selected['mode'],y,p,m);np.savez_compressed(Q/f'calibrator{H}.npz',loq=lo,hiq=hi);results[str(H)]=dict(cv=cv,selected=selected,dev_source_sha256=sha(B/f'round{nr}/results/dev_predictions.npz'),calibrator_sha256=sha(Q/f'calibrator{H}.npz'))
sel=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(pp),horizons=results,no_test_used=True);save('selection.json',sel);(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps(sel,ensure_ascii=False),flush=True)
test={}
for H,nr in [(1,32),(7,31)]:
 p=z(f'round{nr}/results/test_predictions.npz',f'selected_H{H}');y,m=common.truth(d,list(range(22,31)) if H==1 else[22,23,24],H)
 with np.load(Q/f'calibrator{H}.npz') as q:lo=q['loq'];hi=q['hiq']
 test[str(H)]=metrics(y,p,m,lo,hi);np.savez_compressed(Q/f'test_intervals{H}.npz',lower=np.clip(p+lo[:,None,None],0,1),upper=np.clip(p+hi[:,None,None],0,1))
 import csv
 path=B/f'round{nr}/results/exports/selected_{H}day_curve.csv'
 with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 for j,r in enumerate(rows):i=j//(H*96);v=float(r['forecast_normalized_power']);r.update(lower80_empirical=float(np.clip(v+lo[i],0,1)),upper80_empirical=float(np.clip(v+hi[i],0,1)),interval_source='R33 DEV leave-one-origin-out selected residual calibration')
 with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
save('summary.json',dict(selection=sel,selection_sha256=sha(Q/'selection.json'),test=test,point_models_unchanged=True,independent_confirmation=False));print(json.dumps(dict(test=test,selected={H:results[H]['selected']['name'] for H in results}),ensure_ascii=False))
