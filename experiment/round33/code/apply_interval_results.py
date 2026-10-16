from pathlib import Path
import sys,json,hashlib,csv,numpy as np,datetime
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round33';Q=R/'results';sys.path.insert(0,str(B/'code'));import common;d=common.load()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
s=read(Q/'summary.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and s['selection']['no_test_used'];assert (Q/'selection.json').stat().st_mtime<(Q/'test_intervals1.npz').stat().st_mtime
for H,nr in [(1,32),(7,31)]:
 y,m=common.truth(d,list(range(22,31)) if H==1 else[22,23,24],H)
 with np.load(Q/f'test_intervals{H}.npz') as z:lo=z['lower'];hi=z['upper']
 assert np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all();width=hi-lo;wink=width+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0);calc=dict(coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),mean_width_normalized=float(width[m].mean()),winkler80=float(wink[m].mean()));assert all(abs(calc[k]-s['test'][str(H)][k])<1e-12 for k in calc)
 assert s['selection']['horizons'][str(H)]['calibrator_sha256']==sha(Q/f'calibrator{H}.npz')
compar=B/'round32/results/continued_comparison.json';a=read(compar)
for H in [1,7]:a['intervals'].append(dict(horizon_days=H,method=f'R33 当前{H}天推荐 开发交叉核查选择80%经验区间',kind='DEV leave-one-origin-out; no independent temporal guarantee',**s['test'][str(H)]))
a['interval_selection']=s;a['interval_verification']=dict(all_passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(Q/'selection.json'));compar.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8');(Q/'independent_verification.json').write_text(json.dumps(a['interval_verification'],indent=2),encoding='utf-8')
keys=list(dict.fromkeys(k for r in a['intervals'] for k in r));
with (T/'模型预测区间结果.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(a['intervals'])
print(json.dumps(dict(all_passed=True,test=s['test']),ensure_ascii=False))
