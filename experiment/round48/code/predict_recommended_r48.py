"""Reconstruct the explicitly user-adopted frozen R48 historical forecasts."""
from pathlib import Path
import argparse,csv,json,hashlib,sys
sys.dont_write_bytecode=True
import numpy as np
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(root,h):
    t=Path(root);r=t/'experiment/round48';q=r/'deployment';c=read(q/'selected_configuration.json')
    for rel,digest in c['bindings'].items():assert sha(t/rel)==digest,rel
    sys.path.insert(0,str(r/'code'));import robust_hdsf_experiment as e;e.initialize(t)
    task=c['tasks'][str(h)];p=e.predict('test',h,task['variant']);y,m=e.E.common.truth(e.E.d,e.E.origins('test',h),h)
    assert e.E.metric('test',h,p)==task['metrics']
    with np.load(r/'results/predictions.npz') as z:
        assert np.array_equal(p,z[f'DEV_frozen_R48_candidate_H{h}']);lo=z[f'DEV_frozen_R48_candidate_lower80_H{h}'];hi=z[f'DEV_frozen_R48_candidate_upper80_H{h}']
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and (lo<=hi).all()
    times=e.E.F.data.times[e.E.F.canon];curve=q/f'当前推荐_H{h}_历史滚动曲线.csv';daily=q/f'当前推荐_H{h}_每日平均负荷.csv'
    with curve.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model_name','PDU','cell','origin_day','target_day','slot_15min','raw_time','prediction_normalized','lower80','upper80','actual_normalized','evaluation_valid'])
        for i in range(57):
            for j,o in enumerate(e.E.origins('test',h)):
                for k in range(96*h):
                    day=o+k//96;slot=k%96
                    w.writerow([task['model_name'],str(e.E.d['names'][i]),str(e.E.d['cells'][i]),o,day+1,slot,times[i,day,slot],p[i,j,k],lo[i,j,k],hi[i,j,k],y[i,j,k] if m[i,j,k] else '',int(m[i,j,k])])
    with daily.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model_name','PDU','cell','origin_day','target_day','predicted_daily_mean_normalized','actual_daily_mean_normalized','complete96_valid'])
        for i in range(57):
            for j,o in enumerate(e.E.origins('test',h)):
                for k in range(h):
                    sl=slice(k*96,(k+1)*96);ok=bool(m[i,j,sl].all())
                    w.writerow([task['model_name'],str(e.E.d['names'][i]),str(e.E.d['cells'][i]),o,o+k+1,p[i,j,sl].mean(),y[i,j,sl].mean() if ok else '',int(ok)])
    rec=dict(all_passed=True,horizon_days=h,model_name=task['model_name'],metrics=task['metrics'],curve_rows=p.size,daily_rows=p.size//96,curve_sha256=sha(curve),daily_sha256=sha(daily),configuration_sha256=sha(q/'selected_configuration.json'),mode='audited_historical_cache_reconstruction',independent_external_validation=False)
    (q/f'export_H{h}.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=True,H=h,rows=p.size,curve_WAPE=task['metrics']['wape'],daily_WAPE=task['metrics']['daily_wape'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('horizon',type=int,choices=[1,7]);a=ap.parse_args();run(a.root,a.horizon)
