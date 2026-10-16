"""Audited historical prediction reconstruction for the user-adopted R47 method."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def initialize(root):
    global T,R,Q,C,E
    T=Path(root);R=T/'experiment/round47';Q=R/'deployment';C=read(Q/'selected_configuration.json')
    sys.path.insert(0,str(R/'code'))
    import hdsf_shape_experiment as e
    E=e;E.initialize(T)
    assert sha(R/'results/selection.json')==C['frozen_research_selection_sha256']
    assert C['decision_type']=='explicit_user_adoption_after_results'
    for name,digest in C['bindings'].items():assert sha(T/name)==digest,name
def point(phase,h):
    task=C['tasks'][str(h)]
    return E.adapt(E.base(phase,h),E.route_delta(task['route'],phase,h),task['level_alpha'],task['shape_alpha'])
def extra_peak(h,p):
    y,m=E.common.truth(E.d,E.origins('test',h),h)
    yd,pd,md=y.reshape(57,-1,96),p.reshape(57,-1,96),m.reshape(57,-1,96)
    ok=md.any(-1);yp=np.where(md,yd,-np.inf).max(-1);pp=np.where(md,pd,-np.inf).max(-1)
    return float(100*(pp[ok]<yp[ok]).mean())
def export(h):
    p=point('test',h);expected=read(R/'results/summary.json')['scores'][str(h)]['frozen_candidate']['metrics']
    assert E.metric('test',h,p)==expected
    with np.load(R/'results/predictions.npz') as z:
        assert np.array_equal(p,z[f'frozen_candidate_H{h}'])
        lo,hi=z[f'frozen_candidate_lower80_H{h}'].copy(),z[f'frozen_candidate_upper80_H{h}'].copy()
    y,m=E.common.truth(E.d,E.origins('test',h),h)
    times=E.F.data.times[E.F.canon]
    fp=Q/f'当前推荐_H{h}_历史滚动曲线.csv';dp=Q/f'当前推荐_H{h}_每日平均负荷.csv'
    with fp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model_name','PDU','cell','origin_day','target_day','slot_15min','raw_time','prediction_normalized','lower80','upper80','actual_normalized','evaluation_valid'])
        for i in range(57):
            for j,o in enumerate(E.origins('test',h)):
                for k in range(96*h):
                    day=o+k//96;slot=k%96
                    w.writerow([C['tasks'][str(h)]['model_name'],str(E.d['names'][i]),str(E.d['cells'][i]),o,day+1,slot,
                        times[i,day,slot],p[i,j,k],lo[i,j,k],hi[i,j,k],y[i,j,k] if m[i,j,k] else'',int(m[i,j,k])])
    with dp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model_name','PDU','cell','origin_day','target_day','predicted_daily_mean_normalized','actual_daily_mean_normalized','complete96_valid'])
        for i in range(57):
            for j,o in enumerate(E.origins('test',h)):
                for k in range(h):
                    sl=slice(k*96,(k+1)*96);ok=bool(m[i,j,sl].all())
                    w.writerow([C['tasks'][str(h)]['model_name'],str(E.d['names'][i]),str(E.d['cells'][i]),o,o+k+1,
                                p[i,j,sl].mean(),y[i,j,sl].mean() if ok else'',int(ok)])
    result=dict(horizon_days=h,model_name=C['tasks'][str(h)]['model_name'],metrics=expected,
        peak_underestimate_percent=extra_peak(h,p),curve_csv=str(fp),daily_csv=str(dp),curve_rows=int(p.size),daily_rows=57*len(E.origins('test',h))*h,
        curve_sha256=sha(fp),daily_sha256=sha(dp),selected_configuration_sha256=sha(Q/'selected_configuration.json'),
        mode='audited_historical_cache_reconstruction',native_model_replays_verified_in_R47=True,
        calendar_time_note='保留原始数字time及D1至D31序列，不将原始time擅自转换为绝对时区。',
        prediction_unit='normalized power utilization, not kW',independent_confirmation=False)
    (Q/f'export_H{h}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(H=h,curve_WAPE=expected['wape'],daily_WAPE=expected['daily_wape'],peak_MAPE=expected['peak_mape'],
                         curve_rows=p.size,daily_rows=result['daily_rows'],curve=str(fp),daily=str(dp)),ensure_ascii=True))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('horizon',type=int,choices=[1,7]);a=ap.parse_args();initialize(a.root);export(a.horizon)
