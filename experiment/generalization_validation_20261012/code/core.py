"""Portable implementation of the frozen evaluation formulas, not a replacement."""
from pathlib import Path
import csv,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def np_save(path,**x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**x)
    tmp.replace(p)
def export(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def truth(y,m,origins,h):return np.stack([y[:,o:o+h].reshape(len(y),-1) for o in origins],1),np.stack([m[:,o:o+h].reshape(len(y),-1) for o in origins],1)
def baseline(x,origins,h,kind):
    rows=[]
    for o in origins:
        if kind=='last':p=np.tile(x[:,o-1:o],(1,h,1))
        elif kind=='week':p=x[:,o-7:o-7+h]
        elif kind=='mean3':p=np.tile(x[:,o-3:o].mean(1)[:,None],(1,h,1))
        else:raise ValueError(kind)
        rows.append(p.reshape(len(x),-1))
    return np.stack(rows,1)
def metric(y,p,m,points_per_day=96,base=None):
    assert y.shape==p.shape==m.shape and np.isfinite(p).all()
    e=abs(y-p);den=(abs(y)*m).sum((1,2));okden=den>0
    if not okden.any():
        return dict(wape=None,mae=None,rmse=None,p95_ape=None,daily_wape=None,peak_mape=None,
            peak_time_mae_minutes=None,valid_points=int(m.sum()),valid_daily_targets=0,status='undefined: no nonzero valid target denominator')
    ape_mask=m&(abs(y)>0);ape=100*e[ape_mask]/abs(y[ape_mask]);ppd=points_per_day
    yd,pd,md=[a.reshape(len(y),-1,ppd) for a in [y,p,m]]
    complete=md.all(-1);ya,pa=yd.mean(-1),pd.mean(-1);de=pa-ya
    dailyden=(abs(ya)*complete).sum(1);dailyok=dailyden>0
    count=md.sum(-1);some=count>0
    ys=(yd*md).sum(-1)/count.clip(1);ps=(pd*md).sum(-1)/count.clip(1)
    sd=(abs(ys)*some).sum(1);ss=sd>0
    peak_y=np.where(md,yd,-np.inf).max(-1);peak_p=np.where(md,pd,-np.inf).max(-1)
    peak_ok=some&(peak_y>0)
    dap=100*abs(de[complete])/abs(ya[complete])
    out=dict(wape=float(np.mean(100*(e*m).sum((1,2))[okden]/den[okden])),mae=float(e[m].mean()),
        rmse=float(np.sqrt(np.mean(e[m]**2))),p95_ape=float(np.percentile(ape,95)),
        daily_wape=float(np.mean(100*(abs(de)*complete).sum(1)[dailyok]/dailyden[dailyok])) if dailyok.any() else None,
        daily_wape_valid_slots_supplement=float(np.mean(100*(abs(ps-ys)*some).sum(1)[ss]/sd[ss])),
        hit2_percent=float(100*(ape<=2).mean()),hit5_percent=float(100*(ape<=5).mean()),
        absolute_bias_percent=float(abs(100*(p-y)[m].sum()/y[m].sum())),
        daily_mae=float(abs(de)[complete].mean()) if complete.any() else None,
        daily_rmse=float(np.sqrt((de[complete]**2).mean())) if complete.any() else None,
        daily_hit2_percent=float(100*(dap<=2).mean()) if complete.any() else None,
        daily_hit5_percent=float(100*(dap<=5).mean()) if complete.any() else None,
        valid_points=int(m.sum()),valid_daily_targets=int(complete.sum()),undefined_zero_APE_points=int((m&~ape_mask).sum()),
        daily_target_mask='all96valid' if ppd==96 else 'original_daily_observation',
        peak_mape=float(100*(abs(peak_y-peak_p)/np.maximum(peak_y,1e-8))[peak_ok].mean()) if ppd==96 else None,
        peak_time_mae_minutes=float(15*abs(np.argmax(np.where(md,yd,-np.inf),-1)-np.argmax(np.where(md,pd,-np.inf),-1))[some].mean()) if ppd==96 else None,
        peak_underestimate_percent=float(100*(peak_p[some]<peak_y[some]).mean()) if ppd==96 else None)
    if base is not None:
        b=metric(y,base,m,ppd)
        out.update(baseline_wape=b['wape'],baseline_daily_wape=b['daily_wape'],
            relative_curve_error_reduction_percent=float(100*(1-out['wape']/b['wape'])),
            relative_daily_error_reduction_percent=float(100*(1-out['daily_wape']/b['daily_wape'])) if out['daily_wape'] is not None else None,
            improved_pdus=int(((e*m).sum((1,2))< (abs(y-base)*m).sum((1,2))-1e-12).sum()))
    return out
def interval(y,m,p,qlo,qhi):
    lo=np.maximum(0,p+qlo[:,None,None]);hi=np.maximum(lo,p+qhi[:,None,None]);w=hi-lo
    return dict(nominal_percent=80,coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),
        mean_width=float(w[m].mean()),winkler80=float((w+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0))[m].mean()),
        below_percent=float(100*(y<lo)[m].mean()),above_percent=float(100*(y>hi)[m].mean())),lo,hi
def simplex_tensor(P,weights):
    index=np.minimum(np.arange(P.shape[2])//48,len(weights)-1)
    return np.einsum('solk,lk->sol',P,weights[index],optimize=True)
def original_weights(root,h):
    c=read(Path(root)/'experiment/round44/results/selection.json')['configs']['24h' if h==1 else'7d']
    k=str(c['segments'][0]);a=np.array(c['component_a'][k]);b=np.array(c['component_b'][k])
    return c['methods'],(1-c['mix'])*np.column_stack([a,np.zeros(len(a))])+c['mix']*b
def protocol_check(folder):
    p=read(folder/'docs/protocol.json')
    for f,d in p['code_sha256'].items():assert sha(folder/'code'/f)==d,f
    return p
def parity(root,out):
    sys.path[:0]=[str(Path(root)/'experiment/code'),str(Path(root)/'experiment/round29/code')]
    import common
    from aligned_metrics import evaluate
    d=common.load();checks=[]
    with np.load(Path(root)/'experiment/round48/results/predictions.npz') as z:
        for h,oo in [(1,list(range(22,31))),(7,[22,23,24])]:
            p=z[f'DEV_frozen_R48_candidate_H{h}'];y,m=common.truth(d,oo,h);b=common.baseline(d,oo,h,'last' if h==1 else'mean3')
            a=metric(y,p,m,96,b);ref=evaluate(common,d,oo,p,h)
            for key in set(a)&set(ref):
                v=a[key];w=ref[key];good=v==w if isinstance(v,(str,int)) else bool(np.isclose(v,w,rtol=0,atol=1e-11))
                checks.append(dict(horizon=h,key=key,passed=good,portable=v,reference=w));assert good,key
    save(out,dict(all_passed=True,checks=checks,count=len(checks),historical_evaluator_modified=False,utc=utc()))
