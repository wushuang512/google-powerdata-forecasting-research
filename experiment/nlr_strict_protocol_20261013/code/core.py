from pathlib import Path
import csv,datetime,hashlib,json
import numpy as np
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def np_save(filename,**x):
    p=Path(filename);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**x)
    tmp.replace(p)
def export(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def metrics(y,p,threshold=1e-8,ppd=96):
    y=np.asarray(y,float);p=np.asarray(p,float);assert y.shape==p.shape and y.ndim==2 and np.isfinite(y).all() and np.isfinite(p).all()
    e=abs(y-p);keep=abs(y)>threshold;ape=np.divide(100*e,abs(y),out=np.zeros_like(y),where=keep)
    mn=keep.sum(1);om=np.divide((ape*keep).sum(1),mn,out=np.full(len(y),np.nan),where=mn>0)
    den=abs(y).sum();good=np.isfinite(om)
    d=dict(curve_mape_percent=float(om[good].mean()) if good.any() else None,curve_mape_pooled_percent=float(ape[keep].mean()) if keep.any() else None,
      curve_rmse=float(np.sqrt(np.mean((y-p)**2))),curve_wape_percent=float(100*e.sum()/den) if den>0 else None,
      curve_mae=float(e.mean()),mape_n=int(keep.sum()),mape_excluded=int((~keep).sum()),mape_origin_n=int(good.sum()),mape_origin_excluded=int((~good).sum()),
      origins=len(y),points=int(y.size),p95_ape_percent=float(np.percentile(ape[keep],95)) if keep.any() else None,
      hit2_percent=float(100*(ape[keep]<=2).mean()) if keep.any() else None,hit5_percent=float(100*(ape[keep]<=5).mean()) if keep.any() else None,
      absolute_bias_percent=float(100*abs((p-y).sum())/den) if den>0 else None)
    yd=y.reshape(len(y),-1,ppd);pd=p.reshape(len(p),-1,ppd);ya=yd.mean(-1);pa=pd.mean(-1);dk=abs(ya)>threshold;da=100*abs(ya-pa)/np.maximum(abs(ya),threshold)
    nd=dk.sum(1);dm=np.divide((da*dk).sum(1),nd,out=np.full(len(y),np.nan),where=nd>0);valid=np.isfinite(dm);dd=abs(ya).sum()
    d.update(daily_mape_percent=float(dm[valid].mean()) if valid.any() else None,daily_rmse=float(np.sqrt(np.mean((ya-pa)**2))),daily_wape_percent=float(100*abs(ya-pa).sum()/dd) if dd>0 else None,daily_mae=float(abs(ya-pa).mean()),daily_mape_n=int(dk.sum()),daily_mape_excluded=int((~dk).sum()),daily_hit2_percent=float(100*(da[dk]<=2).mean()) if dk.any() else None,daily_hit5_percent=float(100*(da[dk]<=5).mean()) if dk.any() else None)
    ym=yd.max(-1);pm=pd.max(-1);pk=abs(ym)>threshold
    d.update(peak_mape_percent=float((100*abs(ym-pm)/np.maximum(abs(ym),threshold))[pk].mean()) if pk.any() else None,
      peak_time_mae_minutes=float(15*abs(yd.argmax(-1)-pd.argmax(-1)).mean()),peak_underestimate_percent=float(100*(pm<ym).mean()))
    return d
def add_baseline(d,b):
    for key in ['curve_mape_percent','curve_rmse','curve_wape_percent','daily_mape_percent','daily_rmse','daily_wape_percent']:
        d['relative_'+key+'_reduction_percent']=float(100*(1-d[key]/b[key])) if d[key] is not None and b[key] is not None and b[key]>0 else None
    return d
