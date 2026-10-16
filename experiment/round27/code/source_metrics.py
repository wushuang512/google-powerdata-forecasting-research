"""Source-aligned metrics, preserving the original 9.28 valid-slot and macro-PDU definitions."""
import numpy as np
def evaluate(common,d,origins,p,H):
 y,m=common.truth(d,origins,H);a=common.metrics(y,p,m);err=p-y;ape=100*abs(err[m])/np.maximum(abs(y[m]),1e-8)
 a.update(rmse=float(np.sqrt((err[m]**2).mean())),hit2_percent=float(100*(ape<=2).mean()),hit5_percent=float(100*(ape<=5).mean()),absolute_bias_percent=float(abs(100*err[m].sum()/y[m].sum())))
 per=100*(abs(err)*m).sum((1,2))/(abs(y)*m).sum((1,2));a['cella5_wape']=float(per[d['cells']=='a'].mean());a['other52_wape']=float(per[d['cells']!='a'].mean())
 yd=y.reshape(57,-1,96);pd=p.reshape(57,-1,96);md=m.reshape(57,-1,96);cnt=md.sum(-1);dv=cnt>0;ya=(yd*md).sum(-1)/cnt.clip(1);pa=(pd*md).sum(-1)/cnt.clip(1);de=pa-ya
 a.update(daily_mae=float(abs(de)[dv].mean()),daily_rmse=float(np.sqrt((de[dv]**2).mean())),daily_hit2_percent=float(100*(100*abs(de[dv])/abs(ya[dv])<=2).mean()),daily_hit5_percent=float(100*(100*abs(de[dv])/abs(ya[dv])<=5).mean()))
 yp=np.where(md,yd,-np.inf).max(-1);pp=np.where(md,pd,-np.inf).max(-1)
 a['peak_underestimate_percent']=float(100*(pp[dv]<yp[dv]).mean())
 b=common.baseline(d,origins,H,kind='last' if H==1 else 'mean3');bs=common.metrics(y,b,m)
 a.update(source_baseline='前一天同刻' if H==1 else '前三天同刻均值',baseline_wape=bs['wape'],relative_curve_error_reduction_percent=float(100*(1-a['wape']/bs['wape'])),relative_daily_error_reduction_percent=float(100*(1-a['daily_wape']/bs['daily_wape'])),curve_gain_pp=bs['wape']-a['wape'])
 return a
def interval_metrics(y,p,m,lo,hi):
 assert y.shape==p.shape==m.shape==lo.shape==hi.shape and np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all()
 width=hi-lo;winkler=width+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)
 return dict(nominal_percent=80,coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),mean_width_normalized=float(width[m].mean()),winkler80=float(winkler[m].mean()),below_percent=float(100*(y<lo)[m].mean()),above_percent=float(100*(y>hi)[m].mean()),valid_points=int(m.sum()))

