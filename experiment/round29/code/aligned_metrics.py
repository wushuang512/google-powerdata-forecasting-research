"""Versioned 9.28 metrics. Historical evaluator is never modified."""
import numpy as np
def evaluate(common,d,origins,p,H):
 y,m=common.truth(d,origins,H);e=abs(y-p);ape=100*e[m]/np.maximum(abs(y[m]),1e-8)
 per=100*(e*m).sum((1,2))/(abs(y)*m).sum((1,2));a=common.metrics(y,p,m)
 yd=y.reshape(57,-1,96);pd=p.reshape(57,-1,96);md=m.reshape(57,-1,96);ok=md.all(-1);ya=yd.mean(-1);pa=pd.mean(-1);de=pa-ya
 a['daily_wape_valid_slots_supplement']=a['daily_wape'];a['daily_wape']=float(np.mean(100*(abs(de)*ok).sum(1)/(abs(ya)*ok).sum(1)))
 a.update(rmse=float(np.sqrt((e[m]**2).mean())),hit2_percent=float(100*(ape<=2).mean()),hit5_percent=float(100*(ape<=5).mean()),absolute_bias_percent=float(abs(100*(p-y)[m].sum()/y[m].sum())),daily_mae=float(abs(de)[ok].mean()),daily_rmse=float(np.sqrt((de[ok]**2).mean())),daily_hit2_percent=float(100*(100*abs(de[ok])/abs(ya[ok])<=2).mean()),daily_hit5_percent=float(100*(100*abs(de[ok])/abs(ya[ok])<=5).mean()),valid_daily_targets=int(ok.sum()),daily_target_mask='all96valid',cella5_wape=float(per[d['cells']=='a'].mean()),other52_wape=float(per[d['cells']!='a'].mean()))
 b=common.baseline(d,origins,H,'last' if H==1 else 'mean3');be=abs(y-b);bp=100*(be*m).sum((1,2))/(abs(y)*m).sum((1,2));bd=b.reshape(57,-1,96).mean(-1);bw=float(bp.mean());bdw=float(np.mean(100*(abs(ya-bd)*ok).sum(1)/(abs(ya)*ok).sum(1)))
 a.update(baseline_wape=bw,baseline_daily_wape=bdw,relative_curve_error_reduction_percent=100*(1-a['wape']/bw),relative_daily_error_reduction_percent=100*(1-a['daily_wape']/bdw),improved_pdus=int((per<bp-1e-12).sum()),improved_cells=int(sum(per[d['cells']==c].mean()<bp[d['cells']==c].mean()-1e-12 for c in np.unique(d['cells']))),improved_origins=int(sum(np.mean(100*(e[:,j]*m[:,j]).sum(-1)/(abs(y[:,j])*m[:,j]).sum(-1))<np.mean(100*(be[:,j]*m[:,j]).sum(-1)/(abs(y[:,j])*m[:,j]).sum(-1))-1e-12 for j in range(len(origins)))))
 return a
