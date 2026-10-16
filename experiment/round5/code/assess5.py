from lib5 import *
from assess import comparisons
def interval_metrics(y,q,m):
    lo=q[:,:,:,0];hi=q[:,:,:,2];coverage=((y>=lo)&(y<=hi)&m).sum()/m.sum();width=(hi-lo)[m].mean();err=y[:,:,:,None]-q;qs=np.array([.1,.5,.9]);pin=(np.maximum(qs*err,(qs-1)*err)*m[:,:,:,None]).sum()/m.sum()/3
    score=(hi-lo)+10*(lo-y).clip(0)+10*(y-hi).clip(0)
    return dict(coverage=float(coverage),mean_width=float(width),pinball=float(pin),interval_score=float(score[m].mean()))
def main():
    d=load();s=json.loads((R/'results/selection.json').read_text());assert hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text();p=dict(np.load(R/'results/test_candidates.npz'));p['old_equal_mix']=np.load(B/'round3/results/test_predictions.npz')['old_equal_mix'];y,m=truth(d,TEST)
    scores={k:metrics(y,z,m) for k,z in p.items()};cmp={k:comparisons(y,m,z,p['anchor'],d['cells']) for k,z in p.items() if k!='anchor'}
    levels=np.stack([d['y'][:,o-7:o].mean((1,2)) for o in TEST],1)[:,:,None].clip(.05);c=s['residual_interval']['calibrated_halfwidth_relative_to_past_mean'];finalq=np.stack([p['selected']-c*levels,p['selected'],p['selected']+c*levels],-1).clip(0,1)
    fq=np.load(R/'results/test_quantiles.npz')['fusion'];fc=s['fusion_interval']['widening_relative_to_past_mean'];fqc=fq.copy();fqc[:,:,:,0]=(fq[:,:,:,0]-fc*levels).clip(0,1);fqc[:,:,:,2]=(fq[:,:,:,2]+fc*levels).clip(0,1)
    uncertainty={'fusion_raw':interval_metrics(y,fq,m),'fusion_calibrated':interval_metrics(y,fqc,m),'selected_residual_interval':interval_metrics(y,finalq,m)}
    actualw=np.stack([d['prod'][:,o] for o in TEST],1);wm=np.stack([d['pvalid'][:,o]&d['valid'][:,o] for o in TEST],1);wq=np.load(R/'results/test_proxy_quantiles.npz')['fusion'];domainmask=(actualw*wm).sum((1,2))>0
    proxy=dict(valid_domains=int(domainmask.sum()),valid_points=int(wm.sum()),mae=float(np.abs(actualw-wq[:,:,:,1])[wm].mean()),macro_wape=float((100*(np.abs(actualw-wq[:,:,:,1])*wm).sum((1,2))[domainmask]/(actualw*wm).sum((1,2))[domainmask]).mean()),interpretation='production power proxy; no actual compute-load forecasting accuracy is claimed')
    summary=dict(scores=scores,comparison_vs_previous_best=cmp,selected=s['selected'],uncertainty=uncertainty,production_proxy=proxy,independent_confirmation=False,overfitting_guarantee=False,large_gain_threshold_met=bool(cmp['selected']['relative_reduction_percent']>=5 and cmp['selected']['day_and_cell_95'][0]>0));save_json(R/'results/summary.json',summary);np.savez_compressed(R/'results/test_predictions.npz',**p);np.savez_compressed(R/'results/final_intervals.npz',selected=finalq,fusion_calibrated=fqc);pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).sort_values('wape').to_csv(R/'results/test_metrics.csv',index=False)
    domains=[];days=[];records=[]
    for k,z in p.items():
        per=100*(np.abs(y-z)*m).sum((1,2))/(np.abs(y)*m).sum((1,2));domains.extend([dict(model=k,series=n,cell=c,wape=float(a)) for n,c,a in zip(d['names'],d['cells'],per)]);days.extend([dict(model=k,target_day=o+1,**metrics(y[:,j:j+1],z[:,j:j+1],m[:,j:j+1])) for j,o in enumerate(TEST)])
    for i,n in enumerate(d['names']):
        for j,o in enumerate(TEST):
            for t in range(96):records.append((n,d['cells'][i],o+1,t,y[i,j,t],p['selected'][i,j,t],p['anchor'][i,j,t],p['fusion'][i,j,t],finalq[i,j,t,0],finalq[i,j,t,2],bool(m[i,j,t])))
    pd.DataFrame(domains).to_csv(R/'results/per_domain.csv',index=False);pd.DataFrame(domains).groupby(['model','cell'],as_index=False).wape.mean().to_csv(R/'results/per_cell.csv',index=False);pd.DataFrame(days).to_csv(R/'results/per_day.csv',index=False);pd.DataFrame(records,columns=['series','cell','target_day','quarter_hour_slot','actual_util','selected_util','previous_best_util','fusion_util','lower80_empirical','upper80_empirical','valid_target']).to_csv(R/'results/forecasts_1day.csv',index=False);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
