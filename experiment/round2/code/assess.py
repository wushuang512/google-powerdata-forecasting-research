import json,hashlib
from round2_lib import *
from combine import apply_weights
import pandas as pd

def comparisons(y,m,p,ref,cells):
    den=(np.abs(y)*m).sum(-1);ea=(np.abs(y-ref)*m).sum(-1);eb=(np.abs(y-p)*m).sum(-1)
    per=100*(ea.sum(1)-eb.sum(1))/den.sum(1)
    day=100*np.mean((ea-eb)/den,axis=0)
    rng=np.random.default_rng(20260930);bday=[];btwo=[];groups=np.unique(cells)
    for _ in range(5000):
        ix=np.concatenate([np.arange(k,k+3) for k in rng.integers(0,7,3)])
        diff=100*(ea[:,ix].sum(1)-eb[:,ix].sum(1))/den[:,ix].sum(1)
        bday.append(diff.mean())
        ds=np.concatenate([np.where(cells==c)[0] for c in rng.choice(groups,len(groups),replace=True)])
        btwo.append(diff[ds].mean())
    return dict(difference_pp=float(per.mean()),relative_reduction_percent=float(per.mean()/metrics(y,ref,m)['wape']*100),domains_won=int((per>0).sum()),days_won=int((day>0).sum()),cells_won=int(sum(per[cells==c].mean()>0 for c in groups)),day_block_95=np.percentile(bday,[2.5,97.5]).tolist(),day_and_cell_95=np.percentile(btwo,[2.5,97.5]).tolist(),bootstrap_replicates=5000)

def main():
    d=load();s=json.loads((R/'results/selection.json').read_text());report={};bydomain=[];byday=[]
    # The frozen digest was computed from UTF-8 text with LF newlines.
    assert hashlib.sha256((R/'results/selection.json').read_text().encode('utf-8')).hexdigest()==(R/'results/selection.sha256').read_text()
    for h,origins,tag in [(1,TEST,'24'),(7,[22,23,24],'7')]:
        p=dict(np.load(R/f'results/final_chronos{tag}.npz'));p.update(dict(np.load(R/f'results/tirex_test{"7" if h==7 else ""}.npz')))
        p['old_equal_mix']=(p['zero_uni']+p['tirex_uni'])/2
        p['new_curve'],p['new_peak']=apply_weights(p,s,d,origins)
        p['last']=baseline(d,origins,h);p['week']=baseline(d,origins,h,'week');p['mean3']=baseline(d,origins,h,'mean3')
        old=dict(np.load(B/f'results/test{tag}_predictions.npz'))
        p['round1_primary']=old['ensemble_peak' if h==1 else 'ensemble7']
        y,m=truth(d,origins,h);scores={k:metrics(y,z,m) for k,z in p.items()}
        pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).sort_values('wape').to_csv(R/f'results/test{tag}_metrics.csv',index=False)
        np.savez_compressed(R/f'results/test{tag}_predictions.npz',**p)
        report[f'horizon{h}']=scores
        if h==1:
            report['comparison']={k:comparisons(y,m,p['new_curve'],p[k],d['cells']) for k in ['old_equal_mix','round1_primary','last']}
            for k,z in p.items():
                per=100*(np.abs(y-z)*m).sum((1,2))/(np.abs(y)*m).sum((1,2))
                bydomain.extend([dict(model=k,series=str(n),cell=str(c),wape=float(v)) for n,c,v in zip(d['names'],d['cells'],per)])
                byday.extend([dict(model=k,target_day=o+1,**metrics(y[:,j:j+1],z[:,j:j+1],m[:,j:j+1])) for j,o in enumerate(origins)])
        rec=[]
        for i in range(57):
            for j,o in enumerate(origins):
                for t in range(h*96):rec.append((str(d['names'][i]),str(d['cells'][i]),o,o+t//96+1,t%96,y[i,j,t],p['new_curve'][i,j,t],p['new_peak'][i,j,t],bool(m[i,j,t])))
        pd.DataFrame(rec,columns=['series','cell','origin_end_day','target_day','quarter_hour_slot','actual_util','curve_predicted_util','peak_predicted_util','valid_target']).to_csv(R/f'results/forecasts_{h}day.csv',index=False)
    pd.DataFrame(bydomain).to_csv(R/'results/per_domain.csv',index=False)
    pd.DataFrame(bydomain).groupby(['model','cell'],as_index=False).wape.mean().to_csv(R/'results/per_cell.csv',index=False)
    pd.DataFrame(byday).to_csv(R/'results/per_day.csv',index=False)
    comp=report['comparison']['old_equal_mix'];report['goal_achieved_exploratory']=bool(comp['relative_reduction_percent']>=5 and comp['day_and_cell_95'][0]>0)
    report['old_report_wape']=2.416;report['improvement_vs_old_report_percent']=(2.416-report['horizon1']['new_curve']['wape'])/2.416*100
    report['independent_confirmation']=False;report['seven_day_weights']='Transferred unchanged from 24h development; no 7day retuning'
    (R/'results/summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if not k.startswith('horizon')},indent=2),flush=True)

if __name__=='__main__':main()
