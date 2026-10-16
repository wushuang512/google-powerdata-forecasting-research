import hashlib
from lib import *
from assess import comparisons
import pandas as pd

def main():
    d=load();s=json.loads((R/'results/selection.json').read_text());extra=json.loads((R/'results/secondary.json').read_text());p={}
    for name in ['selection','secondary']:assert hashlib.sha256((R/f'results/{name}.json').read_bytes()).hexdigest()==(R/f'results/{name}.sha256').read_text()
    for kind in ['chronos','tirex']:p.update({kind+'_'+k:v for k,v in dict(np.load(R/f'results/{kind}_test.npz')).items()})
    for file in ['residual_test.npz','tf3_test.npz','hierarchy_chronos_test.npz']:p.update(dict(np.load(R/'results'/file)))
    p['old_equal_mix']=(p['chronos_native']+p['tirex_native'])/2
    p['new_curve']=np.clip(np.einsum('k,ksoh->soh',np.array(s['weights']),np.stack([p[k] for k in s['pool']])),0,1)
    p['secondary_equal_mix']=np.clip((p['old_equal_mix']+p[extra['single_model']])/2,0,1)
    p['round2_primary']=np.load(R2/'results/test24_predictions.npz')['new_curve'];p['last']=baseline(d,TEST)
    y,m=truth(d,TEST);scores={k:metrics(y,z,m) for k,z in p.items()}
    comps={k:comparisons(y,m,p[k],p['old_equal_mix'],d['cells']) for k in ['new_curve','secondary_equal_mix',extra['single_model']]}
    summary=dict(scores=scores,comparison_vs_old=comps,primary='new_curve',secondary='secondary_equal_mix',independent_confirmation=False,adaptive_same_month=True)
    (R/'results/summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(R/'results/test_predictions.npz',**p)
    pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).sort_values('wape').to_csv(R/'results/test_metrics.csv',index=False)
    groups=[];days=[]
    for k in ['old_equal_mix','new_curve','secondary_equal_mix',extra['single_model']]:
        per=100*(np.abs(y-p[k])*m).sum((1,2))/(np.abs(y)*m).sum((1,2))
        groups.extend([dict(model=k,series=str(n),cell=str(c),wape=float(a)) for n,c,a in zip(d['names'],d['cells'],per)])
        days.extend([dict(model=k,target_day=o+1,**metrics(y[:,j:j+1],p[k][:,j:j+1],m[:,j:j+1])) for j,o in enumerate(TEST)])
    pd.DataFrame(groups).to_csv(R/'results/per_domain.csv',index=False);pd.DataFrame(days).to_csv(R/'results/per_day.csv',index=False)
    print(pd.DataFrame([dict(model=k,**scores[k]) for k in ['old_equal_mix','new_curve','secondary_equal_mix',extra['single_model']]])[['model','wape','daily_wape','peak_mape']].to_string(index=False),flush=True)
    print(json.dumps(comps,indent=2),flush=True)

if __name__=='__main__':main()
