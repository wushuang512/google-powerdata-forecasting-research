import sys,json,hashlib,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(B/'round3/code'))
from aligned import *
from assess import comparisons
def main():
    s=json.loads((R/'results/selection.json').read_text());assert hashlib.sha256((R/'results/selection.json').read_text(encoding='utf-8').encode('utf-8')).hexdigest()==(R/'results/selection.sha256').read_text()
    d=load();p=dict(np.load(B/'round3/results/test_predictions.npz'));pick=s['selected'];anchor=p['secondary_equal_mix'];out={'old_equal_mix':p['old_equal_mix'],'round3_primary':p['new_curve'],'round3_secondary':anchor}
    if pick['family']=='aligned':
        name,st=pick['model'].rsplit('_s',1);cfg=CONFIGS[name];teacher=base_chronos();teacher.model.eval()
        for x in teacher.model.parameters():x.requires_grad_(False)
        pipe,hist=train_one(d,22,name,cfg,teacher,steps=int(st));save_delta(pipe,R/'models'/f'{pick["model"]}_d22',cfg,22,int(st));out[pick['model']]=np.clip(predict(pipe,d,TEST,cfg['kind']),0,1)
        (R/'results/final_training.json').write_text(json.dumps(hist,indent=2))
    elif pick['family']=='tf25':out[pick['model']]=np.load(R/'results/tf25_test.npz')[pick['model']]
    out['round4_selected']=anchor if pick['alpha']==0 else (1-pick['alpha'])*anchor+pick['alpha']*out[pick['model']]
    # A separately preregistered TimesFM2.5 family winner is a descriptive baseline only.
    if (R/'results/tf25_test.npz').exists():out.update(dict(np.load(R/'results/tf25_test.npz')))
    y,m=truth(d,TEST);scores={k:metrics(y,z,m) for k,z in out.items()};comp={k:comparisons(y,m,z,out['old_equal_mix'],d['cells']) for k,z in out.items() if k!='old_equal_mix'}
    (R/'results/summary.json').write_text(json.dumps(dict(scores=scores,comparison_vs_old=comp,selected=pick,independent_confirmation=False),indent=2));np.savez_compressed(R/'results/test_predictions.npz',**out)
    pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).to_csv(R/'results/test_metrics.csv',index=False);print(json.dumps(scores,indent=2),flush=True)
if __name__=='__main__':main()
