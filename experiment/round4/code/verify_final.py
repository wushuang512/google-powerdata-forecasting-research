import sys,json,hashlib,copy
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'round3/code'));sys.path.insert(0,str(R/'code'))
from residual import get_bank,xfeatures,online
from round2_lib import inputs
from lib import load,np,truth,metrics,DEV,TEST
from aligned import sample
import torch,pandas as pd
def equal(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return np.array_equal(a,b,equal_nan=True)
def main():
    d=load();bank=get_bank(['train','dev','test']);checks={};rng=np.random.default_rng(42)
    for o in [14,22,26,30]:
        altered=copy.deepcopy(d)
        for k in ['y','prod']:altered[k][:,o:]=rng.uniform(0,1,altered[k][:,o:].shape)
        for k in ['valid','pvalid']:altered[k][:,o:]=~altered[k][:,o:]
        checks[f'tf3_cellprod_input_causal_{o}']=equal(inputs(d,o,'cellprod'),inputs(altered,o,'cellprod'))
        checks[f'residual_features_causal_{o}']=np.array_equal(xfeatures(d,o,bank)[0],xfeatures(altered,o,bank)[0],equal_nan=True)
        checks[f'online_causal_{o}']=np.array_equal(online(d,bank,[o],1,.25),online(altered,bank,[o],1,.25))
    for cutoff in [14,22]:
        a=copy.deepcopy(d)
        for k in ['y','prod']:a[k][:,cutoff:]=999
        xx=sample(d,cutoff,np.random.default_rng(5),True);zz=sample(a,cutoff,np.random.default_rng(5),True)
        checks[f'aligned_training_cutoff_{cutoff}']=all(torch.allclose(x,z,equal_nan=True) for x,z in zip(xx,zz))
    for rd,files in [('round3',['selection','secondary']),('round4',['selection'])]:
        for f in files:checks[f'{rd}_{f}_hash']=hashlib.sha256((B/rd/f'results/{f}.json').read_text(encoding='utf-8').encode('utf-8')).hexdigest()==(B/rd/f'results/{f}.sha256').read_text()
    z=np.load(B/'round3/results/test_predictions.npz');expected=np.clip(.5*z['old_equal_mix']+.5*z['tf3_cellprod'],0,1);checks['fixed_mix_exact']=np.array_equal(expected,z['secondary_equal_mix']);checks['bounds']=bool((expected>=0).all() and (expected<=1).all())
    y,m=truth(d,TEST);saved=json.loads((B/'round3/results/summary.json').read_text())['scores']['secondary_equal_mix'];checks['metric_recomputed']=abs(metrics(y,expected,m)['wape']-saved['wape'])<1e-12
    rows=[]
    for i,n in enumerate(d['names']):
        for j,o in enumerate(TEST):
            for t in range(96):rows.append((n,d['cells'][i],o+1,t,y[i,j,t],expected[i,j,t],z['old_equal_mix'][i,j,t],bool(m[i,j,t])))
    pd.DataFrame(rows,columns=['series','cell','target_day','quarter_hour_slot','actual_util','best_predicted_util','old_predicted_util','valid_target']).to_csv(R/'results/forecasts_1day.csv',index=False)
    out=dict(checks={k:bool(v) for k,v in checks.items()},all_passed=all(checks.values()),scope='Implementation leakage checks, not a guarantee against model-selection overfitting; same month has been repeatedly examined')
    (R/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert out['all_passed']
if __name__=='__main__':main()
