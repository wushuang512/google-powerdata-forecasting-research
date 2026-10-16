import copy
from lib5 import *
from train_fusion import training_data
from calibration import bank,features
from round2_lib import inputs
def eq(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(eq(x,z) for x,z in zip(a,b))
    return np.array_equal(a,b,equal_nan=True)
def main():
    d=load();checks={};b=bank(['train','dev','test']);rng=np.random.default_rng(77)
    for o in [14,22,26,30]:
        altered=copy.deepcopy(d)
        for k in ['y','prod']:altered[k][:,o:]=rng.uniform(0,1,altered[k][:,o:].shape)
        for k in ['valid','pvalid']:altered[k][:,o:]=~altered[k][:,o:]
        checks[f'fusion_history_causal_{o}']=all(np.array_equal(x,z) for x,z in zip(history(d,[o*96]),history(altered,[o*96])))
        checks[f'tf3_input_causal_{o}']=eq(inputs(d,o,'cellprod'),inputs(altered,o,'cellprod'))
        newer=copy.deepcopy(b)
        for k in newer:
            if k>o:newer[k]=rng.uniform(0,1,newer[k].shape)
        checks[f'calibration_features_causal_{o}']=np.array_equal(features(d,o,b)[0],features(altered,o,newer)[0])
    for cutoff in [14,22]:
        altered=copy.deepcopy(d)
        for k in ['y','prod']:altered[k][:,cutoff:]=999
        for k in ['valid','pvalid']:altered[k][:,cutoff:]=~altered[k][:,cutoff:]
        first=training_data(d,cutoff);second=training_data(altered,cutoff);checks[f'fusion_training_cutoff_{cutoff}']=all(torch.equal(x,z) for x,z in zip(first,second));del first,second;torch.cuda.empty_cache()
    s=json.loads((R/'results/selection.json').read_text());checks['frozen_hash']=hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text();p=dict(np.load(R/'results/test_candidates.npz'));pick=s['selected'];expected=p['anchor'] if pick['alpha']==0 else np.clip((1-pick['alpha'])*p['anchor']+pick['alpha']*p[pick['family']],0,1);checks['mixture_exact']=np.array_equal(expected,p['selected'])
    pq=np.load(R/'results/test_quantiles.npz')['fusion'];wq=np.load(R/'results/test_proxy_quantiles.npz')['fusion'];checks['finite_quantiles']=bool(np.isfinite(pq).all() and np.isfinite(wq).all());checks['capacity_bounds']=bool((pq>=0).all() and (pq<=1).all());checks['quantiles_monotone']=bool((np.diff(pq,axis=-1)>=-1e-7).all() and (np.diff(wq,axis=-1)>=-1e-7).all());checks['proxy_below_total']=bool((wq<=pq+1e-7).all());checks['two_seed_mean_exact']=np.allclose(p['fusion'],(p['fusion_seed20261001']+p['fusion_seed20261002'])/2,rtol=0,atol=1e-7)
    y,m=truth(d,TEST);score=json.loads((R/'results/summary.json').read_text())['scores']['selected'];checks['metric_recomputed']=abs(metrics(y,expected,m)['wape']-score['wape'])<1e-12
    from finetune_tf3 import load_delta,draw
    chosen=s['chosen']['tf3'];f=load_delta(R/'models'/f'{chosen}_final.pt')
    for p0 in f.model.parameters():p0.requires_grad_(False)
    for p0 in f.model.output_head.parameters():p0.requires_grad_(True)
    target,cov,_,_=draw(d,14,np.random.default_rng(11));official=f.model.decode(target,horizon=96,past_only_covariates=cov);adapted=type(f.model).decode.__wrapped__(f.model,target,horizon=96,past_only_covariates=cov);checks['adaptation_forward_exact']=torch.equal(official,adapted);adapted.sum().backward();checks['gradient_finite_after_fix']=all(p0.grad is not None and torch.isfinite(p0.grad).all().item() for p0 in f.model.output_head.parameters())
    for cutoff in [14,22]:
        a=copy.deepcopy(d)
        for k in ['y','prod']:a[k][:,cutoff:]=999
        x=draw(d,cutoff,np.random.default_rng(123));z=draw(a,cutoff,np.random.default_rng(123));checks[f'tf3_training_cutoff_{cutoff}']=all(v is w if v is None else torch.equal(v,w) for v,w in zip(x,z))
    out=dict(checks={k:bool(v) for k,v in checks.items()},all_passed=all(checks.values()),scope='Implementation checks only; same-month repeated selection overfitting is not ruled out',numerical_fix='Stop gradient through iterative CPM RevIN statistics; forward output unchanged; initial failed run retained')
    save_json(R/'results/verification.json',out);print(json.dumps(out,indent=2),flush=True);assert out['all_passed']
if __name__=='__main__':main()
