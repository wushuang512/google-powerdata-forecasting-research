import json,hashlib
from round2_lib import *
from combine import apply_weights

def equal(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,z) for x,z in zip(a,b))
    return np.array_equal(a,b,equal_nan=True)

def main():
    d=load();alt={k:v.copy() for k,v in d.items()}
    alt['y'][:,14:]=.123;alt['prod'][:,14:]=.987;alt['valid'][:,14:]=False;alt['pvalid'][:,14:]=False
    kinds=['uni','cov','calendar','cov_calendar','cell','cellprod','all','peer','peerprod','parts']
    for kind in kinds:
        for full in [False,True]:assert equal(inputs(d,14,kind,full=full),inputs(alt,14,kind,full=full)),kind
    rng=np.random.default_rng(31);t,p,r=project_components(*rng.uniform(-1,2,(3,10000)))
    assert np.max(np.abs(t-p-r))<1e-6 and np.min(p)>=0 and np.min(r)>=0 and np.max(t)<=1+1e-6
    s=json.loads((R/'results/selection.json').read_text());w=np.array(s['weights'])
    assert (w>=0).all() and np.allclose(w.sum(0),1)
    assert hashlib.sha256((R/'results/selection.json').read_text().encode('utf-8')).hexdigest()==(R/'results/selection.sha256').read_text()
    pred=dict(np.load(R/'results/test24_predictions.npz'));curve,peak=apply_weights(pred,s,d,TEST)
    assert np.array_equal(curve,pred['new_curve']) and np.array_equal(peak,pred['new_peak'])
    assert np.isfinite(curve).all() and curve.min()>=0 and curve.max()<=1
    y,m=truth(d,TEST);score=metrics(y,curve,m)['wape'];summary=json.loads((R/'results/summary.json').read_text())
    assert abs(score-summary['horizon1']['new_curve']['wape'])<1e-10
    report=dict(future_data_perturbation_passed=kinds,training_and_inference_inputs_checked=True,physical_projection_random_cases=10000,nonnegative_simplex_weights=True,frozen_selection_hash_verified=True,saved_prediction_recombination_exact=True,metrics_recomputed=True,capacity_bounds_passed=True,independent_test_claim=False)
    (R/'results/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
