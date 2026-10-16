"""Post-evaluation fixed ablation; no reselection or weight search permitted."""
from pathlib import Path
import importlib.util, sys, json, datetime
import joblib, numpy as np
sys.dont_write_bytecode=True
T=Path(sys.argv[1]);R=T/'experiment/round45';Q=R/'results';A=T/'experiment/round44'
spec=importlib.util.spec_from_file_location('round45core',R/'code/hdsf_next_experiment.py');x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);x.initialize(T)
assert (Q/'selection.sha256').read_text().strip()==x.sha(Q/'selection.json')
cfg=x.source_selection['configs']['24h'];n=2;mix=cfg['mix']
weights=(1-mix)*np.column_stack([np.array(cfg['component_a']['2']),np.zeros(n)])+mix*np.array(cfg['component_b']['2'])
ids=np.minimum(np.arange(96)//48,1)
def predict(P):return np.einsum('olsm,lm->ols',P,weights[ids],optimize=True).transpose(2,0,1)[x.canonical_index]
rows=[];arrays={};replays=[]
for phase,oo,cut in [('dev',[14,16,18,20],14),('test',list(range(22,31)),22)]:
    original,methods=x.members(1,phase);p=predict(original)
    with np.load(A/'results/predictions.npz') as z:stored=z[('dev_' if phase=='dev' else '')+'hdsf_H1']
    difference=float(abs(p-stored).max());assert difference<1e-12
    replays.append({'phase':phase,'original_frozen_weight_replay_max_difference':difference})
    rows.append({'phase':phase,'model':'原HDSF','original_weights_fixed':True,**x.evaluate(x.common,x.d,oo,stored,1)})
    for name in ['dropD1','regularized','direct7']:
        model=joblib.load(R/'models'/f'{name}_cut{cut}.joblib')
        local=np.stack([x.local_predict(model,x.data,o,1) for o in oo],axis=1)
        P=original.copy();P[...,-1]=local.transpose(1,2,0)
        result=predict(P);arrays[f'{phase}_{name}']=result
        score=x.evaluate(x.common,x.d,oo,result,1)
        rows.append({'phase':phase,'model':x.NAMES[name]+'＋原HDSF冻结权重','member':name,'original_weights_fixed':True,
                     'retrospective_fixed_ablation_not_selection':True,**score})
        del model
x.export(Q/'固定权重底层重训练对照.csv',rows)
np.savez_compressed(Q/'fixed_refit_ablation_predictions.npz',**arrays)
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'original_replays':replays,
     'purpose':'Separate local refit effect from changed fusionobjective/geometry; fixed source mixandweights, no search.',
     'retrospective':True,'selection_changed':False,'recommended_scheme_changed':False,
     'independent_confirmation':False,'selection_sha256':x.sha(Q/'selection.json'),
     'source_selection_sha256':x.sha(A/'results/selection.json')}
x.save(Q/'fixed_refit_ablation.json',out)
print(json.dumps({'frozen_weight_replay':replays,'scores':[{k:r[k] for k in ['phase','model','wape','daily_wape','peak_mape']} for r in rows]},ensure_ascii=True))
