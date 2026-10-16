"""Check what was trained, frozen and actually scored in the two fine-tuning runs."""
import sys,json,hashlib,pickle
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,truth,metrics,save_json
from calibration import bank,features

def main():
    d=load();checks={};details={}
    for round in ['round8','round8b']:
        r=B/round;s=json.loads((r/'results/selection.json').read_text());p=np.load(r/'results/test_predictions.npz');y,m=truth(d,list(range(22,31)));q=json.loads((r/'results/summary.json').read_text());checks[round+'_selection_hash']=hashlib.sha256((r/'results/selection.json').read_bytes()).hexdigest()==(r/'results/selection.sha256').read_text();checks[round+'_protocol_hash']=hashlib.sha256((r/'docs/protocol.json').read_bytes()).hexdigest()==s['protocol_sha256'];checks[round+'_score_reproduced']=abs(metrics(y,p['selected'],m)['wape']-q['scores']['selected']['wape'])<1e-10;checks[round+'_finite_power_bounds']=all(np.isfinite(p[k]).all() and p[k].min()>=0 and p[k].max()<=1 for k in p.files);checks[round+'_chosen_blend_exact']=bool(np.array_equal(p['selected'],np.clip((1-s['chosen']['alpha'])*p['prior']+s['chosen']['alpha']*p['candidate' if round=='round8b' else 'finetuned'],0,1)));details[round]=q['scores']['selected']
    for o in [14,22,30]:
        dd={k:v.copy() for k,v in d.items()};dd['y'][:,o:]=999;dd['prod'][:,o:]=999;dd['valid'][:,o:]=~dd['valid'][:,o:];dd['pvalid'][:,o:]=~dd['pvalid'][:,o:];b=bank(['train','dev','test']);checks[f'calibration_features_causal_{o}']=bool(np.array_equal(features(d,o,b)[0],features(dd,o,b)[0]))
    s=json.loads((B/'round8b/results/selection.json').read_text());saved=torch.load(B/'round8b/models'/f'{s["config"]}_final.pt',weights_only=True,map_location='cpu');checks['foundation_final_labels_end_may22']=saved['cutoff']==22;checks['foundation_final_finite_weights']=all(torch.isfinite(p).all().item() for p in saved['state'].values());checks['foundation_changed_only_output_head_and_last_layer']=all(k.startswith('output_head.') or k.startswith('transformer_stack.layers.11.') for k in saved['state'])
    # Layer index is read from the saved delta rather than assumed if architecture differs.
    layerids={k.split('.')[2] for k in saved['state'] if k.startswith('transformer_stack.layers.')};checks['foundation_changed_only_output_head_and_last_layer']=len(layerids)==1 and all(k.startswith('output_head.') or k.startswith('transformer_stack.layers.') for k in saved['state']);details['adapted_transformer_layer']=sorted(layerids);details['counts']=dict(checks=len(checks));save_json(ROOT/'results/finetuning_verification.json',dict(checks=checks,all_passed=all(checks.values()),details=details));print(checks,flush=True);assert all(checks.values())
if __name__=='__main__':main()
