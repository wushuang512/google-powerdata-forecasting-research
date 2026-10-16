"""Bind archived adopted Google controls by prediction SHA, not test ranking."""
from pathlib import Path
import sys,csv,json,hashlib,joblib,numpy as np
sys.dont_write_bytecode=True
root=Path(sys.argv[1]);F=root/'experiment/generalization_extended_20261013';OLD=root/'experiment/generalization_validation_20261012'
sys.path[:0]=[str(OLD/'code'),str(root/'experiment/round44/reference/PowerData_Prediction')]
from core import read,sha,save,np_save
assert all((F/'results'/f'{c}_selection.sha256').exists() for c in ['NLR_2021','NLR_2024_2025','Google2019','Sea2022'])
rows=list(csv.DictReader((root/'模型各项指标完整结果.csv').open(encoding='utf-8-sig',newline='')))
selected_ids={'best','r5','r19_selected','r26_diagnostic','r32_selected_H1','r36_selected_H7','r44_hdsf_H1','r44_hdsf_H7'}
names=np.load(F/'data/Google2019.npz')['names'];source=joblib.load(root/'experiment/round44/results/source_dataset.joblib');canon=np.array([source.series_keys.index(str(n)) for n in names]);arrays={};records=[]
def digest(p):return hashlib.sha256(np.ascontiguousarray(p,dtype=np.float64).tobytes()).hexdigest()
for r in rows:
    if r['model'] not in selected_ids:continue
    ref=json.loads(r['source']);p=root/'experiment'/ref['path'];assert sha(p)==ref['sha256'];q=np.load(p)[ref['array']].astype(float)
    candidates=[q,q[canon]];matches=[v for v in candidates if digest(v)==r['prediction_digest']];assert matches,r['model'];point=matches[0];H=int(r['horizon_days']);key='archive_'+r['model']
    arrays[f'{key}_H{H}']=point;records.append(dict(model=key,model_name=r['model_name'].replace('\n',' '),horizon_days=H,source=ref,prediction_digest=digest(point),historical_deployment_or_frozen_control=True))
current=root/'experiment/round48/results/predictions.npz'
with np.load(current) as z:
    for H in [1,7]:
        key='archive_R48_current';arrays[f'{key}_H{H}']=z[f'DEV_frozen_R48_candidate_H{H}'].copy()
        arrays[f'{key}_lower80_H{H}']=z[f'DEV_frozen_R48_candidate_lower80_H{H}'].copy();arrays[f'{key}_upper80_H{H}']=z[f'DEV_frozen_R48_candidate_upper80_H{H}'].copy()
        records.append(dict(model=key,model_name=read(root/'当前推荐方案.json')['tasks'][str(H)]['model_name'],horizon_days=H,source=dict(path=str(current),sha256=sha(current)),prediction_digest=digest(arrays[f'{key}_H{H}']),previous_user_adopted=True))
np_save(F/'results/Google_archived_controls.npz',**arrays);save(F/'results/Google_archived_controls.json',dict(all_passed=True,records=records,no_test_rank_filter=True,canonical_order_bound_by_archived_prediction_SHA=True))
print('Original adopted Google controls bound',len(records),flush=True)
