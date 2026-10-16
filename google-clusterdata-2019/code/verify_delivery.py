import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent;B=TOP/'experiment'
def main():
    checks={};manifest=json.loads((ROOT/'docs/download_manifest.json').read_text());sample=json.loads((ROOT/'docs/usage_sample_manifest.json').read_text());checks['complete_32_machine_files']=len(manifest)==32;checks['machine_file_sizes']=all((ROOT/'raw'/x['cell']/x['file']).stat().st_size==x['bytes'] for x in manifest);checks['usage_sample_size']=(ROOT/'raw/a'/sample['file']).stat().st_size==sample['bytes']
    for rr in ['round5','round6','round7']:
        s=B/rr/'results/selection.json';h=B/rr/'results/selection.sha256';checks[rr+'_frozen_hash']=hashlib.sha256(s.read_bytes()).hexdigest()==h.read_text()
        for name in (['selected','fusion'] if rr=='round5' else ['candidate']):
            df=pd.read_csv(B/rr/'results'/f'{name}_deployment_origin31_1day.csv');checks[rr+'_'+name+'_deployment_rows']=len(df)==57*96;checks[rr+'_'+name+'_deployment_bounds']=bool(df.predicted_util.between(0,1).all());checks[rr+'_'+name+'_deployment_finite']=bool(np.isfinite(df.predicted_util).all())
    d=np.load(B/'results/data.npz');r=np.load(ROOT/'aggregated/resources_15min.npz');u=np.load(ROOT/'aggregated/usage_sample_15min.npz');checks['resource_names_aligned']=np.array_equal(r['names'],d['names']);checks['usage_names_aligned']=np.array_equal(u['names'],d['names']);checks['resource_shapes']=r['resource'].shape==(57,31,96,8);checks['usage_shapes']=u['features'].shape==(57,31,96,6);checks['resource_checks']=json.loads((ROOT/'results/mapping_audit.json').read_text())['all_passed']
    for rr in ['round6','round7']:checks[rr+'_causal_checks']=json.loads((B/rr/'results/summary.json').read_text())['all_passed']
    controls=np.load(B/'round6/results/test_predictions.npz');samplepred=np.load(B/'round7/results/test_predictions.npz');checks['resource_to_sample_control_exact']=all(np.array_equal(controls[f'ridge_resource_{a}'],samplepred[f'ridge_control_{a}']) for a in [10000,100000,1000000]);checks['resource_primary_retained']=np.array_equal(controls['prior'],controls['selected']);checks['sample_primary_retained']=np.array_equal(samplepred['prior'],samplepred['selected'])
    out=dict(checks={k:bool(v) for k,v in checks.items()},all_passed=all(checks.values()),full_workload_acquired=False,scope='File integrity, frozen decisions and deployment checks. Not proof of generalization or no overfitting.')
    (ROOT/'results/delivery_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert out['all_passed']
if __name__=='__main__':main()
