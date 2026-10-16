"""Reproduce the frozen round8b model on the May31 history, never live power."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round8b'
sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,save_json
from finetune_tf3 import load_delta,forecast,BASECFG

def main():
    s=json.loads((R/'results/selection.json').read_text());assert hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest()==(R/'results/selection.sha256').read_text();d=load();old=pd.read_csv(B/'round5/results/selected_deployment_origin31_1day.csv');prior=old.predicted_util.to_numpy().reshape(57,1,96);path=R/'results/component_deployment_origin31.npz'
    if path.exists():new=np.load(path)['tf3_finetuned']
    else:
        f=load_delta(R/'models'/f'{s["config"]}_final.pt');new=np.clip(forecast(f,d,[31],BASECFG['tf3_cellprod']),0,1);np.savez_compressed(path,tf3_finetuned=new)
    original=np.load(B/'round4/results/deployment_origin31_1day.npz')['tf3_cellprod'];candidate=np.clip(prior+.5*(new-original),0,1);a=s['chosen']['alpha'];selected=np.clip((1-a)*prior+a*candidate,0,1);old['predicted_util']=selected.ravel();old['previous_selected_util']=prior.ravel();old['finetuned_candidate_util']=candidate.ravel()
    # Inherit the same development-calibrated width, recentered at the new point.
    q=json.loads((B/'round5/results/selection.json').read_text());half=np.repeat(q['residual_interval']['calibrated_halfwidth_relative_to_past_mean']*d['y'][:,24:31].mean((1,2)).clip(.05),96);old['lower80_empirical']=np.maximum(selected.ravel()-half,0);old['upper80_empirical']=np.minimum(selected.ravel()+half,1);out=R/'results/selected_deployment_origin31_1day.csv';old.to_csv(out,index=False)
    checks=dict(rows_5472=len(old)==57*96,power_bounds=bool((old.predicted_util.between(0,1)).all()),finite=bool(np.isfinite(old.select_dtypes('number')).all().all()),interval_ordered=bool(((old.lower80_empirical<=old.predicted_util)&(old.predicted_util<=old.upper80_empirical)).all()),input_end='2019-05-31',forecast_start='2019-06-01',selection_hash=True,model_training_cutoff=22,not_live_forecast=True)
    save_json(R/'results/deployment_verification.json',checks);print(out,checks,flush=True)
if __name__=='__main__':main()
