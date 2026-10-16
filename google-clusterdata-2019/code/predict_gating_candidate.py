"""Reproduce the rejected round8c gate for research, preserving best deployment."""
import hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from finetune_best_gating import ROOT, B, R, load, predict, save_json

def main():
    s = json.loads((R/'results/selection.json').read_text())
    assert hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest() == (R/'results/selection.sha256').read_text()
    with (R/'models'/f'{s["config"]}_final.pkl').open('rb') as f:
        model = pickle.load(f)
    df = pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    shape = (57,1,96)
    prior = df.predicted_util.to_numpy().reshape(shape)
    old = df.previous_selected_util.to_numpy().reshape(shape)
    candidate = df.finetuned_candidate_util.to_numpy().reshape(shape)
    t = np.load(B/'round2/results/tirex_legacy_31_1day.npz')['tirex_uni']
    legacy = pd.read_csv(B/'round2/results/legacy_deployment_origin31_1day.csv').predicted_util.to_numpy().reshape(shape)
    c = 2*legacy-t
    fs = np.load(B/'round4/results/deployment_origin31_1day.npz')['tf3_cellprod']
    a0 = json.loads((B/'round8b/results/selection.json').read_text())['chosen']['alpha']
    effective_tf = fs+2*a0*(candidate-old)
    experts = np.stack([c,t,effective_tf],-1)
    d = load()
    pure = predict(model,d,[31],prior,experts)
    a = s['chosen']['alpha']
    selected = np.clip((1-a)*prior+a*pure,0,1)
    df['previous_best_util'] = prior.ravel()
    df['pure_gate_candidate_util'] = pure.ravel()
    df['predicted_util'] = selected.ravel()
    width = .5*(df.upper80_empirical.to_numpy()-df.lower80_empirical.to_numpy())
    # This candidate interval reuses prior development width, not new calibration.
    df['lower80_empirical'] = np.maximum(selected.ravel()-width,0)
    df['upper80_empirical'] = np.minimum(selected.ravel()+width,1)
    df['recommended_for_best_deployment'] = False
    out = R/'results/gating_candidate_deployment_origin31_1day.csv'
    df.to_csv(out,index=False)
    checks = dict(rows_5472=len(df)==5472, finite=bool(np.isfinite(selected).all()),
                  power_bounds=bool((selected>=0).all() and (selected<=1).all()),
                  training_cutoff22=model['cutoff']==22, frozen_selection_hash=True,
                  interval_ordered=bool(((df.lower80_empirical<=df.predicted_util)&(df.predicted_util<=df.upper80_empirical)).all()),
                  independent_confirmation=False, recommended_for_best_deployment=False,
                  input_end='2019-05-31',forecast_start='2019-06-01',not_live_forecast=True,
                  interval='Prior empirical width reused; candidate not separately calibrated')
    assert all(checks[k] for k in ['rows_5472','finite','power_bounds','training_cutoff22','frozen_selection_hash','interval_ordered'])
    save_json(R/'results/deployment_verification.json',checks)
    print(out,checks,flush=True)

if __name__ == '__main__':
    main()
