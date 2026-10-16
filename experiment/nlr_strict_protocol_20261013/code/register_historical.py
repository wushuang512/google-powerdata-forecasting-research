from pathlib import Path
import argparse
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_strict_protocol_20261013'
p=F/'docs/historical_controls.json'
if p.exists():assert sha(p)==(p.with_suffix('.sha256')).read_text().strip();print('Historical controls original registration retained');raise SystemExit()
r5=read(R/'experiment/round5/results/selection.json');r8=read(R/'experiment/round8b/results/selection.json');r46=read(R/'experiment/round46/results/selection.json');r44=read(R/'experiment/round44/results/selection.json')
cal=R/'experiment/round5/models'/f'{r5["chosen"]["calibration"]}_final.pkl';ft8=R/'experiment/round8b/models'/f'{r8["config"]}_final.pt'
bindings={str(p.relative_to(R)):sha(p) for p in [R/'experiment/round44/results/selection.json',R/'experiment/round44/results/local_cut22.joblib',R/'experiment/round48/deployment/selected_configuration.json',R/'experiment/round5/results/selection.json',R/'experiment/round5/docs/calibration_features.json',R/'experiment/round8b/results/selection.json',cal,ft8,R/'experiment/round47/models/patchtst_curve_cut22.pt']}
for family in ['timesfm3','toto2','patchtst']:
    path=R/'experiment/round46/models'/f'{family}_{r46["families"][family]["config"]}_cut22.pt';bindings[str(path.relative_to(R))]=sha(path)
config=dict(registered_utc=utc(),trigger='User explicitly requests full separate rerun; all fourteen registered before validation and test',base_protocol_sha256=sha(F/'docs/protocol.json'),candidate_set_registered_before_test=True,all_schemes_enter_common_validation_selection=True,
 main_test_pipeline_already_completed=False,additional_controls_not_used_to_change_previous_experiment=True,no_new_control_scores_used_to_select_source_configuration=True,
 history_days=14,input_adaptation=True,original_Google_long_history_and_multivariate_modes_not_claimed_identical=True,
 source_weights_frozen=True,target_gradient_updates=False,normalization='training/known train+validation95percentile/0.8; not facility capacity',
 missing_Google_features='production proxy unavailable=>availability0 and absent proxy0 as source missing branch; unknown NLR cell => all source onehot0; local tree missing-feature path; no invented CPU channel',
 physical_output='nonnegative kW; do not impose original Google normalized upper bound1',
 seven_day_calibrator='pointwise extension of source1day Ridge on each predicted day, current known history remains fixed; clearly labelled7day extension, not original seven-day recommendation',
 schemes=['Google_Chronos_TiRex_equal','Google_three_fixed','Google_R5_Ridge','Google_R8b_finetuned','Google_HDSF_original','Google_R48_robust'],
 source_r5_residual_alpha=r5['selected']['alpha'],source_r8_component_alpha=r8['chosen']['alpha'],
 ridge_file=str(cal),r8_weight_file=str(ft8),r46_states={f:str(R/'experiment/round46/models'/f'{f}_{r46["families"][f]["config"]}_cut22.pt') for f in ['timesfm3','toto2','patchtst']},
 original_hdsf_methods={h:c['methods'] for h,c in r44['configs'].items()},bindings=bindings)
save(p,config);p.with_suffix('.sha256').write_text(sha(p));print('All six source configurations registered before any new validation or TEST',sha(p))
