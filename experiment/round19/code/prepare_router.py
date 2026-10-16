from pathlib import Path
import sys,json,shutil,hashlib
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round19';Q=R/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((Q/'independent_verification.json').read_text())['all_passed']
shutil.copyfile(Path(__file__).with_name('predict_horizon_router.py'),R/'code/predict_horizon_router.py')
files=['experiment/round19/results/deployment7_origin31.npz','experiment/round19/results/selection.json','experiment/round19/results/dev_predictions.npz','experiment/round18/results/dev_predictions.npz','experiment/round8b/results/selected_deployment_origin31_1day.csv','experiment/results/data.npz']
(Q/'deployment_bundle.json').write_text(json.dumps(dict(source_hashes={p:sha(T/p) for p in files},known_horizon_routing=True,days1='original round8b unchanged',days7='DEV frozen round19 global-total projection alpha.25',not_one_identical_network=True,cached_verified_May31_inference=True),indent=2),encoding='utf-8')
p=B/'round18/results/deployment_time_erratum.json';p.write_text(json.dumps(dict(original_csv='selected_deployment_origin31_7day.csv',error='Original export mislabeled trace-local PT samples as UTC midnight. Prediction values and scoring arrays unchanged. Original CSV retained as internal evidence and superseded for use.',correct_first_PT='2019-06-01T00:00:00-07:00',correct_first_UTC='2019-06-01T07:00:00+00:00',canonical_export='../round19/results/exports/selected_7day_curve.csv',reexport_changes_model=False),ensure_ascii=False,indent=2),encoding='utf-8')
cmd='@echo off\r\nset "RESEARCH_ROOT=%~dp0"\r\n"%RESEARCH_ROOT%experiment\\.venv\\Scripts\\python.exe" -X utf8 "%RESEARCH_ROOT%experiment\\round19\\code\\predict_horizon_router.py" "%RESEARCH_ROOT%." --days '+ '{days}' +'\r\npause\r\n'
for days in [1,7]:(T/f'17_{days}天按跨度选择模型历史复现.cmd').write_bytes(cmd.format(days=days).encode('utf-8'))
print('Verified historical router and correct trace-time export prepared')
