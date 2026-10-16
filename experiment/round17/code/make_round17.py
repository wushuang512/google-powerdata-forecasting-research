from pathlib import Path
import sys
T=Path(sys.argv[1]);src=Path(__file__).parent;R=T/'experiment/round17';(R/'code').mkdir(parents=True,exist_ok=True)
if (R/'docs/protocol.json').exists() and (R/'code/multihorizon_residual_experiment.py').read_bytes()!=(src/'multihorizon_residual_experiment.py').read_bytes():
 assert not (R/'results/dev_summary.json').exists() and not (R/'results/selection.json').exists(), 'Never amend after scoring'
 import shutil,json
 a=R/'results/preflight_failure1';a.mkdir(exist_ok=False)
 for p in [R/'docs/protocol.json',R/'docs/protocol.sha256',R/'code/multihorizon_residual_experiment.py',R/'results/last_run.json',R/'results/pipeline.log']:shutil.copyfile(p,a/p.name)
 (a/'reason.json').write_text(json.dumps(dict(reason='Original train cache contains cellprod only. Change canary to identical existing DEV origin14 uni21 control; no DEV scores or selection existed; candidate configs and selection unchanged.'),indent=2),encoding='utf-8')
 (R/'docs/protocol.json').unlink();(R/'docs/protocol.sha256').unlink()
(R/'code/multihorizon_residual_experiment.py').write_bytes((src/'multihorizon_residual_experiment.py').read_bytes())
guard=(src/'run_joint_local_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round16'","R=T/'experiment/round17'").replace("['develop','test']","['forecast_train7','develop','test']").replace('code/joint_local_experiment.py','code/multihorizon_residual_experiment.py')
(R/'code/run_multihorizon_residual_guarded.py').write_text(guard,encoding='utf-8')
print('Round17 code saved; no old rounds modified')
