"""Interpreter-only routing; the frozen scientific scripts remain unchanged."""
from pathlib import Path
import subprocess,sys
from core import read,save,sha,utc
import run_pipeline as pipeline
root=Path(sys.argv[1]);folder=root/'experiment/generalization_validation_20261012';Popen=subprocess.Popen
assert not list((folder/'cache').glob('test_*.npz')) or (folder/'results/nlr_selection.sha256').exists()
def routed(argv,**kwargs):
    args=list(argv)
    if len(args)>4 and str(args[2]).endswith('foundation_stage.py'):
        family=args[4]
        runtime=root/'experiment/round44'/('.venv' if family=='tirex2' else'gpu-venv')/'Scripts/python.exe'
        assert runtime.is_file();args[0]=str(runtime)
        with (folder/'results/runtime_routing.log').open('a',encoding='utf-8') as f:f.write(utc()+' '+str(args)+'\n')
    return Popen(args,**kwargs)
subprocess.Popen=routed
save(folder/'docs/runtime_routing.json',dict(utc=utc(),wrapper_sha256=sha(__file__),
    reason='Primaryenvironment TiRex/xLSTM eagerlyrequests absentCUDAcompiler; reuse prior validated CPU/GPU HDSF environments',
    frozen_scientific_scripts_unchanged=True,model_weights_and_inference_flags_unchanged=True,
    fallback='TiRex CPUtorch2.8; allotherneuralfamilies CUDA torch2.11+cu128',
    first_recovery_before_any_test=True))
try:pipeline.main(root)
except BaseException as ex:
    import traceback
    save(folder/'results/pipeline_status.json',dict(state='failed',utc=utc(),error=repr(ex)))
    save(folder/'results/pipeline_recovery_failure.json',dict(utc=utc(),error=repr(ex),traceback=traceback.format_exc()));raise
