"""Sequential recoverable actual experiment; no duplicate live invocation."""
from pathlib import Path
import argparse,ctypes,json,os,subprocess,sys,time,traceback
sys.dont_write_bytecode=True
from core import *
def main(root):
    folder=root/'experiment/generalization_validation_20261012';q=folder/'results';q.mkdir(parents=True,exist_ok=True)
    pidfile=q/'pipeline_pid.json'
    if pidfile.exists():
        old=read(pidfile);h=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(old['pid']))
        if h:
            ctypes.windll.kernel32.CloseHandle(h);raise RuntimeError('An existing pipeline process is live; do not duplicate')
    save(pidfile,dict(pid=os.getpid(),started_utc=utc(),root=str(root)))
    py=root/'experiment/.venv/Scripts/python.exe';code=folder/'code';supp=str(root/'experiment/round44/.venv/Lib/site-packages')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTHONUTF8='1')
    def run(name,args,done=None):
        if done and (q/done).exists():
            state=read(q/done);assert state.get('all_passed',True);print('Already completed, preserve',name,flush=True);return
        save(q/'pipeline_status.json',dict(state='running',step=name,pid=os.getpid(),utc=utc()))
        print('START',name,utc(),flush=True)
        with (q/'pipeline.log').open('a',encoding='utf-8') as log:
            log.write('\nSTART '+name+' '+utc()+'\n');log.flush()
            child=subprocess.Popen([str(py),'-u',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
            save(q/'pipeline_child.json',dict(pid=child.pid,parent_pid=os.getpid(),step=name,started_utc=utc(),command=args))
            rc=child.wait()
        if rc:raise RuntimeError(f'{name} failed ({rc}); see pipeline.log; caches retained')
        print('FINISHED',name,utc(),flush=True)
    # These phases are sequential and recover existing compatible forecast files.
    for f in ['chronos2','tirex2','timesfm25','timesfm3','toto2','patchtst']:
        run(f+' DEV',[str(code/'foundation_stage.py'),str(root),f,'dev'],f+'_dev_complete.json')
    run('NLR freeze and final localfit',[str(code/'evaluate.py'),str(root),'--stage','freeze'])
    run('SeaGuang daily validation',[str(code/'supplementary.py'),str(root),'sea'],'sea_summary.json')
    run('Cluster same-source ablation',[str(code/'supplementary.py'),str(root),'cluster'],'cluster_summary.json')
    for phase in ['test','confirm']:
        for f in ['chronos2','tirex2','timesfm25','timesfm3','toto2','patchtst']:
            run(f+' '+phase,[str(code/'foundation_stage.py'),str(root),f,phase],f+'_'+phase+'_complete.json')
        run('NLR actual scoring '+phase,[str(code/'evaluate.py'),str(root),'--stage','score','--phase',phase],'nlr_'+phase+'_summary.json')
    run('Independent verification and direct report',[str(code/'verify_and_report.py'),str(root)],'final_verification.json')
    save(q/'pipeline_complete.json',dict(all_passed=True,utc=utc(),actual_external_model_scoring=True,final_verification_sha256=sha(q/'final_verification.json')))
    save(q/'pipeline_status.json',dict(state='complete',pid=os.getpid(),utc=utc()))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();root=Path(a.root)
    try:main(root)
    except BaseException as ex:
        folder=root/'experiment/generalization_validation_20261012/results'
        save(folder/'pipeline_failure.json',dict(utc=utc(),error=repr(ex),traceback=traceback.format_exc()))
        save(folder/'pipeline_status.json',dict(state='failed',pid=os.getpid(),utc=utc(),error=repr(ex)));raise
