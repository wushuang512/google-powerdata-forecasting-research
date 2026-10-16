"""One serial, resumable pipeline; every task selected before any test inference."""
import argparse,ctypes,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_user_protocol_20261013';status=F/'results/pipeline_status.json';pidfile=F/'results/pipeline_pid.json';ledger=F/'results/pipeline_jobs.json'
if pidfile.exists():
    prev=read(pidfile)['pid'];h=ctypes.windll.kernel32.OpenProcess(0x1000,False,prev)
    if h and prev!=os.getpid():
        code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
        if code.value==259:raise RuntimeError('Existing research parent still alive; refuse duplicate')
save(pidfile,dict(pid=os.getpid(),utc=utc(),command=sys.argv));done=read(ledger) if ledger.exists() else{}
cpu=R/'experiment/.venv/Scripts/python.exe';gpu=R/'experiment/round44/gpu-venv/Scripts/python.exe';jobs=[(cpu,'evaluate.py',['audit']),(cpu,'statistical.py',['validation'])]
jobs.extend((gpu,'foundation.py',[f,'validation']) for f in ['timesfm3','toto2','moirai2']);jobs.append((cpu,'evaluate.py',['freeze']))
jobs.append((cpu,'statistical.py',['final']));jobs.extend((gpu,'foundation.py',[f,'test']) for f in ['timesfm3','toto2','moirai2']);jobs.extend([(cpu,'evaluate.py',['score']),(cpu,'evaluate.py',['audit'])])
for i,(runtime,script,extra) in enumerate(jobs,1):
    argv=[str(runtime),str(F/'code'/script),str(R)]+extra;identity=hashlib.sha256((json.dumps(argv)+sha(F/'docs/protocol.json')).encode()).hexdigest()
    if identity in done and done[identity]['exit_code']==0:continue
    log=F/'results'/f'job_{i:02}_{script[:-3]}.log';env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    info=dict(state='running',job=i,total_jobs=len(jobs),pid=os.getpid(),command=argv,log=str(log),utc=utc());save(status,info);print('JOB',i,script,extra,flush=True)
    with log.open('a',encoding='utf-8') as f:
        p=subprocess.Popen(argv,cwd=str(F/'code'),stdout=f,stderr=subprocess.STDOUT,env=env);info['child_pid']=p.pid;save(status,info)
        while p.poll() is None:time.sleep(5)
    result=dict(command=argv,exit_code=p.returncode,log=str(log),completed_utc=utc());done[identity]=result;save(ledger,done)
    if p.returncode:save(status,dict(state='failed',job=i,**result));raise SystemExit(p.returncode)
save(status,dict(state='complete',utc=utc(),all_experiments_complete=True,report_not_yet_created=True));save(F/'results/pipeline_complete.json',dict(all_passed=True,utc=utc(),actual_model_scoring_complete=True));print('Actual new-protocol evaluation and audit complete',flush=True)
