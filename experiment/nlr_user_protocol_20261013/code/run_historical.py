import argparse,ctypes,os,subprocess,sys,time,hashlib
from pathlib import Path
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_user_protocol_20261013';assert read(F/'results/pipeline_complete.json')['all_passed']
lock=F/'results/historical_pipeline_pid.json'
if lock.exists():
    previous=read(lock)['pid'];h=ctypes.windll.kernel32.OpenProcess(0x1000,False,previous)
    if h and previous!=os.getpid():
        code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
        if code.value==259:raise RuntimeError('Existing historical pipeline is active; refuse duplicate')
save(lock,dict(pid=os.getpid(),utc=utc(),command=sys.argv));gpu=R/'experiment/round44/gpu-venv/Scripts/python.exe';cpu=R/'experiment/.venv/Scripts/python.exe';jobs=[]
for phase in ['validation','test']:
    for f in ['chronos2','tirex2','timesfm25','patchtst','timesfm3','toto2']:jobs.append((gpu,'historical_foundation.py',[f,phase]))
    jobs.append((cpu,'historical_assemble.py',[phase]))
jobs.extend([(cpu,'score_historical.py',[]),(cpu,'evaluate.py',['audit'])]);ledger=F/'results/historical_pipeline_jobs.json';done=read(ledger) if ledger.exists() else{}
for i,(runtime,script,extra) in enumerate(jobs,1):
    argv=[str(runtime),str(F/'code'/script),str(R)]+extra;identity=hashlib.sha256((str(argv)+sha(F/'docs/historical_controls.json')).encode()).hexdigest()
    if identity in done and done[identity]['exit_code']==0:continue
    log=F/'results'/f'historical_job_{i:02}.log';env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    info=dict(state='running',pid=os.getpid(),job=i,total_jobs=len(jobs),command=argv,log=str(log),utc=utc());save(F/'results/historical_pipeline_status.json',info);print('HISTORICAL JOB',i,script,extra,flush=True)
    with log.open('a',encoding='utf-8') as output:
        p=subprocess.Popen(argv,cwd=str(F/'code'),stdout=output,stderr=subprocess.STDOUT,env=env);info['child_pid']=p.pid;save(F/'results/historical_pipeline_status.json',info)
        while p.poll() is None:time.sleep(5)
    record=dict(exit_code=p.returncode,command=argv,log=str(log),completed_utc=utc());done[identity]=record;save(ledger,done)
    if p.returncode:save(F/'results/historical_pipeline_status.json',dict(state='failed',job=i,**record));raise SystemExit(p.returncode)
save(F/'results/historical_pipeline_status.json',dict(state='complete',utc=utc(),actual_controls_complete=True,report_not_yet_created=True));print('Historical source comparison complete',flush=True)
