"""One fresh serial GPU pipeline, all candidates registered and frozen before TEST."""
import argparse,ctypes,os,subprocess,sys,time,hashlib
from pathlib import Path
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_strict_protocol_20261013'
for name in ['results','docs','models','data','cache']:(F/name).mkdir(parents=True,exist_ok=True)
pidfile=F/'results/pipeline_pid.json';status=F/'results/pipeline_status.json';ledger=F/'results/pipeline_jobs.json'
if pidfile.exists():
    previous=read(pidfile)['pid'];h=ctypes.windll.kernel32.OpenProcess(0x1000,False,previous)
    if h and previous!=os.getpid():
        code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
        if code.value==259:raise RuntimeError('This strict rerun already has an active parent; refuse duplicate')
save(pidfile,dict(pid=os.getpid(),utc=utc(),command=sys.argv));done=read(ledger) if ledger.exists() else{}
cpu=R/'experiment/.venv/Scripts/python.exe';gpu=R/'experiment/round44/gpu-venv/Scripts/python.exe'
jobs=[(cpu,'prepare.py',[]),(cpu,'register_historical.py',[]),(cpu,'evaluate.py',['audit']),(cpu,'verify_raw_stream.py',[]),(cpu,'statistical.py',['validation'])]
jobs.extend((gpu,'foundation.py',[family,'validation']) for family in ['timesfm3','toto2','moirai2'])
jobs.extend((gpu,'historical_foundation.py',[family,'validation']) for family in ['chronos2','tirex2','timesfm25','patchtst','timesfm3','toto2'])
jobs.extend([(cpu,'historical_assemble.py',['validation']),(cpu,'evaluate.py',['freeze']),(cpu,'statistical.py',['final'])])
jobs.extend((gpu,'foundation.py',[family,'test']) for family in ['timesfm3','toto2','moirai2'])
jobs.extend((gpu,'historical_foundation.py',[family,'test']) for family in ['chronos2','tirex2','timesfm25','patchtst','timesfm3','toto2'])
jobs.extend([(cpu,'historical_assemble.py',['test']),(cpu,'evaluate.py',['score']),(cpu,'evaluate.py',['audit']),(cpu,'complete_tables.py',[])])
for i,(runtime,script,extra) in enumerate(jobs,1):
    argv=[str(runtime),str(F/'code'/script),str(R)]+extra
    digest=sha(F/'docs/protocol.json') if (F/'docs/protocol.json').exists() else'preprocessing_before_validation'
    identity=hashlib.sha256((str(argv)+digest+sha(F/'code'/script)).encode()).hexdigest()
    if identity in done and done[identity]['exit_code']==0:continue
    is_test=(script=='statistical.py' and extra==['final']) or extra[-1:] == ['test'] or (script=='evaluate.py' and extra==['score'])
    if is_test:
        selected=read(F/'results/selection.json');assert selected['all_fourteen_candidates_pre_registered'] and selected['all_tasks_frozen_before_test']
        assert sha(F/'results/selection.json')==(F/'results/selection.sha256').read_text().strip()
        assert selected['historical_registration_sha256']==sha(F/'docs/historical_controls.json')
        marker=F/'results/test_inference_started.json'
        if not marker.exists():save(marker,dict(utc=utc(),first_job=i,selection_sha256=sha(F/'results/selection.json'),all_candidates_registered_before_this_first_test=True,previous_test_exposure=True))
    log=F/'results'/f'job_{i:02}_{script[:-3]}.log';env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
    info=dict(state='running',pid=os.getpid(),job=i,total_jobs=len(jobs),command=argv,log=str(log),utc=utc());save(status,info);print('STRICT JOB',i,script,extra,flush=True)
    with log.open('a',encoding='utf-8') as f:
        p=subprocess.Popen(argv,cwd=str(F/'code'),stdout=f,stderr=subprocess.STDOUT,env=env);info['child_pid']=p.pid;save(status,info)
        while p.poll() is None:time.sleep(3)
    item=dict(command=argv,exit_code=p.returncode,log=str(log),completed_utc=utc(),script_sha256=sha(F/'code'/script));done[identity]=item;save(ledger,done)
    if p.returncode:save(status,dict(state='failed',job=i,**item));raise SystemExit(p.returncode)
save(status,dict(state='complete',utc=utc(),fresh_preprocessing_training_and_inference_complete=True,report_not_yet_created=True));save(F/'results/pipeline_complete.json',dict(all_passed=True,utc=utc(),all_fourteen_registered_before_test=True,test_scoring_complete=True));print('Strict full rerun complete',flush=True)
