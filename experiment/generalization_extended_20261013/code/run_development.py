"""Sequential resumable development jobs; never score a test before freezing."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time
import ctypes
from ctypes import wintypes
class MemoryStatus(ctypes.Structure):
    _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in ['total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended']]
def available_memory():
    value=MemoryStatus();value.length=ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value.available_physical
def pid_exists(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=wintypes.HANDLE
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=wintypes.DWORD();kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
    return code.value==259
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
proto=read(F/'docs/protocol.json');lock=F/'results/development_pipeline_pid.json'
if lock.exists():
    previous=read(lock)
    if previous['pid']!=os.getpid() and pid_exists(previous['pid']):
        raise RuntimeError('Previous pipeline PID is still active; inspect it before starting another')
save(lock,dict(pid=os.getpid(),created_utc=utc(),command=sys.argv))
gpu=ROOT/'experiment/round44/gpu-venv/Scripts/python.exe';cpu=ROOT/'experiment/round44/.venv/Scripts/python.exe'
stats_python=ROOT/'experiment/.venv/Scripts/python.exe'
jobs=[]
families=proto['neural_families']+['ttm1024']
for case in ['NLR_2024_2025','NLR_2021','Google2019','Sea2022']:
    cuts=[(proto['cases'][case]['cutoff'],'dev')]
    if case in ['Google2019','Sea2022']:cuts.insert(0,(7,'oof'))
    for cut,phase in cuts:
        for family in families:
            runtime=cpu if family=='tirex2' else gpu
            script=F/'code/neural_stage.py'
            if case!='Sea2022':jobs.append([str(runtime),str(script),str(ROOT),case,family,'--cut',str(cut),'--stage','train'])
            for label in ['s0'] if case=='Sea2022' else['s0','s16','s48']:
                jobs.append([str(runtime),str(script),str(ROOT),case,family,'--cut',str(cut),'--stage','infer','--label',label,'--phase',phase])
        jobs.append([str(stats_python),str(F/'code/statistical_stage.py'),str(ROOT),case,'--cut',str(cut),'--phase',phase])
save(F/'docs/development_job_plan.json',dict(created_utc=utc(),jobs=jobs,total=len(jobs),scientific_protocol_sha256=hashlib.sha256((F/'docs/protocol.json').read_bytes()).hexdigest()))
statepath=F/'results/development_pipeline_status.json';ledger=F/'results/development_jobs.json'
done=read(ledger) if ledger.exists() else{}
for number,argv in enumerate(jobs,1):
    identity=hashlib.sha256(json.dumps(argv,ensure_ascii=False).encode()).hexdigest()
    if identity in done and done[identity]['exit_code']==0:continue
    info=dict(state='running',pid=os.getpid(),job_number=number,total_jobs=len(jobs),command=argv,started_utc=utc(),completed_jobs=len(done))
    save(statepath,info);log=F/'results'/f'job_{number:03}_{argv[3]}_{argv[4] if len(argv)>4 else "stat"}.log'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'experiment/round29/code'),env.get('PYTHONPATH','')]).rstrip(os.pathsep)
    print('JOB',number,len(jobs),' '.join(argv[3:]),flush=True)
    with log.open('a',encoding='utf-8') as output:
        child=subprocess.Popen(argv,stdout=output,stderr=subprocess.STDOUT,env=env,cwd=str(F))
        info['child_pid']=child.pid;save(statepath,info)
        while child.poll() is None:
            # This is monitoring only, not a fixed startup memory gate.
            info.update(last_checked_utc=utc(),available_ram_bytes=available_memory())
            save(statepath,info);time.sleep(5)
        exit_code=child.returncode
    entry=dict(command=argv,exit_code=exit_code,log=str(log),completed_utc=utc());done[identity]=entry;save(ledger,done)
    if exit_code:
        save(statepath,dict(state='failed',job_number=number,**entry));raise SystemExit(exit_code)
save(statepath,dict(state='complete',completed_utc=utc(),completed_jobs=len(done),test_scoring_started=False))
save(F/'results/development_raw_bank_complete.json',dict(all_passed=True,completed_utc=utc(),jobs=len(done),test_scoring_started=False))
print('All actual target fits and development prediction banks complete; test scoring remains sealed',flush=True)
