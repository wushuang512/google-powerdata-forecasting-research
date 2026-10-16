"""Finish raw development, extra historical modes, freeze, then evaluate."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
gpu=ROOT/'experiment/round44/gpu-venv/Scripts/python.exe';cpu=ROOT/'experiment/round44/.venv/Scripts/python.exe';stats=ROOT/'experiment/.venv/Scripts/python.exe'
status=F/'results/extended_pipeline_status.json';ledger=F/'results/extended_jobs.json';done=read(ledger) if ledger.exists() else{}
save(F/'results/extended_pipeline_pid.json',dict(pid=os.getpid(),started_utc=utc(),command=sys.argv))
while not (F/'results/development_raw_bank_complete.json').exists() or not (F/'docs/implementation_ready.json').exists():
    save(status,dict(state='waiting_for_development_and_implementation',pid=os.getpid(),utc=utc()));time.sleep(15)
proto=read(F/'docs/protocol.json')
def call(runtime,script,args):
    argv=[str(runtime),str(F/'code'/script),str(ROOT)]+list(map(str,args));identity=hashlib.sha256(json.dumps(argv,ensure_ascii=False).encode()).hexdigest()
    if identity in done and done[identity]['exit_code']==0:return
    log=F/'results'/f'extended_{identity[:14]}.log';env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'experiment/round29/code'),env.get('PYTHONPATH','')]).rstrip(os.pathsep)
    info=dict(state='running',pid=os.getpid(),command=argv,log=str(log),started_utc=utc(),completed_jobs=len(done));save(status,info);print('EXTENDED',' '.join(argv[3:]),flush=True)
    with log.open('a',encoding='utf-8') as out:
        p=subprocess.Popen(argv,cwd=str(F),stdout=out,stderr=subprocess.STDOUT,env=env);info['child_pid']=p.pid;save(status,info)
        while p.poll() is None:time.sleep(5)
    entry=dict(command=argv,exit_code=p.returncode,log=str(log),completed_utc=utc());done[identity]=entry;save(ledger,done)
    if p.returncode:save(status,dict(state='failed',**entry));raise SystemExit(p.returncode)

# Every extra mode is trained/evaluated on DEV before any TEST job.
for case in ['NLR_2024_2025','NLR_2021','Google2019']:
    stages=[(proto['cases'][case]['cutoff'],'dev')]
    if case=='Google2019':stages.insert(0,(7,'oof'))
    for cut,phase in stages:
        for rate in [4,12]:
            call(gpu,'moirai_modes.py',[case,'--cut',cut,'--rate',rate,'--stage','train'])
            for label in ['s0','s16','s48']:call(gpu,'moirai_modes.py',[case,'--cut',cut,'--rate',rate,'--stage','infer','--label',label,'--phase',phase])
for case in ['NLR_2024_2025','NLR_2021']:
    for family in ['timesfm3','toto2','patchtst','local','assemble']:call(stats if family in ['local','assemble'] else gpu,'source_control.py',[case,family,'--phase','dev'])
save(F/'results/additional_development_complete.json',dict(all_passed=True,utc=utc(),test_scoring_started=False))
for case in proto['cases']:call(stats,'portfolio_stage.py',[case,'--stage','freeze'])
call(stats,'prepare_reference_controls.py',[])
call(stats,'export_cluster_base.py',['--phase','dev'])
call(stats,'cluster_stage.py',['--stage','freeze'])
save(F/'results/before_all_evaluation.json',dict(utc=utc(),test_scoring_started=False,
    selection_sha256={case:hashlib.sha256((F/'results'/f'{case}_selection.json').read_bytes()).hexdigest() for case in proto['cases']},
    cluster_selection_sha256=hashlib.sha256((F/'results/Cluster_selection.json').read_bytes()).hexdigest()))
for case,cfg in proto['cases'].items():
    s=read(F/'results'/f'{case}_selection.json');assert hashlib.sha256((F/'results'/f'{case}_selection.json').read_bytes()).hexdigest()==(F/'results'/f'{case}_selection.sha256').read_text().strip()
    phases=['test','confirm'] if case=='NLR_2024_2025' else['test'];cut=cfg['cutoff'] if case.startswith('NLR') else cfg['final_cutoff']
    families=proto['neural_families']+['ttm1024']
    for family in families:
        runtime=cpu if family=='tirex2' else gpu
        if case not in ['NLR_2024_2025','NLR_2021','Sea2022']:call(runtime,'neural_stage.py',[case,family,'--cut',cut,'--stage','train'])
        labels=sorted(set(['s0']+[s['tasks'][str(H)]['checkpoints'][family] for H in [1,7]]))
        for phase in phases:
            for label in labels:call(runtime,'neural_stage.py',[case,family,'--cut',cut,'--stage','infer','--label',label,'--phase',phase])
    if case!='Sea2022':
        for rate in [4,12]:
            family='moirai_rate'+str(rate)
            if case=='Google2019':call(gpu,'moirai_modes.py',[case,'--cut',cut,'--rate',rate,'--stage','train'])
            labels=sorted(set(['s0']+[s['tasks'][str(H)]['checkpoints'][family] for H in [1,7]]))
            for phase in phases:
                for label in labels:call(gpu,'moirai_modes.py',[case,'--cut',cut,'--rate',rate,'--stage','infer','--label',label,'--phase',phase])
    for phase in phases:
        call(stats,'statistical_stage.py',[case,'--cut',cut,'--phase',phase])
        if case.startswith('NLR'):
            for family in ['timesfm3','toto2','patchtst','local','assemble']:call(stats if family in ['local','assemble'] else gpu,'source_control.py',[case,family,'--phase',phase])
        call(stats,'portfolio_stage.py',[case,'--stage','score','--phase',phase])
call(stats,'export_cluster_base.py',['--phase','test'])
call(stats,'cluster_stage.py',['--stage','score'])
call(stats,'audit_and_summarize.py',[])
save(status,dict(state='complete',utc=utc(),completed_jobs=len(done),actual_scores_complete=True))
save(F/'results/extended_experiment_complete.json',dict(all_passed=True,utc=utc(),actual_scores_complete=True))
