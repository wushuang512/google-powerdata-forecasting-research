"""Build a narrow, explicit plan; this file does not delete anything."""
from pathlib import Path
import sys,json,hashlib,datetime,os
import numpy as np
TOP=Path(sys.argv[1]).resolve();B=TOP/'experiment';O=B/'model_comparison_20261005';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'code'));from common import load,truth,metrics
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
targets=[];d=load();y,m=truth(d,list(range(14,22)))
def add(p,reason,evidence=None):
    p=p.resolve();assert p.is_relative_to(TOP) and not p.is_symlink()
    if not p.exists():return
    fs=[p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()]
    assert all(f.resolve().is_relative_to(TOP) and not f.is_symlink() for f in fs)
    targets.append(dict(path=str(p),relative_path=str(p.relative_to(TOP)),reason=reason,evidence=evidence,
        files=[dict(path=str(f.relative_to(TOP)),bytes=f.stat().st_size,sha256=digest(f)) for f in fs],
        files_count=len(fs),bytes=sum(f.stat().st_size for f in fs)))
def dominated(z,key,other):
    a=metrics(y,z[key].astype(float),m);b=metrics(y,z[other].astype(float),m)
    names=['wape','daily_wape','peak_mape']
    assert all(a[n]>b[n] for n in names),(key,other,a,b)
    return dict(candidate=key,comparator=other,development_origins=list(range(14,22)),candidate_scores={n:a[n] for n in names},comparator_scores={n:b[n] for n in names})
# Completed, unselected full-backbone variants dominated on all three development objectives.
z=np.load(B/'round2/results/development.npz')
for key in ['full_uni_lr1e6_s100','full_uni_lr3e6_s100','full_uni_lr1e6_s300','full_cov_lr1e6_s100']:
    e=dominated(z,key,'full_cov_lr1e6_s300')
    add(B/f'round2/models/{key}_d14','未采用开发检查点且曲线 日均 峰值均劣于该轮冻结协变量候选',e)
z=np.load(B/'round4/results/development.npz')
for key in ['head_uni_s50','head_uni_s150','head_uni_s300','head_cov_s50','head_cov_s150','tail_cov_s50','tail_cov_s150','tail_cov_s300','full_cov_s150','full_cov_s300']:
    e=dominated(z,key,'head_cov_s300')
    add(B/f'round4/models/{key}_d14','未采用且三个开发指标均被保留的协变量模型支配',e)
# Every Round14 candidate is at least 20% worse on both early and late DEV curve WAPE.
s=json.loads((B/'round14/results/dev_summary.json').read_text())
assert all(r['early4']['wape']>1.2*s['prior_early4']['wape'] and r['late4']['wape']>1.2*s['prior_late4']['wape'] for r in s['candidates'])
sel=json.loads((B/'round14/results/selection.json').read_text());assert sel['chosen']['alpha']==0
add(B/'round14/models','十八个候选前后开发曲线误差均增加至少20% 正式权重0 用户要求删除太差模型',dict(alpha=0,development_candidates=18,threshold='both early4 and late4 > 120% prior WAPE'))
# Source bytecode is regenerable. Do not traverse .venv, runtime or original source byte cache.
for parent in [B/'code',TOP/'google-clusterdata-2019/code']+list(B.glob('round*/code')):
    for p in parent.rglob('__pycache__'):
        add(p,'仅自编源程序字节码 可自动重建')
# Superseded report layout renders. Final QA records and images remain.
for rd in B.glob('round*'):
    q=rd/'docs/report_qa'
    if not q.exists():continue
    v=q/'verification.json'
    if not v.exists():continue
    record=json.loads(v.read_text())
    if not record.get('all_passed'):continue
    for p in q.iterdir():
        if p.is_dir() and (p.name.startswith('layout_draft') or p.name.startswith('draft')):add(p,'排版修改前过期页面草稿 正式Word与通过核对的最终页面保留')
protected=[B/'round8b/models/best_tail_lr1e6_teacher01_s100_final.pt',B/'round8b/results/selection.json',B/'round8b/results/selection.sha256',
    B/'results/data.npz',TOP/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz',
    B/'round2/models/full_cov_lr1e6_s300_d22/finetuned-ckpt/model.safetensors',
    B/'round2/models/full_parts_lr1e6_s300_d22/finetuned-ckpt/model.safetensors']
protected += [p for p in (B/'models').rglob('*') if p.is_file()]
prot={str(p.relative_to(TOP)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in protected}
for t in targets:assert not any(Path(k).is_relative_to(Path(t['relative_path'])) for k in prot)
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(TOP),
    targets=targets,total_files=sum(t['files_count'] for t in targets),total_logical_bytes=sum(t['bytes'] for t in targets),
    protected_files=prot,policy='Delete unselected dominated dedicated weights, clearly poor R14 weights and only regenerable/draft outputs; all scores, source bytes, data, installed dependencies, useful models and final reports retained',
    deprecated_entry='16_直接预测共享状态_预测次日.cmd',entry_action='replace with explicit notice and route to retained round8b; keep original entry text in research cleanup record',
    state='planned')
(O/'cleanup_plan.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(targets=len(targets),files=out['total_files'],logical_bytes=out['total_logical_bytes'],gib=out['total_logical_bytes']/2**30,protected=len(prot)),ensure_ascii=False))
