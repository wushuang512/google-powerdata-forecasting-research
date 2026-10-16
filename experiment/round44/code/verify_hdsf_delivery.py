from pathlib import Path
import sys,json,hashlib,csv,math
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round44';Q=R/'results';sys.path[:0]=[str(B/'code'),str(B/'round29/code'),str(B/'round28/code')]
import common
from aligned_metrics import evaluate
from source_metrics import interval_metrics
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
a=read(Q/'hdsf_comparison.json');s=read(Q/'summary.json');selection=read(Q/'selection.json');protocol=read(R/'docs/protocol.json');d=common.load();z=np.load(Q/'predictions.npz');check('canonical_reference_rule',a['reference_audit']['rule']=='perPDU WAPE then equal average57');check('canonical_code_unchanged',sha(B/'round29/code/aligned_metrics.py')==protocol['evaluation_sha256']);check('common_unchanged',sha(B/'code/common.py')==protocol['common_sha256']);check('freeze_unchanged',sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and selection['no_test_used']);check('freeze_precedes_TEST',all((Q/'selection.json').stat().st_mtime<(Q/'member_frames'/f'test_{m}.joblib').stat().st_mtime for m in selection['development_member_sha256']));check('patch_variation_registered_before_freeze',(R/'docs/resource_fallback.json').stat().st_mtime<(Q/'selection.json').stat().st_mtime and selection['patch_adapter_sha256']==sha(R/'code/patch_low_memory.py'))
for H in [1,7]:
 oo=common.TEST24 if H==1 else common.TEST7;do=common.DEV24 if H==1 else common.DEV7;y,m=common.truth(d,oo,H);yd,md=common.truth(d,do,H)
 for method,stored in s['scores'][str(H)].items():
  p=z[f'{method}_H{H}'];dp=z[f'dev_{method}_H{H}'];check(f'{method} H{H} exact metric replay',evaluate(common,d,oo,p,H)==stored);check(f'{method} H{H} DEV exact replay',evaluate(common,d,do,dp,H)==s['development'][str(H)][method]);iv=np.load(Q/f'interval_{method}_H{H}.npz');lo=np.array([np.quantile((yd-dp)[i][md[i]],.1) for i in range(57)]);hi=np.array([np.quantile((yd-dp)[i][md[i]],.9) for i in range(57)]);check(f'{method} H{H} interval onlyDEV',np.array_equal(lo,iv['loq']) and np.array_equal(hi,iv['hiq']) and np.array_equal(iv['lower'],np.clip(p+lo[:,None,None],0,1)) and np.array_equal(iv['upper'],np.clip(p+hi[:,None,None],0,1)));calc=interval_metrics(y,p,m,iv['lower'],iv['upper']);stored_iv=next(v for v in a['new_intervals'] if v['horizon_days']==H and v['method'].startswith(next(r['model_name'] for r in a['new_rows'] if r['model']==f'r44_{method}_H{H}')));check(f'{method} H{H} interval scoring',all(calc[k]==stored_iv[k] for k in calc))
 for daily in [False,True]:
  file=Q/(f'HDSF_H{H}_每日平均负荷.csv' if daily else f'HDSF_H{H}_历史滚动预测.csv');count=0;valid=0;good=True;p=z[f'hdsf_H{H}'];lookup={str(n):i for i,n in enumerate(d['names'])}
  with file.open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):
    count+=1;i=lookup[r['series_key']];j=oo.index(int(r['origin_day']))
    if daily:
     k=int(r['target_day'])-int(r['origin_day'])-1;actual=float(p[i,j,k*96:(k+1)*96].mean());good &= float(r['power_daily_mean'])==actual;valid+=r['all96valid']=='True'
    else:
     k=(count-1)%(H*96);good &= float(r['power_normalized'])==p[i,j,k] and math.isfinite(float(r['power_normalized']));valid+=r['target_valid']=='True'
  check(f'H{H} full historical export '+('daily' if daily else 'curve'),good and count==(57*len(oo)*H if daily else 57*len(oo)*H*96) and valid==((508 if H==1 else 1187) if daily else (49243 if H==1 else 114902)))
check('source science actualpass',read(Q/'science_verification.json')['all_passed']);check('GPU actual not substituted',read(Q/'gpu_probe.json')['actual_gpu'] and read(Q/'gpu_probe.json')['formal_HDSF_result_remains_original_CPU']);check('GPU shared freeze',read(Q/'gpu_probe.json')['cpu_freeze_sha256']==sha(Q/'selection.json'))
qa=read(R/'docs/report_qa/verification.json');doc=T/'HDSF半日单纯形融合复现与统一评价汇报.docx';check('report actualallpageQA',qa['all_passed'] and qa['all_pages_visually_reviewed'] and qa['layout_passed'] and qa['document_sha256']==sha(doc) and qa['page_numbers_reviewed']==list(range(1,qa['pages']+1)))
for img in qa['images']:check('final reviewed image '+Path(img['file']).name,sha(img['file'])==img['sha256'])
check('PDF reviewed hash',sha(qa['pdf'])==qa['pdf_sha256'])
for src,dest in [('HDSF与此前模型全部指标比较.csv','模型各项指标完整结果.csv'),('各任务单项最优结果_含HDSF.csv','各任务单项最优结果.csv'),('预测区间结果_含HDSF.csv','模型预测区间结果.csv')]:check('top CSV '+dest,sha(Q/src)==sha(T/dest))
check('entry32 exists',(T/'32_HDSF半日单纯形融合复现.cmd').exists());check('old deployment retained',a['old_deployment_retained'] and all((T/n).exists() for n in ['19_1天当前推荐模型历史复现.cmd','19_7天当前推荐模型历史复现.cmd']))
for H in [1,7]:
 prior=next(r for r in a['unique_rows'] if r['horizon_days']==H and r['status']=='当前综合推荐');check(f'old H{H} prediction bytes retained',sha(B/prior['source']['path'])==prior['source']['sha256'])
check('official checkpoints actually verified',read(Q/'final_checkpoint_verification.json')['all_passed']);check('GPU metric identicalcode',read(Q/'gpu_probe.json')['canonical_metric_sha256']==sha(B/'round29/code/aligned_metrics.py'))
print(json.dumps(dict(all_passed=all(x['passed'] for x in checks),check_count=len(checks),checks=checks,report_sha256=sha(doc),rows=len(a['rows']),unified_metric_unchanged=True,independent_confirmation=False),ensure_ascii=False,indent=2))
