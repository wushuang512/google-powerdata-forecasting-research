from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';Q=B/'round20/results';a=json.loads((Q/'original_metrics_complete.json').read_text(encoding='utf-8'));sys.path.insert(0,str(B/'code'));import common
d=common.load();checks={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks['all_original_categories_and_reference']=a['all_original_metric_categories_covered'] and len(a['reference_inventory']['metrics'])==11 and sha(T/'参考材料/9.28算力中心需求预测实验.docx')==a['reference_inventory']['reference_sha256']
checks['actual_fiftyeight_named_rows']=len(a['rows'])==58 and len(list(csv.DictReader((T/'9月28日指标完整模型对照.csv').open(encoding='utf-8-sig'))))==58 and all(r['model_name'] for r in a['rows'])
for H in [1,7]:
 row=next(r for r in a['rows'] if r['model']=='current_selected' and r['horizon_days']==H)
 with np.load(B/'round26/results/test_predictions.npz') as z:p=z[f'selected{H}']
 y,m=common.truth(d,list(range(22,31)) if H==1 else[22,23,24],H);macro=[];ape=[]
 for i in range(57):
  yy=y[i][m[i]];pp=p[i][m[i]];macro.append(100*sum(abs(pp-yy))/sum(abs(yy)));ape.extend((100*abs(pp-yy)/abs(yy)).tolist())
 checks[f'macro{H}']=abs(np.mean(macro)-row['wape'])<1e-10
 checks[f'thresholds{H}']=abs(100*sum(x<=2 for x in ape)/len(ape)-row['hit2_percent'])<1e-10 and abs(100*sum(x<=5 for x in ape)/len(ape)-row['hit5_percent'])<1e-10
 baseline=common.metrics(y,common.baseline(d,list(range(22,31)) if H==1 else[22,23,24],H,kind='last' if H==1 else'mean3'),m)['wape']
 checks[f'original_baseline{H}']=abs(baseline-row['baseline_wape'])<1e-12 and abs(100*(1-row['wape']/baseline)-row['relative_curve_error_reduction_percent'])<1e-12
checks['tirex_quote_not_fake_new_measurement']=any(r['kind'].startswith('历史引用') and r['coverage_percent']==56.6 and r['mean_width_normalized'] is None for r in a['intervals'])
checks['all_seven_completed_and_verified']=all(json.loads((B/f'round{n}/results/independent_verification.json').read_text(encoding='utf-8'))['all_passed'] for n in range(20,27))
assert all(checks.values()),checks
v=dict(all_passed=True,checks={k:bool(v) for k,v in checks.items()},summary_sha256=sha(Q/'original_metrics_complete.json'),csv_sha256=sha(T/'9月28日指标完整模型对照.csv'),interval_csv_sha256=sha(T/'9月28日指标预测区间对照.csv'),independent_confirmation=False)
(Q/'original_metrics_final_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=True,checks=len(checks))))

