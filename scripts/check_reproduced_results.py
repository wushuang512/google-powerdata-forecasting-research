"""Compare newly computed metric tables to published references; never make forecasts."""
import argparse,csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--task',choices=['google','nlr'],required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
 if a.task=='google':
  reference=[r for r in rows(ROOT/'本轮方案全部指标对照.csv') if r['role']=='DEV_frozen_R48_candidate'];actual=rows(a.work/'reproduction/results/google_R48_all_metrics.csv')
  key=lambda r:(r['horizon_days'],);fields=['wape','daily_wape','mae','rmse','p95_ape','peak_mape','peak_time_mae_minutes','hit2_percent','hit5_percent','absolute_bias_percent','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent','valid_points','valid_daily_targets','peak_underestimate_percent','interval80_coverage_percent','interval80_mean_width_normalized','interval80_winkler80']
 else:
  reference=rows(ROOT/'NLR严格重跑_新协议主要模型测试全指标.csv');actual=rows(a.work/'experiment/nlr_strict_protocol_20261013/results/新协议主要模型测试全指标.csv')
  key=lambda r:(r['cohort'],r['horizon_days'],r['model']);fields=['curve_mape_percent','curve_rmse','curve_wape_percent','daily_mape_percent','daily_rmse','daily_wape_percent','peak_mape_percent','peak_time_mae_minutes','p95_ape_percent','mape_n','mape_excluded','origins','points']
 expected={key(r):r for r in reference};got={key(r):r for r in actual};checks=[]
 for k,r in expected.items():
  if k not in got:checks.append({'key':k,'error':'Missing reproduced model/task'});continue
  for field in fields:
   if field not in r or field not in got[k]:checks.append({'key':k,'field':field,'error':'Missing metric'});continue
   x,y=r[field],got[k][field]
   if x in ['','null','None'] or y in ['','null','None']:passed=x in ['','null','None'] and y in ['','null','None'];diff=None;tol=0
   else:
    x,y=float(x),float(y);diff=y-x
    tol=0 if field in ['valid_points','valid_daily_targets','mape_n','mape_excluded','origins','points'] else 1. if field=='peak_time_mae_minutes' else 1e-3 if a.task=='nlr' and 'rmse' in field else 1e-6 if any(t in field for t in ['mae','rmse','width','winkler']) and 'mape' not in field else 5e-4
    passed=math.isfinite(x) and math.isfinite(y) and abs(diff)<=tol
   checks.append({'key':k,'metric':field,'difference':diff,'absolute_tolerance':tol,'passed':passed})
 result={'task':a.task,'reference_rows':len(expected),'reproduced_rows':len(got),'same_model_task_keys':set(expected)==set(got),'all_passed':set(expected)==set(got) and all(x.get('passed',False) for x in checks),'checks':checks,'not_an_inference_or_training_test':True,'tolerance_does_not_imply_bitwise_reproducibility':True}
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False));raise SystemExit(0 if result['all_passed'] else 1)
if __name__=='__main__':main()
