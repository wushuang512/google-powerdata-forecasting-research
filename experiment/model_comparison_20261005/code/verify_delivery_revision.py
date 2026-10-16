from pathlib import Path
import json,hashlib,os
import pandas as pd
from pypdf import PdfReader
T=Path(r'${PROJECT_ROOT}');R=T/'experiment/model_comparison_20261005/results/task_revision';Q=T/'experiment/model_comparison_20261005/docs/report_qa_task_revision'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
M=read(T/'交付文件清单.json');S=read(R/'summary.json');V=read(R/'final_delivery_verification.json');QA=read(Q/'verification.json')
actual={}
for base,dirs,files in os.walk(T,followlinks=False):
 dirs[:]=[n for n in dirs if not (Path(base)/n).is_symlink()]
 for n in files:
  p=Path(base)/n
  if p==T/'交付文件清单.json':continue
  actual[p.relative_to(T).as_posix()]=p.stat().st_size
expected={a['path']:a['bytes'] for a in M['all_files']}
assert actual==expected and len(actual)==M['files'] and sum(actual.values())==M['bytes']
assert sha(T/M['latest_report'])==QA['document_sha256']==V['report_sha256']
assert sha(Q/'report_final.pdf')==QA['pdf_sha256'] and len(PdfReader(Q/'report_final.pdf').pages)==15
assert QA['page_numbers_reviewed']==list(range(1,16)) and all(sha(Path(i['file']))==i['sha256'] for i in QA['images'])
assert QA['all_pages_visually_reviewed'] and QA['all_passed'] and V['all_passed']
f=pd.read_csv(T/'7天完整曲线模型比较.csv');g=pd.read_csv(T/'7天每日平均负荷模型比较.csv');assert len(f)==len(g)==18
assert f.iloc[:,2].notna().all() and g.iloc[:,2].notna().all() and not f.iloc[:,1].str.contains('未评估').any()
assert abs(f.iloc[:,2].min()-2.7797074655345204)<1e-12 and abs(g.iloc[:,2].min()-2.00147685008163)<1e-12
assert len(pd.read_csv(R/'seven_day_daily_means_detail.csv'))==21546
assert S['actual_missing_tasks_completed'] and len(S['checks'])==58 and all(S['checks'].values())
for stage in ['tf3_uni21_7','tf3_finetuned7','chronos_mirror7','tf25_7']:
 meta=read(T/f'experiment/seven_day_completion_20261005/cache/{stage}_control.json');assert meta['checks']['max_abs']<1e-5
for r in [15,16]:assert read(T/f'experiment/round{r}/results/independent_verification.json')['all_passed']
print(json.dumps(dict(all_passed=True,independent_read_only=True,pages=15,files=len(actual),bytes=sum(actual.values()),seven_methods=18,mean_details=21546,actual_missing_methods=5,joint_trials=2),ensure_ascii=False))
