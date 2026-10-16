from pathlib import Path
import sys,json,hashlib,datetime,csv,numpy as np
from docx import Document
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round32';Q=R/'results';checks=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(n,v):checks.append(dict(check=n,passed=bool(v)));assert v,n
a=read(Q/'continued_comparison.json');qa=read(R/'docs/report_qa/verification.json');doc=T/'Google PowerData 2019 三项预测任务与模型比较汇报.docx';ref=T/'参考材料/9.28算力中心需求预测实验.docx'
check('reference original unchanged',sha(ref)==a['reference_audit']['reference_sha256']);check('full original implementation uncertainty stated',not a['reference_audit']['exact_original_implementation_verified']);check('86 uniform rows and78 unique',len(a['rows'])==86 and len(a['unique_rows'])==78);check('new independent science verification',read(B/'round29/results/new_methods_verification.json')['all_passed']);check('recommendations retained by DEV',a['recommendation']['recommended24']=='round32' and a['recommendation']['recommended7']=='round31' and all(a['new_rounds'][str(n)]['selection']['retain_controls'] for n in [29,30]));check('latest9page Word SHA and visualQA',qa['all_passed'] and qa['pages']==9 and qa['page_numbers_reviewed']==list(range(1,10)) and qa['document_sha256']==sha(doc));check('QA page images andPDF match',all(sha(v['file'])==v['sha256'] for v in qa['images']) and sha(qa['pdf'])==qa['pdf_sha256'])
text='\n'.join(p.text for p in Document(doc).paragraphs);check('report uncertainty and3 tasks explicit','不能声称已完全复现' in text and '未来24小时完整曲线模型结果' in text and '未来7天每日平均负荷模型结果' in text and '不声称所有指标第一' in text)
with (T/'模型各项指标完整结果.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
check('86 CSV records consistent scores',len(rows)==86 and all(abs(float(r['wape'])-s['wape'])<1e-11 and abs(float(r['daily_wape'])-s['daily_wape'])<1e-11 for r,s in zip(rows,a['rows'])));check('top-level aliases aligned',sha(T/'9月28日指标完整模型对照.csv')==sha(T/'模型各项指标完整结果.csv') and sha(T/'9月28日指标预测区间对照.csv')==sha(T/'模型预测区间结果.csv'))
with np.load(B/'results/data.npz') as z:y=z['y']
for mode in ['head','all']:
 for cutoff in [14,22]:
  train=read(B/f'round30/results/train_{mode}_cut{cutoff}.json');mp=B/f'round30/models/{mode}_cut{cutoff}'
  with np.load(mp/'training_scaler.npz') as z:mu=z['mean'];sd=z['std']
  check(f'{mode} scaler and labels train-only{cutoff}',np.array_equal(mu,y[:,:cutoff].reshape(57,-1).mean(-1)) and np.array_equal(sd,y[:,:cutoff].reshape(57,-1).std(-1).clip(1e-4)) and train['last_label_index_exclusive']==cutoff*96 and train['epochs']==4 and sha(mp/'model.safetensors')==train['weights_sha256'])
check('R31/32 independent reconstruction',read(Q/'independent_verification.json')['all_passed'])
check('R33 independent interval verification',read(B/'round33/results/independent_verification.json')['all_passed'])
for r in read(Q/'current_entry_verification.json')['entries']:
 H=r['horizon_days'];check(f'current{H} entry SHA',sha(r['curve'])==r['curve_sha256'] and sha(r['daily'])==r['daily_sha256'])
 with Path(r['curve']).open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
 check(f'current{H} dates numbers intervals',len(records)==57*H*96 and len({v['pdu'] for v in records})==57 and min(v['timestamp_local_PT'] for v in records)=='2019-06-01T00:00' and max(v['timestamp_local_PT'] for v in records)==('2019-06-01T23:45' if H==1 else'2019-06-07T23:45') and all(0<=float(v['forecast_normalized_power'])<=1 and 0<=float(v['lower80_empirical'])<=float(v['upper80_empirical'])<=1 for v in records))
 with Path(r['daily']).open(encoding='utf-8-sig') as f:days=list(csv.DictReader(f))
 check(f'current{H} daily coherence',len(days)==57*H and all(abs(float(days[i*H+j]['daily_mean_normalized_power'])-np.mean([float(v['forecast_normalized_power']) for v in records[(i*H+j)*96:(i*H+j+1)*96]]))<1e-12 for i in range(57) for j in range(H)))
for H in [1,7]:check(f'current{H} entry preserved',(T/f'19_{H}天当前推荐模型历史复现.cmd').exists())
check('data downloads zero andcpu complete',read(T/'google-clusterdata-2019/results/projection_progress_a.json')['completed']==12085 if 'completed' in read(T/'google-clusterdata-2019/results/projection_progress_a.json') else read(T/'google-clusterdata-2019/results/projection_progress_a.json')['processed_files']==12085)
check('automation remains paused','status = "PAUSED"' in Path('${USER_HOME}/.codex/automations/google/automation.toml').read_text(encoding='utf-8'))
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=all(c['passed'] for c in checks),checks=checks,check_count=len(checks),report_sha256=sha(doc),metric_version=a['metric_version'],evaluation_source_arrays_sha256=sha(Q/'continued_comparison.json'),inventory_scope='Full inventory is snapshotted after this file and then checked independently without rewriting scientific results.',no_universal_upgrade=True,independent_confirmation=False)
(Q/'final_delivery_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=out['all_passed'],check_count=len(checks),report_sha256=sha(doc)),ensure_ascii=False))
