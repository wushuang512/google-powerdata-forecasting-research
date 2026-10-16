from pathlib import Path
import sys,json,csv,hashlib,datetime,numpy as np
from docx import Document
T=Path(sys.argv[1]);B=T/'experiment';Q=B/'round43/results';checks=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(n,v):checks.append(dict(check=n,passed=bool(v)));assert v,n
a=read(Q/'factor_comparison.json');v=read(Q/'independent_verification.json');qa=read(B/'round43/docs/report_qa/verification.json');doc=T/'Google PowerData 2019 三项预测任务与模型比较汇报.docx'
check('126 unified records 98 unique forecasts',len(a['rows'])==126 and len(a['unique_rows'])==98)
check('126 scientific checks passed',v['all_passed'] and len(v['checks'])==v['check_count']==126 and all(c['passed'] for c in v['checks']))
check('reference document immutable',sha(T/'参考材料/9.28算力中心需求预测实验.docx')==a['reference_audit']['reference_sha256'])
check('original implementation limits still acknowledged',not a['reference_audit']['exact_original_implementation_verified'])
check('nine pages individually visually reviewed',qa['all_pages_visually_reviewed'] and qa['all_passed'] and qa['layout_passed'] and qa['pages']==9 and qa['page_numbers_reviewed']==list(range(1,10)) and qa['document_sha256']==sha(doc))
check('PDF and all page hashes',sha(qa['pdf'])==qa['pdf_sha256'] and len(qa['images'])==9 and all(sha(p['file'])==p['sha256'] for p in qa['images']))
text='\n'.join(p.text for p in Document(doc).paragraphs)
check('all three tasks and truthful limits included',all(x in text for x in ['未来24小时完整负荷曲线模型结果','未来7天完整负荷曲线模型结果','未来7天每日平均负荷模型结果','0.003072','尚未达到单一','共同因子','NCU','CC BY-NC4.0','大幅领跑']))
check('all four release gates reject deployment',all(not s['adopt'] for r in a['release_decisions'].values() for s in r.values()))
check('24h accepted DEV result properly rejected by TEST gate',a['release_decisions']['38']['1']['dev_selected'] and not a['release_decisions']['38']['1']['test_registered_gates']['eligible'])
check('current recommendation stays R32 R36',a['recommendation']['recommended24']=='round32' and a['recommendation']['recommended7']=='round36' and a['recommended_deployment_unchanged'])
check('R36 historical a5 exception retained',not read(B/'round36/results/deployment_decision.json')['original_all_tolerances_passed'])
with (T/'模型各项指标完整结果.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
check('every saved CSV metric exactly agrees with comparison',len(rows)==126 and all(all(abs(float(r[k])-v)<1e-10 for k,v in s.items() if isinstance(v,(int,float)) and not isinstance(v,bool)) for r,s in zip(rows,a['rows'])))
check('all model score numbers finite',all(np.isfinite(v) for r in a['rows'] for v in r.values() if isinstance(v,(int,float))))
check('current CSV aliases agree',sha(T/'9月28日统一口径模型复算.csv')==sha(T/'模型各项指标完整结果.csv')==sha(T/'9月28日指标完整模型对照.csv') and sha(T/'9月28日指标预测区间对照.csv')==sha(T/'模型预测区间结果.csv'))
for H,key,expected in [(1,'wape',2.367483112706865),(7,'wape',2.694095946387253),(7,'daily_wape',1.889162798887832)]:check(f'H{H} actual all-model {key} minimum',abs(min(r[key] for r in a['unique_rows'] if r['horizon_days']==H)-expected)<1e-11)
check('one common frozen method not falsely declared winner',not a['all_curves_leading_one_frozen_method'] and not a['substantial_improvement'])
rng=np.random.default_rng(20261007);sys.path.insert(0,str(B/'code'));import common;d=common.load()
for item in a['continued_bootstrap']:
 H=item['horizon_days'];nr=item['round'];oo=list(range(22,31)) if H==1 else[22,23,24]
 with np.load(B/f'round{nr}/results/test_predictions.npz') as z:new=z[f'diagnostic_H{H}'];old=z[f'selected_H{H}']
 y,m=common.truth(d,oo,H);mass=(np.abs(y)*m).sum(-1);ea=(np.abs(y-old)*m).sum(-1);eb=(np.abs(y-new)*m).sum(-1);ids=rng.integers(0,len(oo),(5000,len(oo)));vals=100*np.mean((ea[:,ids].sum(-1)-eb[:,ids].sum(-1))/mass[:,ids].sum(-1),axis=0)
 check(f'H{H} whole-PDU macro WAPE bootstrap replay',np.allclose(np.quantile(vals,[.025,.975]),item['paired_origin_bootstrap95_gain_pp'],rtol=0,atol=1e-12) and item['bootstrap_recalculates_whole_PDU_WAPE'] and not item['independent_confirmation'])
check('three factor rounds not adopted',all(not s['adopt_shared_scheme'] for s in a['factor_rounds'].values()))
entries=read(B/'round36/results/current_entry_verification.json')['entries']+read(Q/'research_entry_verification.json')['entries'];check('two current and six research entries',len(entries)==8)
for r in entries:
 H=r['horizon_days'];label=f'R{r.get("round","current")} H{H}';check(label+' source SHA',sha(r['curve'])==r['curve_sha256'] and sha(r['daily'])==r['daily_sha256'])
 with Path(r['curve']).open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
 pred=np.array([float(x['forecast_normalized_power']) for x in rr]);lower=np.array([float(x['lower80_empirical']) for x in rr]);upper=np.array([float(x['upper80_empirical']) for x in rr]);check(label+' points/57domains/finite/bounded/intervals',len(rr)==57*H*96 and len({x['pdu'] for x in rr})==57 and all(np.isfinite(x).all() for x in [pred,lower,upper]) and all(x.min()>=0 and x.max()<=1 for x in [pred,lower,upper]) and (lower<=upper).all())
 check(label+' correct June historical boundary',min(x['timestamp_local_PT'] for x in rr)=='2019-06-01T00:00' and max(x['timestamp_local_PT'] for x in rr)==f'2019-06-{H:02d}T23:45' and all(x['historical_replay']=='True' for x in rr))
 with Path(r['daily']).open(encoding='utf-8-sig') as f:dd=list(csv.DictReader(f))
 check(label+' daily average of same 96 points',len(dd)==57*H and np.max(np.abs(np.array([float(x['daily_mean_normalized_power']) for x in dd])-pred.reshape(57*H,96).mean(-1)))<1e-12)
 if 'round' in r:
  with np.load(B/f'round{r["round"]}/results/deployment.npz') as z:check(label+' exact actual deployment array',np.max(np.abs(pred-z[f'diagnostic_H{H}'].ravel()))<1e-12)
  check(label+' only research status',r['adopted']==False and r['current_live_forecast']==False)
 else:check(label+' current entry unchanged','round36\\code\\read_current_moirai.py' in (T/f'19_{H}天当前推荐模型历史复现.cmd').read_text(encoding='utf-8'))
source=read(T/'google-clusterdata-2019/results/complete_projection_audit.json');check('source audit passed',source['all_passed'])
with np.load(T/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz') as z:check('only official cell-a12085 necessary columns complete',bool(z['complete']) and int(z['source_files'])==int(z['expected_source_files'])==12085)
for cut in [14,22]:
 u=read(B/'round40/results'/f'usage_source_cut{cut}.json');check(f'CPU source cut{cut} actual ten historical channels',len(u['used_channels'])==10 and u['only_a5_realCPU'] and u['training_dedup_end_exclusive']==cut*96 and u['source_sha256']==sha(T/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz'))
rt=read(Q/'final_runtime_verification.json');check('no owned new experiment process remains',rt['all_new_rounds_ended'] and not rt['active_owned_new_model_processes']);check('desktop still compatible D junction',rt['desktop_link_type']=='Junction' and str(T) in rt['desktop_target'])
check('automation remains paused','status = "PAUSED"' in Path('${USER_HOME}/.codex/automations/google/automation.toml').read_text(encoding='utf-8'))
check('both exact rendering draft targets removed only',all(not Path(x['path']).exists() for x in read(B/'round43/docs/report_qa/layout_cleanup.json')['removed_only_superseded_report_renderings']))
o=dict(all_passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),check_count=len(checks),checks=checks,rows=126,unique_predictions=98,pages=9,report_sha256=sha(doc),comparison_sha256=sha(Q/'factor_comparison.json'),recommendation24='round32',recommendation7='round36',no_universal_winner=True,substantial_improvement=False,independent_confirmation=False,automation_status='PAUSED');(Q/'final_delivery_verification.json').write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=True,checks=len(checks),report_sha256=sha(doc),current_entries=2,research_entries=6)))
