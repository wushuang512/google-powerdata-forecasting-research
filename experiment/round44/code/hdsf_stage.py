from pathlib import Path
import sys,json,hashlib,datetime,argparse,copy
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);ap.add_argument('--member');ap.add_argument('--phase');args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round44';Q=R/'results';repo=R/'reference/PowerData_Prediction';sys.path[:0]=[str(repo),str(B/'code'),str(B/'round29/code')]
import numpy as np,joblib
from power_forecast.data import load_dataset
from power_forecast.config import TASKS
from power_forecast.hdsf_experiment import HDSFCollector,HDSF_MEMBERS,selection_policy
from power_forecast.attainment_local import fit_local24
from power_forecast.attainment_fusion import select_mix,apply_mix,nested_loo
import common
from aligned_metrics import evaluate
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,a):(Q/name).write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
def dataset():return joblib.load(Q/'source_dataset.joblib')
def align(data,p):
 old=common.load();index=[data.series_keys.index(str(x)) for x in old['names']];return p[index]
def array(data,frame,column):
 oo=sorted(map(int,frame.origin_day.unique()));H=1 if len(frame.loc[frame.origin_day==oo[0]])==57*96 else 7;result=np.zeros((57,len(oo),H*96))
 for j,o in enumerate(oo):
  part=frame.loc[frame.origin_day==o];ids=part.series_id.to_numpy(int);steps=(part.target_day.to_numpy(int)-o-1)*96+part.slot.to_numpy(int);assert len(part)==57*H*96;result[ids,j,steps]=part[column].to_numpy(float)
 return align(data,result),oo,H
def load_frames(phase):
 import pandas as pd
 frames={}
 for task,methods in HDSF_MEMBERS.items():
  parts=[joblib.load(Q/'member_frames'/f'{phase}_{m}.joblib')[task] for m in methods];frame=parts[0].copy()
  for method,part in zip(methods,parts):
   assert np.array_equal(frame[['origin_day','series_id','target_day','slot']].to_numpy(),part[['origin_day','series_id','target_day','slot']].to_numpy());frame[method]=part[method].to_numpy()
  frames[task]=frame
 return frames
if args.stage=='data':
 data=load_dataset(T/'google-powerdata-2019/raw');old=common.load();idx=[data.series_keys.index(str(x)) for x in old['names']];m=data.target_valid[idx];diff=abs(data.target[idx]-old['y']);assert np.array_equal(m,old['valid']) and diff[m].max()<1e-12 and data.target.shape==(57,31,96);joblib.dump(data,Q/'source_dataset.joblib');save('data_alignment.json',dict(all_passed=True,shape=list(data.target.shape),source_quality=data.quality,modeling_order=list(data.series_keys),canonical_order=list(map(str,old['names'])),reorder_index=idx,max_valid_truth_difference=float(diff[m].max()),invalid_context_preprocessing_max_difference=float(diff[~m].max()),mask_equals_previous=True,valid_truth_equals_previous=True,source_dataset_sha256=sha(Q/'source_dataset.joblib'),canonical_input_sha256=sha(B/'results/data.npz'),source_day_major_sha256=hashlib.sha256(np.ascontiguousarray(data.target.transpose(1,2,0)).tobytes()).hexdigest()));print('Exact source data loaded; evaluation valid truthandmask identical; modeling order preserved',flush=True)
elif args.stage=='local':
 data=dataset();cut=14 if args.phase=='dev' else 22;model=fit_local24(data,cut,cache_dir=R/'model-cache',jobs=8);joblib.dump(model,Q/f'local_cut{cut}.joblib');save(f'local_cut{cut}.json',dict(cutoff=model.cutoff,models=len(model.models),trees=[m.current_iteration() for m in model.models],lags=list(model.lags),coefficients=model.coefficients.tolist(),source_issue_D1_preserved=True,model_sha256=sha(Q/f'local_cut{cut}.joblib'),training_data_sha256=hashlib.sha256(np.ascontiguousarray(data.target[:,:cut]).tobytes()).hexdigest()));print('Fitted original fourseedlocal model',cut,flush=True)
elif args.stage=='member':
 data=dataset();phase=args.phase;member=args.member;tasks=[t for t,mm in HDSF_MEMBERS.items() if member in mm]
 if member=='patchtstfm_long':
  from patch_low_memory import load_predictor as patch_factory
 else:patch_factory=None
 col=HDSFCollector(data,tasks,cache_dir=R/'model-cache',cache_predictions=member!='patchtstfm_long',batch_size=16,jobs=8,device='cpu',predictor_factory=patch_factory);col.members={t:(member,) for t in tasks}
 if member=='local24':
  if phase=='dev':col.development_local=joblib.load(Q/'local_cut14.joblib')
  else:col.frozen_local=joblib.load(Q/'local_cut22.joblib')
 assert phase=='dev' or (Q/'frozen_before_test.json').exists();frames=col.collect(development=phase=='dev');dest=Q/'member_frames';dest.mkdir(exist_ok=True);joblib.dump(frames,dest/f'{phase}_{member}.joblib');save(f'{phase}_{member}_complete.json',dict(phase=phase,member=member,source_frame_sha256=sha(dest/f'{phase}_{member}.joblib'),tasks=tasks,origin_sets={t:sorted(map(int,f.origin_day.unique())) for t,f in frames.items()},completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print(phase,member,'membercomplete',flush=True)
elif args.stage=='freeze':
 data=dataset();dev=load_frames('dev');configs={t:select_mix(f,t,methods=HDSF_MEMBERS[t],selection_policy=selection_policy(t)) for t,f in dev.items()};configs['24h']['nested_loo']=nested_loo(dev['24h'],'24h',methods=HDSF_MEMBERS['24h'],selection_policy=selection_policy('24h'))
 for c in configs.values():c['development_local_training_label_end']=14
 assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=configs,no_test_used=True,protocol_sha256=sha(R/'docs/protocol.json'),resource_fallback_sha256=sha(R/'docs/resource_fallback.json'),patch_adapter_sha256=sha(R/'code/patch_low_memory.py'),development_member_sha256={m:sha(Q/'member_frames'/f'dev_{m}.joblib') for m in dict.fromkeys(m for mm in HDSF_MEMBERS.values() for m in mm)},data_alignment_sha256=sha(Q/'data_alignment.json'),official_new_weights_manifest_sha256=sha(Q/'official_models_downloads.json'),shared_weights_manifest_sha256=sha(Q/'shared_weights.json'),runtime_versions_sha256=sha(R/'docs/runtime_versions.json'),granite_source_manifest_sha256=sha(Q/'granite_source.json')));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));joblib.dump(dev,Q/'development_frames.joblib');print('Original HDSF DEV configs frozen',json.dumps({t:c['mix'] for t,c in configs.items()}),flush=True)
elif args.stage=='before_test':
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();save('frozen_before_test.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(Q/'selection.json'),final_local_sha256=sha(Q/'local_cut22.joblib'),protocol_sha256=sha(R/'docs/protocol.json'),model_source_snapshot=read(R/'docs/protocol.json')['source_sha256'],source_issue_D1_preserved=True));print('FinalD22 andfusion frozen before first TESTmember',flush=True)
elif args.stage=='score':
 from power_forecast.compact_experiment import attach_labels
 from power_forecast.evaluation import summarize_predictions,daily_mean_metrics
 data=dataset();d=common.load();s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();dev=load_frames('dev');test=load_frames('test');out={};scores={};source_scores={};devscores={}
 for task,frame in test.items():
  cfg=s['configs'][task];frame['hdsf']=apply_mix(frame.drop(columns=['y_true','target_valid'],errors='ignore'),cfg);attach_labels(frame,data,development=False);dv=dev[task];dv['hdsf']=apply_mix(dv.drop(columns=['y_true','target_valid'],errors='ignore'),cfg);methods=[*HDSF_MEMBERS[task],'hdsf'];source_scores[task]=dict(curve=summarize_predictions(frame,methods)[0],daily=daily_mean_metrics(frame,methods));H=TASKS[task].horizon_days;scores[str(H)]={};devscores[str(H)]={}
  for method in methods:
   p,oo,_=array(data,frame,method);out[f'{method}_H{H}']=p;scores[str(H)][method]=evaluate(common,d,oo,p,H);dp,do,_=array(data,dv,method);devscores[str(H)][method]=evaluate(common,d,do,dp,H);out[f'dev_{method}_H{H}']=dp
 np.savez_compressed(Q/'predictions.npz',**out);save('summary.json',dict(scores=scores,development=devscores,source_scores=source_scores,selection=s,canonical_metric_sha256=sha(B/'round29/code/aligned_metrics.py'),test_prediction_sha256=sha(Q/'predictions.npz'),source_score_not_comparable_to_previous_headline=True,independent_confirmation=False));joblib.dump(test,Q/'test_frames.joblib');print('Source andunchangedcanonical metrics both saved separately',flush=True)
elif args.stage=='audit':
 import subprocess
 targets=['tests/test_hdsf.py','tests/test_hdsf_training_audit.py','tests/test_attainment.py','tests/test_evaluation.py','tests/test_data.py']
 cp=subprocess.run([sys.executable,'-m','pytest','-q',*targets,'--junitxml='+str(Q/'source_tests.xml')],cwd=repo,capture_output=True,text=True,encoding='utf-8');(Q/'source_tests.log').write_text(cp.stdout+'\n'+cp.stderr,encoding='utf-8');assert cp.returncode==0,cp.stdout[-6000:]+cp.stderr[-2000:]
 from dataclasses import replace
 from scripts.reproduction.audit_hdsf_training import model_signature,audit_training_features,audit_d1
 data=dataset();checks=[]
 def check(n,v):
  checks.append(dict(name=n,passed=bool(v)));assert v,n
 protocol=read(R/'docs/protocol.json');check('canonical_code_unmodified',sha(B/'round29/code/aligned_metrics.py')==protocol['evaluation_sha256']);check('common_unmodified',sha(B/'code/common.py')==protocol['common_sha256'])
 for f,h in protocol['source_sha256'].items():check('source_unmodified '+f,sha(repo/f)==h)
 check('selection_sha_frozen',sha(Q/'selection.json')==(Q/'selection.sha256').read_text());check('data_mask_and_valid_truth_equal',read(Q/'data_alignment.json')['all_passed'])
 poisoned=replace(data,target=data.target.copy(),production=data.production.copy(),production_causal_median=data.production_causal_median.copy())
 for cut,oo in [(14,[14,16,18,20]),(22,list(range(22,31)))]:
  model=joblib.load(Q/f'local_cut{cut}.joblib');signature=model_signature(model)
  poisoned.target=data.target.copy();poisoned.production=data.production.copy();poisoned.production_causal_median=data.production_causal_median.copy();poisoned.target[:,cut:]=999;poisoned.production[:,cut:]=999;poisoned.production_causal_median[:,cut:]=999
  refit=fit_local24(poisoned,cut,cache_dir=None,jobs=8);check(f'uncached_full_4seed_fit_future_poison_cut{cut}',signature==model_signature(refit));del refit
  for o in oo:
   poisoned.target=data.target.copy();poisoned.production=data.production.copy();poisoned.production_causal_median=data.production_causal_median.copy();poisoned.target[:,o:]=999;poisoned.production[:,o:]=999;poisoned.production_causal_median[:,o:]=999
   check(f'local_predict_future_poison_D{o}_fit{cut}',np.array_equal(model.predict(data,o),model.predict(poisoned,o)))
  check(f'local_parameters_unchanged_fit{cut}',model_signature(model)==signature)
 configs=read(Q/'selection.json')['configs'];pred=np.load(Q/'predictions.npz');bounds={}
 for task,cfg in configs.items():
  for name,groups in [('a',cfg['component_a']),('b',cfg['component_b'])]:
   for seg,w in groups.items():
    a=np.asarray(w);check(f'simplex {task} {name} {seg}',np.isfinite(a).all() and (a>=0).all() and np.allclose(a.sum(1),1,atol=1e-10))
  H=TASKS[task].horizon_days;p=pred[f'hdsf_H{H}'];check(f'finite_H{H}',np.isfinite(p).all());check(f'shape_H{H}',p.shape==(57,9 if H==1 else 3,H*96));stats=read(Q/'summary.json')['scores'][str(H)]['hdsf'];check(f'validpoints_H{H}',stats['valid_points']==(49243 if H==1 else 114902));check(f'daily_targets_H{H}',stats['valid_daily_targets']==(508 if H==1 else 1187));bounds[str(H)]=dict(minimum=float(p.min()),maximum=float(p.max()),outside_unit_interval=int(((p<0)|(p>1)).sum()),silent_clipping=False)
 save('science_verification.json',dict(all_passed=all(x['passed'] for x in checks),checks=checks,check_count=len(checks),source_pytest_passed=True,source_tests_log_sha256=sha(Q/'source_tests.log'),bounds=bounds,source_issue_D1_preserved=True,independent_confirmation=False));print('Source tests andindependent causal freeze/scoring audit passed',len(checks),flush=True)
elif args.stage=='gpu_probe':
 import torch,time,gc
 from power_forecast.new_foundation import load_new_foundation
 from power_forecast.multiscale_experiment import toto_groups
 assert torch.cuda.is_available();assert (Q/'test_toto2_joint_complete.json').exists();data=dataset();o=22;context=data.target[:,o-14:o].reshape(57,-1).astype(np.float32);groups=toto_groups(data);frames=joblib.load(Q/'member_frames/test_toto2_joint.joblib')['24h'];part=frames.loc[frames.origin_day==o];reference=np.empty((57,96));reference[part.series_id.to_numpy(int),part.slot.to_numpy(int)]=part.toto2_joint.to_numpy(float)
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter();model=load_new_foundation('toto2',cache_dir=R/'model-cache',batch_size=16,jobs=8,device='cuda',groups=groups);load_seconds=time.perf_counter()-started;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.synchronize();started=time.perf_counter();prediction=model.predict(context,96).astype(np.float32).astype(float);torch.cuda.synchronize();seconds=time.perf_counter()-started;diff=abs(prediction-reference);before=read(Q/'selection.json');gpu=evaluate(common,common.load(),[o],align(data,prediction[:,None,:]),1);cpu=evaluate(common,common.load(),[o],align(data,reference[:,None,:]),1);np.savez_compressed(Q/'gpu_toto2_joint_D22_probe.npz',gpu=prediction,cpu=reference)
 save('gpu_probe.json',dict(passed=bool(np.isfinite(prediction).all()),torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0),device_total_bytes=torch.cuda.get_device_properties(0).total_memory,actual_gpu=True,prediction_dtype='float32',tf32=False,model='Datadog Toto2.0 313M 同cell/设备类型联合14天',origin_day=o,horizon=96,load_seconds=load_seconds,inference_seconds=seconds,peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),max_absolute_difference=float(diff.max()),mean_absolute_difference=float(diff.mean()),cpu_single_origin_metrics=cpu,gpu_single_origin_metrics=gpu,formal_HDSF_result_remains_original_CPU=True,no_test_selection=True,cpu_freeze_sha256=sha(Q/'selection.json'),runtime_differs_from_source_cpu_torch28=True));print('Actual GPUprobe inference',seconds,'seconds; maxCPUdifference',float(diff.max()),flush=True)
