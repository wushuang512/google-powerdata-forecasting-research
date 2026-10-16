from pathlib import Path
import datetime,hashlib,json,sys
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round46';A=B/'round44'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for sub in ['code','docs','models','results']:(R/sub).mkdir(parents=True,exist_ok=True)
pre={f:read(R/f'results/preflight/{f}.json') for f in ['chronos2','timesfm25','timesfm3','toto2','patchtst','tirex2']}
assert all(x['all_passed'] for x in pre.values())
families={'chronos2':['chronos2_long'],'tirex2':['tirex2'],'timesfm25':['timesfm25'],
          'timesfm3':['timesfm30','timesfm30_long','timesfm30_uni'],'toto2':['toto2_joint','toto2_single'],'patchtst':['patchtstfm_long']}
p={'round':46,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_request':'Actually retrain/adapt every HDSF bottom model to Googleload, notlocalGBMonly.',
   'dataset':'Original GooglePowerData2019 only;57PDU,31days,15min. NLR andCPU ClusterData notused.',
   'script_sha256':sha(R/'code/hdsf_foundation_stage.py'),'adapter_sha256':sha(R/'code/hdsf_foundation_adapters.py'),
   'source_data_sha256':sha(A/'results/source_dataset.joblib'),'source_predictions_sha256':sha(A/'results/predictions.npz'),
   'source_selection_sha256':sha(A/'results/selection.json'),'canonical_evaluation_sha256':sha(B/'round29/code/aligned_metrics.py'),
   'families':families,'local_members':['local24','local7'],'member_model_scopes':['head','tail'],
   'head_lr':3e-5,'tail_lr':3e-6,'DEV_steps':80,'DEV_checkpoints':[40,80],'fixed_lr_schedule_denominator':80,
   'optimizer':'AdamW, weightdecay.01, normclip1, seed20261012; model.eval for deterministic forward while real gradientsandupdates enabled.',
   'train_batch_series':2,'training_context_days':[3,7,14],'training_origin_stride_points':48,
   'training_horizons':{'chronos2':[96],'tirex2':[96],'timesfm25':[96],'timesfm3':[96,672],'toto2':[96,672],'patchtst':[672]},
   'supervised_loss':'Per-series normalizedabsolute curve +.25completedaymean +.1dailymaximum +.05nativepinball +.05firstdifference +.1teacherpointMSE +.05relativeparameterprox +.1boundpenalty.',
   'minimum_history_days':3,'training_label_end_dev':14,'training_label_end_final':22,
   'no_true_future_inputs':'Only observed powerhistory enters neural forward; training future truth is loss target only. No futuretrue production/CPU/weather.',
   'members_shared_backbone':'TimesFM3 three modesandToto two modes share each pretrained checkpoint; common trained weights evaluated for every original mode/history. Notnine separate pretrained architectures.',
   'device':{f:'CPU native PyTorch;CUDA buildtoolchain absent' if f=='tirex2' else'CUDA' for f in families},
   'preflight':'Allsix families completed true single-step gradientandhash-change checks; notcounted as fullfinetuning. Failed interfaces retainedinpreflight/history.',
   'prediction_control':'Native same-device originalweights forecast beforetraining. For comparisons use sourceCPUcachedpred+(updatedmatcheddevice-originalmatcheddevice). Zero delta exactlypreservesoldscore; device/precision change notcounted asfine-tuning gain.',
   'local24':'Fresh4seedD1-excluded original350round15leaf model, unchangedlag blend. No residual labelsfromD1.',
   'local7':'Refit9predeclared causal smoother configurations on TRAIN rolling7daylabels only: levelhalflifepairs2/4,3/5,4/7 andshapehalflives5/10/20; history24days.',
   'family_selection':'EarlyDEV H1 targetD15/D17 andH7 target≤D18 choose head/tail and40/80 steps bymean member/task macro curveanddaily errorratios. LaterDEV target≥D19 choose alpha0/.25/.5/1 conservatively using pairedcalendar-dateoneSE; neverTEST.',
   'fusion_registry':[{'hours':12,'overlap':False,'shrink':0},{'hours':12,'overlap':True,'shrink':.5},{'hours':24,'overlap':True,'shrink':.5}],
   'fusion_selection':'Originalfrozenweight combo and3registered macronormalized simplexfits; chronological targetdatepurgedDEVvalidation asR45; alpha0/.25/.5/1 againstoriginal. RequirefullDEVcurve/daily nonworse, pairedforwardgain>.005ppandoneSE, peakregression≤.1pp; otherwiseoriginalreference. Rawall-finetunedcombo retaineddiagnostic.',
   'freeze_before_TEST':'Freezeall familyconfig/alpha andfusion weights/alpha withSHA beforeD22refitsandfirst newTEST inference. Same recipe restarts originalcheckpoint.',
   'dev24':[14,16,18,20],'dev7':[14,15],'test24':list(range(22,31)),'test7':[22,23,24],
   'metrics':'Unmodified9.28 perPDUWAPE mean57; dailyfull96valid; samecurve/day/peak/MAE/RMSE/AP95/hit/bias/group and80%empiricalinterval definitions.',
   'historical_test_informed_design':True,'independent_confirmation':False,'no_new_Word':True,
   'do_not_claim_full_from_scratch_pretraining':'Fine-tuningofficial outputheads andfinalencoderblocks withverifiedparameterupdates; nottraining allweightsfromrandominit.',
   'license':'Original licenses retained; TimesFM3 weights noncommercialresearchonly.'}
path=R/'docs/protocol.json'
assert not path.exists(),'Do not overwrite registeredprotocol'
path.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'registered':True,'families':6,'neural_members':9,'trained_scopes':2,'checkpoints':2,'all_preflight_passed':True}))
