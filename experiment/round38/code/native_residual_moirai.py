from pathlib import Path
import sys,json,hashlib,datetime,argparse,gc,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round38';Q=R/'results';SRC=B/'round35';sys.path[:0]=[str(SRC/'runtime'),str(SRC/'code'),str(SRC/'code/official_uni2ts/src'),str(B/'code'),str(B/'round29/code')]
import torch,common
from torch import nn
from aligned_metrics import evaluate
from moirai_native_adapter import build
torch.set_num_threads(2);torch.manual_seed(20261007);assert torch.cuda.is_available();DEV=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d=common.load();flat=d['y'].reshape(57,-1);valid=d['valid'].reshape(57,-1);proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def origins(phase,H):return ([14,16,18,20] if H==1 else[14,15]) if phase=='dev' else([31] if phase=='deploy' else(list(range(22,31)) if H==1 else[22,23,24]))
def prior(phase,H):
 with np.load(B/f'round{32 if H==1 else 36}'/'results'/('deployment.npz' if phase=='deploy' else f'{phase}_predictions.npz')) as z:return z[('TTM_head_sparsepeak50_strength0.1_H1' if H==1 else 'a0.1_global_daily_peak0_H7') if phase=='dev' else f'selected_H{H}'].astype('float64')
def profile(t):return flat[:,t-3*96:t].reshape(57,3,96).mean(1)
def hist(t,rate,days=7,mode='level'):
 x=flat[:,t-days*96:t].reshape(57,-1,rate).mean(-1);m=valid[:,t-days*96:t].reshape(57,-1,rate).all(-1)
 if mode=='seasonal':x=x-np.tile(profile(t).reshape(57,-1,rate).mean(-1),(1,days))
 assert x.shape[-1]==days*96//rate;return x,m
class DeltaHead(nn.Module):
 def __init__(self,rank):
  super().__init__();self.a=nn.Linear(384,rank,bias=False);self.b=nn.Linear(rank,4*9*16,bias=False);nn.init.normal_(self.a.weight,std=.01);nn.init.zeros_(self.b.weight)
 def forward(self,x):return .5*torch.tanh(self.b(self.a(x)))
class AdaptedOut(nn.Module):
 def __init__(self,original,delta):super().__init__();self.original=original;self.delta=delta
 def forward(self,x):return self.original(x)+self.delta(x)
def model(ctx,horizon,dim,rate,variant,cut):
 m=build(SRC,ctx,horizon,dim).to(DEV)
 for p in m.parameters():p.requires_grad_(False)
 if variant!='base':
  cfg=next(v for v in proto['adapters'] if v['id']==variant);head=DeltaHead(cfg['rank']).to(DEV);head.load_state_dict(torch.load(R/f'models/{variant}_rate{rate}_cut{cut}.pt',map_location=DEV,weights_only=True));m.module.out_proj=AdaptedOut(m.module.out_proj,head)
 return m.eval()
def features(cut,rate,mode):
 path=Q/f'features_{mode}_rate{rate}_cut{cut}.npz';finger=hashlib.sha256((sha(SRC/'models/moirai2/model.safetensors')+sha(SRC/'code/moirai_native_adapter.py')+sha(R/'code/native_residual_moirai.py')+sha(R/'docs/protocol.json')+json.dumps({'rate':rate,'mode':mode,'cut':cut,'torch':torch.__version__})).encode()+np.ascontiguousarray(flat[:,:cut*96]).tobytes()+np.ascontiguousarray(valid[:,:cut*96]).tobytes()).hexdigest()
 if path.exists():
  with np.load(path) as z:assert str(z['signature'])==finger;return {k:z[k].copy() for k in z.files if k!='signature'}
 bank={k:[] for k in ['x','teacher','loc','scale','truth','mask','pdu','time','norm']};models={};ctx=7*96//rate;ct=(ctx+15)//16
 for groupkind in ['uni','cell']:
  groups=[np.arange(i,min(i+4,57)) for i in range(0,57,4)] if groupkind=='uni' else[np.where(d['cells']==c)[0] for c in np.unique(d['cells'])]
  for t in range(7*96,cut*96-rate+1,48):
   hx,hm=hist(t,rate,mode=mode);norm=(flat[:,t-7*96:t]*valid[:,t-7*96:t]).sum(-1)/valid[:,t-7*96:t].sum(-1).clip(1);remaining=min(64,(cut*96-t)//rate);truth=np.zeros((57,64),np.float32);mask=np.zeros((57,64),bool);truth[:,:remaining]=flat[:,t:t+remaining*rate].reshape(57,remaining,rate).mean(-1);mask[:,:remaining]=valid[:,t:t+remaining*rate].reshape(57,remaining,rate).all(-1)
   if mode=='seasonal':truth[:,:remaining]-=np.tile(profile(t),(1,4))[:,:remaining]
   for ix in groups:
    dim=1 if groupkind=='uni' else len(ix);key=dim
    if key not in models:
     for old in models.values():del old
     models.clear();gc.collect();torch.cuda.empty_cache();models[key]=model(ctx,64,dim,rate,'base',cut)
    m=models[key];x=torch.tensor(hx[ix],dtype=torch.float32,device=DEV)[:,:,None] if dim==1 else torch.tensor(hx[ix].T[None],dtype=torch.float32,device=DEV);obs=torch.tensor(hm[ix],device=DEV)[:,:,None] if dim==1 else torch.tensor(hm[ix].T[None],device=DEV);pad=torch.zeros(x.shape[:2],dtype=torch.bool,device=DEV)
    with torch.inference_mode():
     packed=m._convert(16,x,obs,pad);target,observed,sample,time,var,predmask=packed;loc,scale=m.module.scaler(target,observed*~predmask.unsqueeze(-1),sample,var);idx=torch.arange(ct-1,ct*dim,ct,device=DEV);cap={};hook=m.module.out_proj.register_forward_pre_hook(lambda module,inp:cap.update(x=inp[0][...,idx,:].detach().cpu().numpy().copy()))
     try:raw=m.module(*packed,training_mode=False)
     finally:hook.remove()
     lr=loc[...,idx,0].cpu().numpy().reshape(-1);sr=scale[...,idx,0].cpu().numpy().reshape(-1);tr=((raw[...,idx,:]-loc[...,idx,:])/scale[...,idx,:]).cpu().numpy().reshape(-1,4,9,16).transpose(0,2,1,3).reshape(-1,9,64)
    bank['norm'].append(norm[ix].astype(np.float32));bank['x'].append(cap['x'].reshape(-1,384));bank['teacher'].append(tr);bank['loc'].append(lr);bank['scale'].append(sr);bank['truth'].append(truth[ix]);bank['mask'].append(mask[ix]);bank['pdu'].append(ix);bank['time'].append(np.tile(t//rate+np.arange(64),(len(ix),1)))
   print('features',cut,rate,groupkind,t,flush=True)
 data={k:np.concatenate(v) for k,v in bank.items()};counts=np.zeros((57,cut*96//rate),np.int32)
 for i in range(len(data['pdu'])):
  good=data['mask'][i];np.add.at(counts[data['pdu'][i]],data['time'][i,good],1)
 w=np.zeros_like(data['truth'],np.float32)
 for i in range(len(w)):
  good=data['mask'][i];w[i,good]=1/counts[data['pdu'][i],data['time'][i,good]]
 data['weight']=w;np.savez_compressed(path,**data,signature=finger);save(f'feature_audit_{mode}_rate{rate}_cut{cut}.json',dict(label_end_exclusive=cut*96,max_label_time_exclusive=int(data['time'][data['mask']].max()+1)*rate,input_context_days=7,rows=len(w),features=384,all_label_times_before_cut=True,weighting='inverse PDU targetbin occurrence across overlapping originsanduni/cell inputs',normalization='official scaler uses only eachwindow observedpast, futuretargetzero andunobserved',source_sha256=sha(path)));return data
def train(cut,rate,mode,variants):
 data=features(cut,rate,mode);xs=torch.tensor(data['x'],device=DEV);teacher=torch.tensor(data['teacher'],device=DEV);loc=torch.tensor(data['loc'],device=DEV);scale=torch.tensor(data['scale'],device=DEV);truth=torch.tensor(data['truth'],device=DEV);ww=torch.tensor(data['weight'],device=DEV);norm=torch.tensor(data["norm"],device=DEV).abs().clamp_min(.05);levels=torch.arange(1,10,device=DEV)/10
 for name in variants:
  cfg=next(v for v in proto['adapters'] if v['id']==name);path=R/f'models/{name}_rate{rate}_cut{cut}.pt'
  if path.exists():continue
  torch.manual_seed(20261007);head=DeltaHead(cfg['rank']).to(DEV);opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=.001,betas=(.9,.99));losses=[];g=torch.Generator().manual_seed(20261007)
  for epoch in range(4):
   order=torch.randperm(len(xs),generator=g).to(DEV);total=[]
   for ids in order.split(64):
    delta=head(xs[ids]).reshape(-1,4,9,16).permute(0,2,1,3).reshape(-1,9,64);p=(teacher[ids]+delta)*scale[ids,None,None]+loc[ids,None,None];err=truth[ids,None,:]-p;pin=torch.maximum(levels[None,:,None]*err,(levels[None,:,None]-1)*err)/norm[ids,None,None];anchor=(delta*scale[ids,None,None]/norm[ids,None,None]).square();cross=torch.relu(p[:,:-1]-p[:,1:])/norm[ids,None,None];w=ww[ids];loss=((pin+cfg['teacher_lambda']*anchor)*w[:,None,:]).sum()/(9*w.sum().clamp_min(1))+.1*(cross*w[:,None,:]).sum()/(8*w.sum().clamp_min(1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step();total.append(float(loss.detach()))
   losses.append(float(np.mean(total)));print('train',cut,rate,name,epoch,losses[-1],flush=True)
  torch.save(head.cpu().state_dict(),path);save(f'train_{name}_rate{rate}_cut{cut}.json',dict(cfg,epochs=4,batch_size=64,parameters=sum(p.numel() for p in head.parameters()),weights_sha256=sha(path),training_labels_end_exclusive=cut*96,feature_source_sha256=sha(Q/f'features_{mode}_rate{rate}_cut{cut}.npz'),loss_by_epoch=losses,backbone_frozen=True,delta_bound_half_historical_std=True,all_original_weights_sha256=sha(SRC/'models/moirai2/model.safetensors')))
def infer_one(family,H,o,cut):
 rate=family['rate'];days=family['context_days'];variant=family['variant'];hx,hm=hist(o*96,rate,days,mode=family['mode']);out=np.zeros((57,9,H*96//rate));groups=[np.arange(i,min(i+4,57)) for i in range(0,57,4)] if family['group']=='uni' else[np.where(d['cells']==c)[0] for c in np.unique(d['cells'])];m=None;last=None
 for ix in groups:
  dim=1 if family['group']=='uni' else len(ix)
  if last!=dim:
   if m is not None:del m
   gc.collect();torch.cuda.empty_cache();m=model(days*96//rate,H*96//rate,dim,rate,variant,cut);last=dim
  x=torch.tensor(hx[ix],dtype=torch.float32,device=DEV)[:,:,None] if dim==1 else torch.tensor(hx[ix].T[None],dtype=torch.float32,device=DEV);obs=torch.tensor(hm[ix],device=DEV)[:,:,None] if dim==1 else torch.tensor(hm[ix].T[None],device=DEV)
  with torch.inference_mode():p=m(x,obs,torch.zeros(x.shape[:2],dtype=torch.bool,device=DEV)).cpu().numpy()
  out[ix]=p if dim==1 else p[0].transpose(2,0,1)
 if family['mode']=='seasonal':out+=np.tile(profile(o*96),(1,H))[:,None,:]
 assert np.isfinite(out).all();return np.clip(out,0,1)
def infer(phase):
 requests=[(f,H,o) for f in proto['families'] for H in [1,7] for o in origins('dev',H)] if phase=='dev' else[(read(Q/'selection.json')['horizons'][str(H)]['family'],H,o) for H in [1,7] for o in origins(phase,H)];bank={}
 for f,H,o in requests:
  cut=14 if phase=='dev' else 22;key=f'{f["id"]}_H{H}_o{o}';p=Q/'native_cache'/f'{phase}_{key}.npz';p.parent.mkdir(exist_ok=True);sig=hashlib.sha256((sha(R/'docs/protocol.json')+sha(R/'code/native_residual_moirai.py')+sha(SRC/'code/moirai_native_adapter.py')+sha(SRC/'models/moirai2/model.safetensors')+(sha(R/f'models/{f["variant"]}_rate{f["rate"]}_cut{cut}.pt') if f['variant']!='base' else'base')).encode()+np.ascontiguousarray(flat[:,(o-f['context_days'])*96:o*96]).tobytes()+np.ascontiguousarray(valid[:,(o-f['context_days'])*96:o*96]).tobytes()).hexdigest()
  if p.exists():
   with np.load(p) as z:assert str(z['signature'])==sig;result=z['quantiles'].copy()
  else:
   result=infer_one(f,H,o,cut);tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,quantiles=result,signature=sig);tmp.replace(p)
  bank[key]=result;print(phase,key,flush=True)
 np.savez_compressed(Q/f'native_{phase}.npz',**bank)
def reconstruct(base,coarse,rate,alpha,total):
 b=base.reshape(*base.shape[:-1],-1,rate);target=(1-alpha)*b.mean(-1)+alpha*coarse;lo=np.full(target.shape,-1.);hi=np.full(target.shape,1.)
 for _ in range(48):
  delta=(lo+hi)/2;p=np.clip(b+delta[...,None],0,1);over=p.mean(-1)>target;hi=np.where(over,delta,hi);lo=np.where(over,lo,delta)
 p=np.clip(b+((lo+hi)/2)[...,None],0,1).reshape(base.shape)
 if total=='global_daily':
  pp=p.reshape(57,p.shape[1],-1,96);bb=base.reshape(pp.shape);target=bb.sum((0,3));lo=np.full(target.shape,-1.);hi=np.full(target.shape,1.)
  for _ in range(48):
   delta=(lo+hi)/2;v=np.clip(pp+delta[None,:,:,None],0,1).sum((0,3));over=v>target;hi=np.where(over,delta,hi);lo=np.where(over,lo,delta)
  p=np.clip(pp+((lo+hi)/2)[None,:,:,None],0,1).reshape(base.shape)
 return p
def eligible(s,b,H):
 primary=s['wape']<b['wape']-.001 and(H==1 or s['daily_wape']<=b['daily_wape']+1e-10);tol={'mae':b['mae']*.005,'rmse':b['rmse']*.005,'p95_ape':.02,'hit2_percent':.1,'hit5_percent':.1,'cella5_wape':.01,'other52_wape':.005,'absolute_bias_percent':.01,'peak_time_mae_minutes':5,'daily_mae':b['daily_mae']*.005,'daily_rmse':b['daily_rmse']*.005,'daily_hit5_percent':.5};secondary=all(s[k]>=b[k]-v if 'hit' in k else s[k]<=b[k]+v for k,v in tol.items());peak=s['peak_mape']<=b['peak_mape']+.1;cost=float(np.mean([s['wape']/b['wape']]+([s['daily_wape']/b['daily_wape']] if H==7 else[])));return dict(eligible=bool(primary and secondary and peak),primary_cost=cost,primary_nonworse=primary,secondary_small=secondary,peak_close=peak)
def develop():
 checks=[]
 with np.load(Q/'native_dev.npz') as new,np.load(SRC/'results/native_dev.npz') as old:
  for H in [1,7]:
   for group,family in [('uni','uni7'),('cell','cell14')]:
    for o in origins('dev',H):
     a=new[f'level_{group}_base_H{H}_o{o}'];b=old[f'{family}_H{H}_o{o}'];difference=float(np.max(abs(a-b)));assert difference<1e-6,'Nativecontrol regression';checks.append(dict(H=H,origin=o,group=group,maxdifference=difference))
 save('native_control_regression.json',dict(all_passed=True,checks=checks,scope='same officialweights andhistorical inputs, noTEST used'))
 bank={};scores={};sel={}
 with np.load(Q/'native_dev.npz') as z:
  for H in [1,7]:
   oo=origins('dev',H);base=prior('dev',H);early=[0,1] if H==1 else[0];late=[2,3] if H==1 else[1];be=evaluate(common,d,[oo[i] for i in early],base[:,early],H);families=[];scores[str(H)]=dict(prior=evaluate(common,d,oo,base,H),families={},weights={})
   for f in proto['families']:
    raw=np.stack([z[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);p=reconstruct(base,raw,f['rate'],.25,'none');sc=evaluate(common,d,[oo[i] for i in early],p[:,early],H);cost=sc['wape']/be['wape'];families.append(dict(family=f,early_cost=cost));scores[str(H)]['families'][f['id']]=evaluate(common,d,oo,p,H)
   families.sort(key=lambda x:(x['early_cost'],x['family']['id']));f=families[0]['family'];raw=np.stack([z[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);bl=evaluate(common,d,[oo[i] for i in late],base[:,late],H);rr=[]
   for alpha in [0,.1,.25,.5,1]:
    for total in ['none','global_daily']:
     key=f'a{alpha:g}_{total}';p=reconstruct(base,raw,f['rate'],alpha,total);sc=evaluate(common,d,[oo[i] for i in late],p[:,late],H);allsc=evaluate(common,d,oo,p,H);bank[f'{key}_H{H}']=p;scores[str(H)]['weights'][key]=dict(late=sc,all=allsc);rr.append(dict(alpha=alpha,total=total,key=key,**eligible(sc,bl,H),full_dev_nonworse=allsc['wape']<=scores[str(H)]['prior']['wape']+1e-10))
   rr.sort(key=lambda x:(not(x['eligible'] and x['full_dev_nonworse']),x['primary_cost'],x['alpha'],x['total']));best=next((r for r in rr if r['eligible'] and r['full_dev_nonworse']),None)
   if best:
    # Prefer smaller weights within0.1% normalized cost of minimum eligible.
    near=[r for r in rr if r['eligible'] and r['full_dev_nonworse'] and r['primary_cost']<=best['primary_cost']+.001];best=min(near,key=lambda r:(r['alpha'],r['primary_cost'],r['total']))
   diag=next((r for r in rr if r['alpha']>0),rr[0]);sel[str(H)]=dict(family=f,selected=best,diagnostic=best or diag,retain_prior=best is None,family_ranking=families,weight_ranking=rr)
 np.savez_compressed(Q/'dev_predictions.npz',**bank);save('dev_summary.json',scores);assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(R/'docs/protocol.json'),native_dev_sha256=sha(Q/'native_dev.npz'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),no_test_used=True,horizons=sel));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps({h:{'family':v['family'],'selected':v['selected'],'diagnostic':v['diagnostic']} for h,v in sel.items()},ensure_ascii=False),flush=True)
def score_or_deploy(phase):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();bank={};scores={}
 with np.load(Q/f'native_{phase}.npz') as z:
  for H in [1,7]:
   sh=s['horizons'][str(H)];f=sh['family'];cfg=sh['selected'] or sh['diagnostic'];oo=origins(phase,H);base=prior(phase,H);raw=np.stack([z[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);p=reconstruct(base,raw,f['rate'],cfg['alpha'],cfg['total']);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=p if sh['selected'] else base
   if phase=='test':scores[str(H)]={k:evaluate(common,d,oo,v,H) for k,v in [('prior',base),('diagnostic',p),('selected',bank[f'selected_H{H}'])]}
 np.savez_compressed(Q/('test_predictions.npz' if phase=='test' else'deployment.npz'),**bank)
 if phase=='test':save('summary.json',dict(selection=s,scores=scores,independent_confirmation=False,no_test_grid=True));print(json.dumps(scores,ensure_ascii=False),flush=True)
stage=args.stage
if stage=='smoke':
 m=build(SRC,56,8,1).to(DEV);h=DeltaHead(4).to(DEV);x=torch.randn(2,384,device=DEV)
 with torch.inference_mode():a=m.module.out_proj(x);b=AdaptedOut(m.module.out_proj,h)(x)
 assert torch.equal(a,b);save('adapter_verification.json',dict(zero_adapter_equals_original=True,max_difference=float((a-b).abs().max()),backbone_weights_sha256=sha(SRC/'models/moirai2/model.safetensors'),new_trainable_rank4=sum(v.numel() for v in h.parameters()),cuda=torch.cuda.get_device_name(),torch_version=torch.__version__,numpy_version=np.__version__));print('zero residual head exact equality passed',flush=True)
elif stage=='train_dev':
 for mode in ['level','seasonal']:train(14,1,mode,[v['id'] for v in proto['adapters'] if v['mode']==mode])
elif stage=='train_final':
 s=read(Q/'selection.json');needed={}
 for v in s['horizons'].values():
  f=v['family']
  if f['variant']!='base':needed.setdefault((f['rate'],f['mode']),set()).add(f['variant'])
 for (rate,mode),names in needed.items():train(22,rate,mode,sorted(names))
elif stage.startswith('native_'):infer(stage.split('_')[1])
elif stage=='develop':develop()
else:score_or_deploy('test' if stage=='test' else'deploy')
