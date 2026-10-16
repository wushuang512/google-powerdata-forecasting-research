from pathlib import Path
import argparse,sys,os,json,hashlib,datetime,gc,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=Path(__file__).resolve().parents[1];Q=R/'results';SRC=B/'round35'
sys.path[:0]=[str(SRC/'runtime'),str(SRC/'code'),str(SRC/'code/official_uni2ts/src'),str(B/'code'),str(B/'round29/code')]
import torch,common
from aligned_metrics import evaluate
from moirai_native_adapter import build
torch.set_num_threads(2);torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();DEV=torch.device('cuda');d=common.load();flat=d['y'].reshape(57,-1);valid=d['valid'].reshape(57,-1);proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def origins(phase,H):return ([14,16,18,20] if H==1 else[14,15]) if phase=='dev' else([31] if phase=='deploy' else(list(range(22,31)) if H==1 else[22,23,24]))
def prior(phase,H):
 with np.load(B/f'round{32 if H==1 else 36}'/'results'/('deployment.npz' if phase=='deploy' else f'{phase}_predictions.npz')) as z:return z[('TTM_head_sparsepeak50_strength0.1_H1' if H==1 else'a0.1_global_daily_peak0_H7') if phase=='dev' else f'selected_H{H}'].astype('float64')
def factor_basis(cut):
 path=R/f'models/factor_basis_cut{cut}.npz';finger=hashlib.sha256(sha(R/'docs/protocol.json').encode()+np.ascontiguousarray(flat[:,:cut*96]).tobytes()+np.ascontiguousarray(valid[:,:cut*96]).tobytes()).hexdigest()
 if path.exists():
  with np.load(path) as z:assert str(z['signature'])==finger;return {k:z[k].copy() for k in ['mean','scale','vectors','variance']}
 y=flat[:,:cut*96];m=valid[:,:cut*96];count=m.sum(-1);mu=(y*m).sum(-1)/count;std=np.sqrt(((y-mu[:,None])**2*m).sum(-1)/count).clip(.01);x=np.where(m,(y-mu[:,None])/std[:,None],0);u,s,v=np.linalg.svd(x,full_matrices=False)
 sign=np.where(u[np.abs(u).argmax(0),np.arange(len(s))]>=0,1,-1);vec=(u*sign).T[:6];var=s**2/(s**2).sum();z=dict(mean=mu,scale=std,vectors=vec,variance=var[:6]);np.savez_compressed(path,**z,signature=finger);save(f'factor_basis_cut{cut}.json',dict(cut_end_exclusive=cut*96,source_sha256=sha(path),input_prefix_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),valid_prefix_sha256=hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest(),orthogonality_max_error=float(abs(vec@vec.T-np.eye(6)).max()),variance_explained=var[:6].tolist(),mean_std_from_valid_training_only=True,minimum_scale=.01,sign_canonical=True));return z
def history(t,days,rate,k,basis):
 y=flat[:,t-days*96:t];m=valid[:,t-days*96:t];x=np.where(m,(y-basis['mean'][:,None])/basis['scale'][:,None],0);f=basis['vectors'][:k]@x;obs=np.broadcast_to((m.mean(0)>=.95)[None],f.shape).copy();return f.reshape(k,-1,rate).mean(-1),obs.reshape(k,-1,rate).all(-1)
def set_scope(m,f):
 scope=f['scope'];last=6-f.get('last_layers',0)
 for name,p in m.module.named_parameters():p.requires_grad_(scope=='full' or scope=='last_head' and (name.startswith('out_proj.') or any(name.startswith(f'encoder.layers.{i}.') for i in range(last,6))))
 n=sum(p.numel() for p in m.parameters() if p.requires_grad);assert n>0 and any(p.requires_grad for name,p in m.module.named_parameters() if name.startswith('encoder.layers.'));return n
def model(f,context,horizon,cut,trained=True):
 m=build(SRC,context,horizon,f['rank']).to(DEV).eval()
 for p in m.parameters():p.requires_grad_(False)
 if f['variant']!='base' and trained:
  path=R/f'models/{f["id"]}_cut{cut}.pt';m.module.load_state_dict(torch.load(path,map_location=DEV,weights_only=True),strict=True)
 return m
def tensors(h,m):
 x=torch.tensor(h.T[None],dtype=torch.float32,device=DEV);obs=torch.tensor(m.T[None],device=DEV);pad=torch.zeros(x.shape[:2],dtype=torch.bool,device=DEV);return x,obs,pad
def direct_quantiles(m,x,obs,pad):
 packed=m._convert(16,x,obs,pad);raw=m.module(*packed,training_mode=False);ct=(m.hparams.context_length+15)//16;k=m.hparams.target_dim;idx=torch.arange(ct-1,ct*k,ct,device=DEV);p=raw[:,idx].reshape(1,k,4,9,16).permute(0,1,3,2,4).reshape(k,9,64);loc,scale=m.module.scaler(packed[0],packed[1]*~packed[-1].unsqueeze(-1),packed[2],packed[4]);return p,scale[:,idx,0].reshape(k).clamp_min(1e-5)
def train(cut,requested=None):
 basis=factor_basis(cut);families=proto['families'] if requested is None else requested
 for f in families:
  if f['variant']=='base':continue
  path=R/f'models/{f["id"]}_cut{cut}.pt'
  if path.exists():
   meta=read(Q/f'train_{f["id"]}_cut{cut}.json');assert meta['weights_sha256']==sha(path) and meta['protocol_sha256']==sha(R/'docs/protocol.json');continue
  k=f['rank'];rate=f['rate'];ctx=7*96//rate;windows=[]
  for t in range(7*96,cut*96-rate+1,48):
   hx,hm=history(t,7,rate,k,basis);n=min(64,(cut*96-t)//rate);y=np.zeros((57,64));mask=np.zeros((57,64),bool);y[:,:n]=flat[:,t:t+n*rate].reshape(57,n,rate).mean(-1);mask[:,:n]=valid[:,t:t+n*rate].reshape(57,n,rate).all(-1);truth=basis['vectors'][:k]@((y-basis['mean'][:,None])/basis['scale'][:,None]);windows.append(dict(t=t,x=hx,obs=hm,y=truth,mask=mask,time=t//rate+np.arange(64)))
  counts=np.zeros((57,cut*96//rate),int)
  for w in windows:
   for i in range(57):np.add.at(counts[i],w['time'][w['mask'][i]],1)
  for w in windows:
   ww=np.zeros_like(w['mask'],float)
   for i in range(57):ww[i,w['mask'][i]]=1/counts[i,w['time'][w['mask'][i]]]
   w['weight']=torch.tensor(ww,dtype=torch.float32,device=DEV)
  torch.manual_seed(20261007);m=model(f,ctx,64,cut,False);teacher=model(f,ctx,64,cut,False);nparams=set_scope(m,f);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=f['lr'],weight_decay=.01,betas=(.9,.99));vv=torch.tensor(basis['vectors'][:k].T*basis['scale'][:,None],dtype=torch.float32,device=DEV);norm=torch.tensor(basis['mean'].clip(.05),dtype=torch.float32,device=DEV);levels=torch.arange(1,10,device=DEV)/10;g=np.random.default_rng(20261007);losses=[];teacher_cache={};max_t=0
  for step in range(proto['train_steps']):
   ids=g.choice(len(windows),size=2,replace=False);ll=[]
   for ii in ids:
    w=windows[ii];x,obs,pad=tensors(w['x'],w['obs']);pred,scale=direct_quantiles(m,x,obs,pad)
    if ii not in teacher_cache:
     with torch.inference_mode():teacher_cache[ii]=direct_quantiles(teacher,x,obs,pad)[0].detach().clone()
    truth=torch.tensor(w['y'],dtype=torch.float32,device=DEV);ok=w['mask'].mean(0)>=.95;mask=torch.tensor(ok,dtype=torch.float32,device=DEV);err=(truth[:,None]-pred)/scale[:,None,None];pin=torch.maximum(levels[None,:,None]*err,(levels[None,:,None]-1)*err);diff=vv@(pred[:,4]-truth);weight=w['weight'];power=((diff.abs()/norm[:,None])*weight).sum()/weight.sum().clamp_min(1);anchor=(((pred-teacher_cache[ii])/scale[:,None,None])**2*mask[None,None]).sum()/(mask.sum().clamp_min(1)*k*9);qloss=(pin*mask[None,None]).sum()/(mask.sum().clamp_min(1)*k*9);cross=(torch.relu(pred[:,:-1]-pred[:,1:])/scale[:,None,None]*mask[None,None]).sum()/(mask.sum().clamp_min(1)*k*8);ll.append(power+.25*qloss+f['anchor']*anchor+.1*cross);max_t=max(max_t,int(w['time'][w['mask'].any(0)].max()+1)*rate)
   loss=torch.stack(ll).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1);opt.step();losses.append(float(loss.detach()))
   if step%10==0:print('train',cut,f['id'],step,losses[-1],nparams,flush=True)
  assert max_t<=cut*96;torch.save({name:p.detach().cpu() for name,p in m.module.state_dict().items()},path);save(f'train_{f["id"]}_cut{cut}.json',dict(config=f,parameters=nparams,steps=proto['train_steps'],actual_label_end_exclusive=max_t,training_label_cut=cut*96,history_days=7,inference_days=f['context_days'],loss_by_step=losses,weights_sha256=sha(path),protocol_sha256=sha(R/'docs/protocol.json'),basis_sha256=sha(R/f'models/factor_basis_cut{cut}.npz'),backbone_attention_actually_trainable=True,teacher_original_frozen=True,training_inputs_only_completed_history=True,loss='PDU mean-normalized projected power L1 +.25 latent quantile pinball + originalteacher anchor +.1 crossings',labels='Projected training-only true power; never passed into modelinput',windows=len(windows),original_weights_sha256=sha(SRC/'models/moirai2/model.safetensors')));del m,teacher,opt,teacher_cache,windows;gc.collect();torch.cuda.empty_cache()
def infer_one(f,H,o,cut):
 basis=factor_basis(cut);k=f['rank'];rate=f['rate'];hx,hm=history(o*96,f['context_days'],rate,k,basis);m=model(f,hx.shape[-1],H*96//rate,cut);x,obs,pad=tensors(hx,hm)
 with torch.inference_mode():p=m(x,obs,pad).cpu().numpy()
 out=p[0][None] if k==1 else p[0].transpose(2,0,1);assert out.shape==(k,9,H*96//rate) and np.isfinite(out).all();return out
def infer(phase):
 selection=read(Q/'selection.json') if phase!='dev' else None;families=proto['families'] if phase=='dev' else[selection['family']];bank={};cut=14 if phase=='dev' else 22
 for f in families:
  for H in [1,7]:
   for o in origins(phase,H):
    key=f'{f["id"]}_H{H}_o{o}';path=Q/'native_cache'/f'{phase}_{key}.npz';basis=factor_basis(cut);weight=sha(R/f'models/{f["id"]}_cut{cut}.pt') if f['variant']!='base' else'base';finger=hashlib.sha256((sha(R/'docs/protocol.json')+sha(Path(__file__))+sha(SRC/'code/moirai_native_adapter.py')+sha(SRC/'models/moirai2/model.safetensors')+sha(R/f'models/factor_basis_cut{cut}.npz')+weight).encode()+np.ascontiguousarray(flat[:,(o-f['context_days'])*96:o*96]).tobytes()+np.ascontiguousarray(valid[:,(o-f['context_days'])*96:o*96]).tobytes()).hexdigest()
    if path.exists():
     with np.load(path) as z:assert str(z['signature'])==finger;p=z['quantiles'].copy()
    else:p=infer_one(f,H,o,cut);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,quantiles=p,signature=finger);tmp.replace(path)
    bank[key]=p;print(phase,key,flush=True)
 np.savez_compressed(Q/f'native_{phase}.npz',**bank)
def reconstruct(base,factors,rate,alpha,total,basis,k):
 b=base.reshape(*base.shape[:-1],-1,rate);coarse=b.mean(-1);z=(coarse-basis['mean'][:,None,None])/basis['scale'][:,None,None];v=basis['vectors'][:k];fb=np.einsum('kd,dot->kot',v,z);delta=np.einsum('dk,kot->dot',v.T,factors-fb)*basis['scale'][:,None,None];target=np.clip(coarse+alpha*delta,0,1);lo=np.full(target.shape,-1.);hi=np.full(target.shape,1.)
 for _ in range(48):
  shift=(lo+hi)/2;p=np.clip(b+shift[...,None],0,1);over=p.mean(-1)>target;hi=np.where(over,shift,hi);lo=np.where(over,lo,shift)
 p=np.clip(b+((lo+hi)/2)[...,None],0,1).reshape(base.shape)
 if total=='global_daily':
  pp=p.reshape(57,p.shape[1],-1,96);bb=base.reshape(pp.shape);target=bb.sum((0,3));lo=np.full(target.shape,-1.);hi=np.full(target.shape,1.)
  for _ in range(48):
   shift=(lo+hi)/2;over=np.clip(pp+shift[None,:,:,None],0,1).sum((0,3))>target;hi=np.where(over,shift,hi);lo=np.where(over,lo,shift)
  p=np.clip(pp+((lo+hi)/2)[None,:,:,None],0,1).reshape(base.shape)
 return p
def eligible(s,b,H):
 primary=s['wape']<b['wape']-.001 and(H==1 or s['daily_wape']<=b['daily_wape']+1e-10);tol={'mae':b['mae']*.005,'rmse':b['rmse']*.005,'p95_ape':.02,'hit2_percent':.1,'hit5_percent':.1,'cella5_wape':.01,'other52_wape':.005,'absolute_bias_percent':.01,'peak_time_mae_minutes':5,'daily_mae':b['daily_mae']*.005,'daily_rmse':b['daily_rmse']*.005,'daily_hit5_percent':.5};secondary=all(s[k]>=b[k]-v if'hit'in k else s[k]<=b[k]+v for k,v in tol.items());peak=s['peak_mape']<=b['peak_mape']+.1;return dict(eligible=bool(primary and secondary and peak),primary=bool(primary),secondary_small=bool(secondary),peak_close=bool(peak),curve_ratio=s['wape']/b['wape'])
def develop():
 basis=factor_basis(14);bs={H:prior('dev',H) for H in [1,7]};early={1:[0,1],7:[0]};late={1:[2,3],7:[1]};be={H:evaluate(common,d,[origins('dev',H)[i] for i in early[H]],bs[H][:,early[H]],H) for H in [1,7]};ba={H:evaluate(common,d,origins('dev',H),bs[H],H) for H in [1,7]};bl={H:evaluate(common,d,[origins('dev',H)[i] for i in late[H]],bs[H][:,late[H]],H) for H in [1,7]};scores=dict(prior={str(h):s for h,s in ba.items()},families={},weights={});ranking=[];bank={}
 with np.load(Q/'native_dev.npz') as z:
  for f in proto['families']:
   hh={};ratios=[]
   for H in [1,7]:
    oo=origins('dev',H);native=np.stack([z[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);p=reconstruct(bs[H],native,f['rate'],.25,'none',basis,f['rank']);se=evaluate(common,d,[oo[i] for i in early[H]],p[:,early[H]],H);hh[str(H)]=evaluate(common,d,oo,p,H);ratios.append(se['wape']/be[H]['wape'])
   ranking.append(dict(family=f,early_worst_curve_ratio=max(ratios),early_mean_curve_ratio=float(np.mean(ratios))));scores['families'][f['id']]=hh
  ranking.sort(key=lambda r:(r['early_worst_curve_ratio'],r['early_mean_curve_ratio'],r['family']['id']));f=ranking[0]['family'];rr=[]
  for alpha in [0,.05,.1,.25,.5,1]:
   for total in ['none','global_daily']:
    key=f'a{alpha:g}_{total}';gates={};full={};isgood=True;ratios=[]
    for H in [1,7]:
     oo=origins('dev',H);native=np.stack([z[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);p=reconstruct(bs[H],native,f['rate'],alpha,total,basis,f['rank']);bank[f'{key}_H{H}']=p;s=evaluate(common,d,[oo[i] for i in late[H]],p[:,late[H]],H);alls=evaluate(common,d,oo,p,H);gates[str(H)]=eligible(s,bl[H],H);full[str(H)]=dict(score=alls,curve_nonworse=alls['wape']<=ba[H]['wape']+1e-10);isgood &= gates[str(H)]['eligible'] and full[str(H)]['curve_nonworse'];ratios.append(gates[str(H)]['curve_ratio'])
    scores['weights'][key]=dict(gates=gates,full=full);rr.append(dict(alpha=alpha,total=total,key=key,eligible=bool(isgood),late_worst_curve_ratio=max(ratios),late_mean_curve_ratio=float(np.mean(ratios))))
 rr.sort(key=lambda r:(not r['eligible'],r['late_worst_curve_ratio'],r['late_mean_curve_ratio'],r['alpha'],r['total']));best=next((r for r in rr if r['eligible']),None)
 if best:best=min([r for r in rr if r['eligible'] and r['late_worst_curve_ratio']<=best['late_worst_curve_ratio']+.001],key=lambda r:(r['alpha'],r['late_worst_curve_ratio'],r['total']))
 diag=best or next(r for r in rr if r['alpha']>0);np.savez_compressed(Q/'dev_predictions.npz',**bank);save('dev_summary.json',scores);assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),family=f,selected=best,diagnostic=diag,one_shared_configuration_both_horizons=True,no_test_used=True,protocol_sha256=sha(R/'docs/protocol.json'),native_dev_sha256=sha(Q/'native_dev.npz'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),family_ranking=ranking,weight_ranking=rr));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps(dict(family=f,selected=best,diagnostic=diag),ensure_ascii=False),flush=True)
def score_deploy(phase):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();f=s['family'];cfg=s['selected'] or s['diagnostic'];basis=factor_basis(22);bank={};scores={}
 with np.load(Q/f'native_{phase}.npz') as z:
  for H in [1,7]:
   oo=origins(phase,H);base=prior(phase,H);native=np.stack([z[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);p=reconstruct(base,native,f['rate'],cfg['alpha'],cfg['total'],basis,f['rank']);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=p if s['selected'] else base
   if phase=='test':scores[str(H)]={k:evaluate(common,d,oo,v,H) for k,v in [('prior',base),('diagnostic',p),('selected',bank[f'selected_H{H}'])]}
 np.savez_compressed(Q/('test_predictions.npz' if phase=='test' else'deployment.npz'),**bank)
 if phase=='test':
  gates={str(H):eligible(scores[str(H)]['diagnostic'],scores[str(H)]['prior'],H) for H in [1,7]};release=bool(s['selected'] and all(v['eligible'] for v in gates.values()));save('summary.json',dict(selection=s,scores=scores,test_gates=gates,adopt_shared_scheme=release,independent_confirmation=False,no_test_grid=True,all_curves_leading_descriptive=all(scores[str(H)]['diagnostic']['wape']<proto['previous_test_min_wape'][str(H)] for H in [1,7]),substantial_descriptive=all(scores[str(H)]['diagnostic']['wape']<=.95*proto['previous_test_min_wape'][str(H)] for H in [1,7])));print(json.dumps(dict(scores=scores,gates=gates,adopt=release),ensure_ascii=False),flush=True)
stage=args.stage
if stage=='smoke':
 f=proto['families'][0];b=factor_basis(14);assert np.allclose(b['vectors']@b['vectors'].T,np.eye(6),atol=1e-10);m=model(f,64,64,14,False);hx,hm=history(14*96,7,12,f['rank'],b);x,obs,pad=tensors(hx[:,-64:],hm[:,-64:]);p,_=direct_quantiles(m,x,obs,pad);assert torch.isfinite(p).all();save('smoke.json',dict(passed=True,shape=list(p.shape),cuda=torch.cuda.get_device_name(),factor_basis_orthogonal=True,official_weights_sha256=sha(SRC/'models/moirai2/model.safetensors')));print('PCA and differentiable next64 officialmodule passed',flush=True)
elif stage=='train_dev':train(14)
elif stage=='train_final':
 s=read(Q/'selection.json');train(22,[s['family']]) if s['family']['variant']!='base' else factor_basis(22)
elif stage.startswith('native_'):infer(stage.split('_')[1])
elif stage=='develop':develop()
else:score_deploy('test' if stage=='test' else'deploy')
