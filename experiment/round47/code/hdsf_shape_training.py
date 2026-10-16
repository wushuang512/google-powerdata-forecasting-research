"""Round47 load-oriented real gradient training; registered before DEV."""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, os, sys, traceback
sys.dont_write_bytecode=True
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def initialize(root):
    global T,R,Q,P,stage,Adapter
    T=Path(root);R=T/'experiment/round47';Q=R/'results';P=read(R/'docs/protocol.json')
    assert P['training_script_sha256']==sha(__file__)
    sys.path.insert(0,str(T/'experiment/round46/code'))
    import hdsf_foundation_stage as st
    from hdsf_foundation_adapters import Adapter as Ad
    stage,Adapter=st,Ad;stage.initialize(T)
    assert sha(T/'experiment/round46/results/selection.json')==P['round46_selection_sha256']

def batches(cut,family):
    rng=np.random.default_rng(20261012);seq=[]
    for i in range(P['steps']):
        h=96 if family=='chronos2' else [96,672][i%2]
        o=int(rng.choice(np.arange(3*96,cut*96-h+1,48)))
        ctx=min(int(rng.choice([3,7,14]))*96,o)//32*32
        seq.append(dict(origin_tick=o,context_ticks=ctx,horizon_ticks=h,
                        series_ids=rng.choice(57,2,replace=False).tolist(),mode='uni',last_label_tick=o+h-1))
    return seq

def tasks(family):return [('chronos2_long',1)] if family=='chronos2' else [('patchtstfm_long',1),('patchtstfm_long',7)]
def collect(adapter,family,phase,label):
    path=Q/f'{family}_{phase}_{label}.npz';meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        old=read(meta);assert old['npz_sha256']==sha(path) and old['protocol_sha256']==sha(R/'docs/protocol.json')
        return {k:v.copy() for k,v in np.load(path).items()}
    out={};digests={}
    for member,h in tasks(family):
        points=[]
        for o in stage.origins(phase,h):
            print('INFER',family,phase,label,member,h,o,flush=True)
            points.append(stage.predict_member(adapter,member,o,h)[stage.canon])
            digests[f'{member}_H{h}_o{o}']=hashlib.sha256(stage.context(member,o).tobytes()).hexdigest()
        out[f'{member}_H{h}']=np.stack(points,axis=1)
    stage.np_save(path,out)
    save(meta,dict(utc=utc(),npz_sha256=sha(path),protocol_sha256=sha(R/'docs/protocol.json'),
                   input_context_digests=digests,trainable_sha256=adapter.hash_trainable(),family=family,phase=phase,label=label))
    return out

def load_loss(adapter,q,p,target,mask,teacher,recipe):
    y=torch.as_tensor(np.ascontiguousarray(target,dtype=np.float32),device=p.device)
    m=torch.as_tensor(np.ascontiguousarray(mask),device=p.device)
    scale=(y.abs()*m).sum(-1).clamp_min(.05)
    point=((p-y).abs()*m).sum(-1)/scale
    pd,yd,md=p.reshape(len(p),-1,96),y.reshape(len(y),-1,96),m.reshape(len(y),-1,96)
    ok=md.all(-1)
    daily=(((pd.mean(-1)-yd.mean(-1)).abs()*ok).sum(-1)/(yd.mean(-1).abs()*ok).sum(-1).clamp_min(.05)).mean()
    peak=(((pd.max(-1).values-yd.max(-1).values).abs()/yd.max(-1).values.clamp_min(.05))*ok).sum()/ok.sum().clamp_min(1)
    shapeerr=((pd-pd.mean(-1,keepdim=True))-(yd-yd.mean(-1,keepdim=True))).abs()*ok[:,:,None]
    shape=(shapeerr.sum((1,2))/scale).mean()
    slope=((torch.diff(p,dim=1)-torch.diff(y,dim=1)).abs()*(m[:,1:]&m[:,:-1])).sum(-1)/scale
    levels=torch.as_tensor(adapter.levels,device=p.device,dtype=p.dtype);err=y[:,:,None]-q
    pin=(torch.maximum(levels*err,(levels-1)*err)*m[:,:,None]).sum((1,2))/scale/len(levels)
    teacher_anchor=(p-teacher).square().mean()/teacher.abs().mean().clamp_min(.05).square()
    bounds=(torch.relu(-p)+torch.relu(p-1)).mean()
    c=P['loss_recipes'][recipe]
    loss=point.mean()+c['daily']*daily+c['peak']*peak+c['shape']*shape+c['slope']*slope.mean()+.05*pin.mean()+.15*teacher_anchor+.1*bounds
    return loss,dict(point=float(point.mean().detach()),daily=float(daily.detach()),peak=float(peak.detach()),
                      shape=float(shape.detach()),pinball=float(pin.mean().detach()),teacher_anchor=float(teacher_anchor.detach()),loss=float(loss.detach()))

def run(family,recipe,phase):
    status=Q/f'{family}_{recipe}_{phase}_status.json';done=Q/f'{family}_{recipe}_{phase}_complete.json'
    if done.exists():assert read(done)['all_passed'];return
    if phase=='test':
        s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
        assert [family,recipe] in s['required_final_training']
    cut=14 if phase=='dev' else 22;steps=P['steps'];scope='tail'
    torch.manual_seed(20261012)
    save(status,dict(state='running',pid=os.getpid(),started_utc=utc(),family=family,recipe=recipe,phase=phase))
    adapter=Adapter(T,family,'cuda');control=collect(adapter,family,phase,'original_control')
    # Native original control must agree with previous experiment on common tasks.
    priorcontrol=T/'experiment/round46/results'/f'{family}_{phase}_original_control.npz'
    with np.load(priorcontrol) as prior:
        for k in set(prior.files)&set(control):
            assert np.allclose(prior[k],control[k],atol=2e-6,rtol=1e-6),(family,k,'control changed')
    adapter.set_scope('head');head=set(adapter.train_names);adapter.set_scope(scope)
    encoder=set(adapter.train_names)-head;assert encoder
    initial=adapter.trainable_state();initial_hash=adapter.hash_trainable()
    refs={n:x.to(adapter.device) for n,x in initial.items()}
    params=[p for p in adapter.model.parameters() if p.requires_grad]
    lr=P['lr'];opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
    seq=batches(cut,family);flat=stage.data.target[:,:cut].reshape(57,-1);valid=stage.data.target_valid[:,:cut].reshape(57,-1)
    teacher=[]
    for i,b in enumerate(seq,1):
        ix,o,ctx,h=b['series_ids'],b['origin_tick'],b['context_ticks'],b['horizon_ticks'];assert o+h<=cut*96
        with torch.no_grad():_,p=adapter.tensor_forecast(flat[ix,o-ctx:o],h,'uni',False)
        teacher.append(p.detach().float().cpu())
        if i%40==0:print('TEACHER',family,recipe,phase,i,flush=True)
    history=[];encoder_grad=False
    for i,b in enumerate(seq,1):
        ix,o,ctx,h=b['series_ids'],b['origin_tick'],b['context_ticks'],b['horizon_ticks']
        q,p=adapter.tensor_forecast(flat[ix,o-ctx:o],h,'uni',True)
        assert q.requires_grad and p.requires_grad
        loss,info=load_loss(adapter,q.float(),p.float(),flat[ix,o:o+h],valid[ix,o:o+h],teacher[i-1].to(adapter.device),recipe)
        prox=torch.stack([(dict(adapter.model.named_parameters())[n]-ref).square().mean()/ref.square().mean().clamp_min(1e-4) for n,ref in refs.items()]).mean()
        loss=loss+.05*prox;assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        gradients={n:float(v.grad.abs().max()) for n,v in adapter.model.named_parameters() if v.requires_grad and v.grad is not None}
        assert gradients and max(gradients.values())>0 and all(np.isfinite(v) for v in gradients.values())
        encoder_grad |= any(gradients.get(n,0)>0 for n in encoder)
        torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        for group in opt.param_groups:group['lr']=lr*(.2+.8*(1-i/steps))
        history.append(dict(step=i,UTC=utc(),**info,max_gradient=max(gradients.values()),prox=float(prox.detach())))
        if i%20==0:print('TRAIN',family,recipe,phase,i,history[-1],flush=True)
    state=adapter.trainable_state();changedhead=[n for n in head if not torch.equal(state[n],initial[n])]
    changedencoder=[n for n in encoder if not torch.equal(state[n],initial[n])]
    assert encoder_grad and changedhead and changedencoder and adapter.hash_trainable()!=initial_hash
    model=R/'models'/f'{family}_{recipe}_cut{cut}.pt';torch.save(state,model)
    save(model.with_suffix('.json'),dict(family=family,recipe=recipe,scope=scope,cutoff=cut,steps=steps,
        initial_trainable_sha256=initial_hash,updated_trainable_sha256=adapter.hash_trainable(),
        model_sha256=sha(model),CUDA=True,changed_head_tensors=changedhead,changed_encoder_tensors=changedencoder,
        encoder_gradient_observed=encoder_grad,last_training_label_tick=max(x['last_label_tick'] for x in seq),
        batch_sequence_sha256=hashlib.sha256(json.dumps(seq,sort_keys=True).encode()).hexdigest(),protocol_sha256=sha(R/'docs/protocol.json'),utc=utc()))
    save(Q/f'{family}_{recipe}_{phase}_training.json',history)
    collect(adapter,family,phase,recipe)
    save(done,dict(all_passed=True,completed_utc=utc(),real_optimizer_updates=steps,model_sha256=sha(model)))
    save(status,dict(state='complete',family=family,recipe=recipe,phase=phase,completed_utc=utc()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('family',choices=['chronos2','patchtst']);ap.add_argument('recipe',choices=['curve','structure']);ap.add_argument('--phase',choices=['dev','test'],required=True);args=ap.parse_args()
    initialize(args.root)
    try:run(args.family,args.recipe,args.phase)
    except BaseException as e:
        save(Q/f'{args.family}_{args.recipe}_{args.phase}_status.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc(),utc=utc()));raise
