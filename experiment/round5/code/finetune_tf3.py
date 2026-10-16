import argparse,copy,gc,time
from lib5 import *
from timesfm3 import TimesFM3Forecaster
from timesfm3.torch.timesfm3_forecaster import linear_interpolation
from round2_lib import inputs
from timesfm3_run import forecast,CONFIGS as BASECFG
from timesfm3.torch import cpm_revin_refine as cpm
# The inference implementation differentiates sqrt(variance) at zero when
# gradients are enabled. RevIN statistics are treated as fixed statistics in
# this custom adaptation; the forward values remain exactly unchanged.
cpm.cpm_iterative_revin_refine=torch.no_grad()(cpm.cpm_iterative_revin_refine)
CONFIGS={'tf3_head_lr3e6':dict(scope='head',lr=3e-6),'tf3_head_lr1e5':dict(scope='head',lr=1e-5),'tf3_tail_lr3e7':dict(scope='tail',lr=3e-7)}
STEPS=[50,150,300]
def draw(d,cutoff,rng,context=672):
    cells=np.unique(d['cells']);sizes=np.array([(d['cells']==c).sum() for c in cells]);cell=rng.choice(cells,p=sizes/57);ix=np.where(d['cells']==cell)[0];end=int(rng.integers(672,cutoff*96-96+1));start=end-context
    yt=d['y'].reshape(57,-1);pt=np.where(d['pvalid'],d['prod'],np.nan).reshape(57,-1)
    target=tensor(yt[ix,start:end].astype('float32')[None]);cov=pt[ix,start:end];cov=cov[np.isfinite(cov).mean(1)>=.2];cov=tensor(linear_interpolation(cov.astype('float32'))[None]) if len(cov) else None
    labels=tensor(yt[ix,end:end+96].astype('float32'));valid=tensor(d['valid'].reshape(57,-1)[ix,end:end+96]);return target,cov,labels,valid
def save_delta(f,path,cfg,cutoff,step):
    torch.save(dict(state={k:p.detach().cpu().clone() for k,p in f.model.named_parameters() if p.requires_grad},configuration=cfg,cutoff=cutoff,step=step,base_revision='43046b85ec22d584a13f8098c2ed39c889e129c2',license='TimesFM Non-Commercial License v1.0'),path)
def load_delta(path):
    f=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device=DEVICE,per_core_batch_size=4,local_files_only=True);x=torch.load(path,map_location='cpu',weights_only=True);result=f.model.load_state_dict(x['state'],strict=False);assert not result.unexpected_keys;return f
def train(d,cutoff,cfg,steps=300,callback=None):
    torch.manual_seed(20261001);rng=np.random.default_rng(20261001)
    f=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device=DEVICE,per_core_batch_size=4,local_files_only=True);model=f.model;teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.output_head.parameters():p.requires_grad_(True)
    if cfg['scope']=='tail':
        for p in model.transformer_stack.layers[-1].parameters():p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01);quant=tensor(np.array(model.quantiles,dtype='float32'));hist=[]
    print('PARAMETERS',sum(p.numel() for p in params),flush=True);model.eval();other={}
    if cfg.get('objective')=='ensemble':
        for split in (['train'] if cutoff==14 else ['train','dev']):
            c=np.load(B/f'round3/results/chronos_{split}.npz')['native'];t=np.load(B/f'round3/results/tirex_{split}.npz')['native']
            for j,o in enumerate(TRAIN if split=='train' else DEV):other[o]=.25*c[:,j]+.25*t[:,j]
    for step in range(1,steps+1):
        if cfg.get('objective')=='ensemble':
            o=int(rng.choice(list(other)));cells=np.unique(d['cells']);sizes=np.array([(d['cells']==c).sum() for c in cells]);ci=int(rng.choice(len(cells),p=sizes/57));ix=np.where(d['cells']==cells[ci])[0]
            inp=inputs(d,o,'cellprod',context=1344)[ci];target=tensor(linear_interpolation(inp['target'])[None]);pc=inp.get('past_covariates',{});cov=tensor(linear_interpolation(np.stack(list(pc.values())))[None]) if pc else None;y=tensor(d['y'][ix,o].astype('float32'));m=tensor(d['valid'][ix,o]);shift=tensor(other[o][ix].astype('float32'))
        else:target,cov,y,m=draw(d,cutoff,rng);shift=None
        with torch.no_grad():tq=teacher.decode(target,horizon=96,past_only_covariates=cov)[:,:target.shape[1]][0]
        q=type(model).decode.__wrapped__(model,target,horizon=96,past_only_covariates=cov)[:,:target.shape[1]][0]
        den=(y.abs()*m).sum(-1).clamp_min(1e-4);optimized_q=q.sort(-1).values*.5+shift[:,:,None] if shift is not None else q
        point=((optimized_q[:,:,4]-y).abs()*m).sum(-1)/den
        err=y[:,:,None]-optimized_q;pin=(torch.maximum(quant*err,(quant-1)*err)*m[:,:,None]).sum((1,2))/den/9
        anchor_loss=(q-tq).abs().mean((1,2))/(den/m.sum(-1).clamp_min(1));bounds=(torch.relu(-q[:,:,4])+torch.relu(q[:,:,4]-1)).mean();cross=torch.relu(q[:,:,:-1]-q[:,:,1:]).mean()
        loss=point.mean()+.1*pin.mean()+.5*anchor_loss.mean()+.1*bounds+.1*cross
        assert torch.isfinite(loss), 'Nonfinite loss; candidate rejected'
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params), 'Nonfinite gradient; candidate rejected'
        torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        for g in opt.param_groups:g['lr']=cfg['lr']*(.2+.8*(1-step/300))
        if step%25==0:
            row=dict(step=step,point=float(point.mean().detach()),anchor=float(anchor_loss.mean().detach()),loss=float(loss.detach()));hist.append(row);print('TRAIN',cutoff,cfg,row,flush=True)
        if callback and step in STEPS:callback(f,step)
    del teacher;gc.collect();torch.cuda.empty_cache();return f,hist
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev');a=ap.parse_args();d=load();origins=DEV if a.split=='dev' else TEST;cutoff=14 if a.split=='dev' else 22;prefix=a.split;p={};rows=[];y,m=truth(d,origins);configs=CONFIGS;maxstep=300
    if a.split=='test':
        s=json.loads((R/'results/selection.json').read_text());key=s['chosen']['tf3'];name,step=key.rsplit('_s',1);configs={name:CONFIGS[name]};maxstep=int(step)
    for name,cfg in configs.items():
        def checkpoint(f,step):
            key=f'{name}_s{step}';p[key]=np.clip(forecast(f,d,origins,BASECFG['tf3_cellprod']),0,1);row=dict(model=key,**metrics(y,p[key],m));rows.append(row);save_delta(f,R/'models'/f'{key}_{prefix}.pt',cfg,cutoff,step);np.savez_compressed(R/f'results/tf3_{prefix}.npz',**p);pd.DataFrame(rows).to_csv(R/f'results/tf3_{prefix}.csv',index=False);print('RESULT',row,flush=True)
        f,hist=train(d,cutoff,cfg,maxstep,checkpoint if a.split=='dev' else None)
        if a.split=='test':checkpoint(f,maxstep)
        save_json(R/'results'/f'{name}_history_{prefix}.json',hist);del f;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
