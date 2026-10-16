"""Models for a COMPLETE official cell-a projection; future workloads are forecast."""
import sys,copy,json,pickle,datetime,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round9'
sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,truth,metrics,TRAIN,DEV,TEST,anchor,save_json,tensor,DEVICE,history
from calibration import bank,features as basefeatures,fit as basefit
import resource_experiment as resource
from fusion_model import FusionMoE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import os
torch.set_num_threads(int(os.environ.get('GOOGLE_POWERDATA_CPU_THREADS','4')))

def inputs():
    z=np.load(ROOT/'aggregated/complete_usage_a_15min.npz')
    assert bool(z['complete']) and int(z['source_files'])==int(z['expected_source_files'])==12085,'Refuse incomplete CPU export'
    d=load();assert np.array_equal(z['names'],d['names']);u=z['aggregate'];rr=np.load(ROOT/'aggregated/resources_15min.npz');r=rr['resource'];cnt=rr['mapped_machines']
    cap=r[:,:,:,2:4]*cnt[:,None,None,None];present=u[:,:,:,3]>0;ratios=np.divide(u[:,:,:,:2],cap,out=np.zeros_like(cap),where=cap>0)
    return d,dict(u=u,r=r,cap=cap,present=present,ratios=ratios)

def previous(split):
    if split=='dev':
        s=json.loads((B/'round8b/results/selection.json').read_text());k=s['config'];a=s['chosen']['alpha'];old=.5*anchor('dev')+.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000']
        return (1-a)*old+a*np.load(B/'round8b/results/dev_predictions.npz')[k]
    return np.load(B/'round8b/results/test_predictions.npz')['selected']

def teacher_bank(d,origins):
    b=bank(['train','dev','test']);model,_=basefit(d,b,origins,100000)
    p={}
    for o in origins:
        x,base,_=basefeatures(d,o,b);p[o]=np.clip(base+.5*model.predict(x).reshape(57,96),0,1)
    return p

def forecasts(d,v):
    """All inputs end strictly before o*96, including the test-period rolling history."""
    from low_memory_timesfm import load_forecaster
    path=R/'results/workload_forecasts.npz'
    result=dict(np.load(path)) if path.exists() else {}
    if all(str(o) in result for o in TRAIN+DEV+TEST+[31]):return result
    ix=np.where(d['cells']=='a')[0];f,load_audit=load_forecaster(B/'models/timesfm3-base',batch=4)
    # Allocation placement must reproduce an already cached development forecast.
    # This compares model outputs only, without using future power labels to fit.
    from round2_lib import inputs as original_foundation_inputs
    regression_input=original_foundation_inputs(d,14,'cellprod',h=96,context=1344)[0]
    covariance=regression_input.get('past_covariates',{})
    regression=f.predict(regression_input['target'],horizon=96,past_only_covariates=np.stack(list(covariance.values())) if covariance else None,use_symmetric_averaging=False).forecast
    expected=np.load(B/'round3/results/tf3_dev.npz')['tf3_cellprod'][ix,0]
    same=bool(np.allclose(regression,expected,atol=2e-6,rtol=1e-6))
    load_audit['inference_regression']=dict(development_origin=14,domains=5,points=480,maximum_absolute_difference=float(np.abs(regression-expected).max()),absolute_tolerance=2e-6,relative_tolerance=1e-6,passed=same)
    load_audit['checks']['cached_development_inference_regression']=same;load_audit['all_passed']=all(load_audit['checks'].values());save_json(ROOT/'results/low_memory_model_load_verification.json',load_audit);assert load_audit['all_passed'],'Low-memory allocation changed the existing model prediction';print('LOW_MEMORY_LOAD_VERIFIED',load_audit['inference_regression'],flush=True)
    raw=v['u'][:,:,:,[0,1,2]].transpose(0,3,1,2).reshape(57,3,-1)
    for o in TRAIN+DEV+TEST+[31]:
        if str(o) in result:continue
        e=o*96;target=raw[ix,:,max(0,e-1344):e].reshape(15,-1);pred=f.predict(target,horizon=96).forecast
        out=np.zeros((57,96,3));out[ix]=np.maximum(pred,0).reshape(5,3,96).transpose(0,2,1);result[str(o)]=out
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**result);temporary.replace(path);print('WORKLOAD_FORECAST',o,flush=True)
    del f;torch.cuda.empty_cache();return result

def residual_features(d,v,o,b,group,wf):
    x,base,names=resource.features(d,o,b,v['r'],True)
    if group=='control':return x,base,names
    extra=[];nn=[];present=v['present'];raw=np.concatenate([np.log1p(v['u'][:,:,:,:3]),v['ratios'],present[:,:,:,None]],-1)
    def add(a,n):
        if a.ndim==1:a=np.repeat(a[:,None],96,1)
        extra.append(a.reshape(-1,1));nn.append(n)
    for k in [1,3,7]:
        h=raw[:,o-k:o];mask=present[:,o-k:o];count=mask.sum((1,2)).clip(1)
        for j in range(5):
            mean=(h[:,:,:,j]*mask).sum((1,2))/count;sd=np.sqrt(((h[:,:,:,j]-mean[:,None,None])**2*mask).sum((1,2))/count)
            add(mean,f'complete_usage_mean{j}_{k}');add(sd,f'complete_usage_std{j}_{k}')
        add(mask.mean((1,2)),f'usage_present{k}')
    for j in range(6):add(raw[:,o-1,:,j],f'complete_usage_lastslot{j}')
    for j in [0,1]:
        add(v['ratios'][:,o-1,:,j]-v['ratios'][:,o-7:o,:,j].mean((1,2))[:,None],f'usage_profile_change{j}')
    if group=='forecast':
        for j in range(3):
            q=wf[str(o)][:,:,j];last=v['u'][:,o-1,:,j];scale=v['u'][:,o-7:o,:,j].mean((1,2)).clip(1e-6)
            add(np.log1p(q),f'forecast_usage_log{j}');add((q-last)/scale[:,None],f'forecast_usage_change{j}')
    return np.concatenate([x,*extra],1),base,names+nn

def fit_residual(d,v,origins,cfg,wf):
    b=bank(['train','dev','test']);xs=[];ys=[];ws=[]
    for o in origins:
        x,base,names=residual_features(d,v,o,b,cfg['group'],wf);m=d['valid'][:,o].ravel();xs.append(x[m]);ys.append((d['y'][:,o]-base).ravel()[m]);ws.append(np.repeat(1/d['y'][:,:o].mean((1,2)).clip(.05),96)[m])
    model=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha']));model.fit(np.concatenate(xs),np.concatenate(ys),ridge__sample_weight=np.concatenate(ws));return model,names

def predict_residual(model,d,v,origins,cfg,wf,prior):
    b=bank(['train','dev','test']);out=prior.copy();ix=d['cells']=='a'
    for j,o in enumerate(origins):
        x,base,_=residual_features(d,v,o,b,cfg['group'],wf);p=np.clip(base+model.predict(x).reshape(57,96),0,1);out[ix,j]=p[ix]
    return out

def monotone_features(v,o,wf,i):
    last=v['u'][i,o-1,:,:2];scale=v['u'][i,o-7:o,:,:2].mean((0,1)).clip(1e-6)
    return (wf[str(o)][i,:,:2]-last)/scale[None]

def fit_candidate(d,v,origins,cfg,wf):
    if cfg['kind']=='ridge':return fit_residual(d,v,origins,cfg,wf)
    tb=teacher_bank(d,origins);models={}
    for i in np.where(d['cells']=='a')[0]:
        x=np.concatenate([monotone_features(v,o,wf,i) for o in origins]);y=np.concatenate([d['y'][i,o]-tb[o][i] for o in origins]);mask=np.concatenate([d['valid'][i,o] for o in origins]);reg=Ridge(alpha=cfg['alpha'],positive=True,solver='lbfgs',max_iter=1000,tol=1e-8)
        if cfg['kind']=='physical_control':x=x*0
        reg.fit(x[mask],y[mask]);models[int(i)]=reg
    return models,['forecast_cpu_relative_change','forecast_memory_relative_change']

def predict_candidate(model,d,v,origins,cfg,wf,prior):
    if cfg['kind']=='ridge':return predict_residual(model,d,v,origins,cfg,wf,prior)
    out=prior.copy()
    for i,reg in model.items():
        for j,o in enumerate(origins):
            x=monotone_features(v,o,wf,i)
            if cfg['kind']=='physical_control':x=x*0
            out[i,j]=np.clip(prior[i,j]+reg.predict(x),0,1)
    return out

class CapacityFusionMoE(FusionMoE):
    """True JOB CPU NCU and memory. CPU is nonnegative; it is NOT bounded by P."""
    def __init__(self,use_cpu=True,**cfg):
        super().__init__(**cfg);self.use_cpu=use_cpu;self.resource_embed=nn.Linear(6,self.width,bias=False);nn.init.zeros_(self.resource_embed.weight)
    def forward(self,x,calendar,cell,base):
        p,pmu,psd,pz=self._parts(x[:,:,0],x[:,:,2]);mask=x[:,:,3] if self.use_cpu else x[:,:,3]*0;w,wmu,wsd,wz=self._parts(x[:,:,1],mask);mem,_,_,_=self._parts(x[:,:,4],mask);w=torch.stack([w[:,:,0],w[:,:,1],mem[:,:,0],mask],-1);availability=mask.mean(1);embedded=[];rb=self.resource_embed(x[:,:,5:11].mean(1))
        for j,s in enumerate(self.scales):
            b,l,c=p.shape;pt=p[:,-(l//s)*s:].reshape(b,-1,s*4);wt=w[:,-(l//s)*s:].reshape(b,-1,s*4);pe=self.pembed[j](pt);we=self.wembed[j](wt);pos=torch.arange(pe.shape[1],device=x.device).float();dim=torch.arange(self.width,device=x.device).float();enc=torch.sin(pos[:,None]/(10000**(dim[None]/self.width)));pe=self.penc[j](pe+enc);we=self.wenc[j](we+enc)
            if self.cross:pe=pe+self.att[j](pe,we,we,need_weights=False)[0]*availability[:,None,None]
            pooled=pe.mean(1)+rb
            if self.freq:
                fp=torch.fft.rfft(pz.float(),norm='forward')[:,1:17];fw=torch.fft.rfft(wz.float(),norm='forward')[:,1:17];pooled=pooled+self.fft_p(torch.stack([fp.real,fp.imag],-1)).mean(1)
                if self.cross:pooled=pooled+self.fft_w(torch.stack([fw.real,fw.imag],-1)).mean(1)*availability[:,None]
            embedded.append(pooled)
        ce=self.cell(cell.long());prob=self.gate(torch.cat([*embedded,ce,availability[:,None],pmu,psd],-1)).softmax(-1)
        if self.sparse:active=torch.zeros_like(prob).scatter_(1,prob.topk(2,-1).indices,1)
        else:active=torch.ones_like(prob)
        weights=prob*active;weights=weights/weights.sum(-1,keepdim=True);latent=torch.zeros_like(embedded[0]);loads=[]
        for j,expert in enumerate(self.experts):
            ix=active[:,j].bool();part=torch.zeros_like(latent)
            if ix.any():part[ix]=expert(embedded[j][ix])*weights[ix,j,None]
            latent=latent+part;loads.append(ix.float().mean())
        balance=3*(prob.mean(0)*torch.stack(loads)).sum()/active.sum(1).mean();lp=(x[:,-96:,0]-base)/psd;lw=((x[:,-96:,1]-wmu)/wsd)*availability[:,None];h=self.head(torch.cat([latent[:,None].expand(-1,96,-1),ce[:,None].expand(-1,96,-1),calendar,torch.stack([lp,lw],-1)],-1));cp=base+psd*.5*torch.tanh(h[:,:,0]);cw=x[:,-96:,1]+wsd*1.5*torch.tanh(h[:,:,3]);pq=torch.stack([cp-psd*F.softplus(h[:,:,1]-2),cp,cp+psd*F.softplus(h[:,:,2]-2)],-1).clamp(0,1);wq=torch.stack([cw-wsd*F.softplus(h[:,:,4]-2),cw,cw+wsd*F.softplus(h[:,:,5]-2)],-1).clamp_min(0)
        return pq,wq,balance,prob

def neural_inputs(d,v,origins,base):
    x,c,ci=history(d,np.array(origins)*96);x[:,:,1]=0;x[:,:,3]=0;extra=[]
    for o in origins:
        u=v['u'][:,o-7:o].reshape(57,672,6);mask=v['present'][:,o-7:o].reshape(57,672);rr=v['r'][:,o-7:o].reshape(57,672,8);extra.append(np.concatenate([u[:,:,[0]],mask[:,:,None],u[:,:,[1]],rr[:,:,[1,2,3,0,4,5]]],-1))
    z=np.concatenate(extra);x=np.concatenate([x[:,:,:1],z[:,:,:1],x[:,:,2:3],z[:,:,1:]],-1).astype('float32');bb=np.concatenate([base[o] for o in origins]).astype('float32');return tuple(tensor(a) for a in [x,c,ci,bb])

@torch.inference_mode()
def neural_predict(net,d,v,origins,base):
    net.eval();xx=neural_inputs(d,v,origins,base);ps=[];ws=[];gs=[]
    for i in range(0,len(xx[0]),64):
        p,w,_,g=net(*(a[i:i+64] for a in xx));ps.append(p.cpu().numpy());ws.append(w.cpu().numpy());gs.append(g.cpu().numpy())
    def pack(a):return np.concatenate(a).reshape(len(origins),57,96,3).transpose(1,0,2,3)
    return pack(ps),pack(ws),np.concatenate(gs)

def train_neural(d,v,origins,cfg,seed,epochs,callback=None):
    torch.manual_seed(seed);np.random.seed(seed);net=CapacityFusionMoE(**cfg).to(DEVICE);old=torch.load(B/'round5/models'/f'fusion16_e10_final_seed{seed}.pt' if origins==TRAIN+DEV else B/'round5/models'/f'fusion16_e10_dev{"" if seed==20261001 else "_seed"+str(seed)}.pt',weights_only=True,map_location=DEVICE)
    if old:
        net.load_state_dict(old['state'],strict=False);nn.init.zeros_(net.head[-1].weight);nn.init.zeros_(net.head[-1].bias)
    # Validation initialization never uses weights fitted after May 14.
    tb=teacher_bank(d,origins);xx=neural_inputs(d,v,origins,tb);y=tensor(np.concatenate([d['y'][:,o] for o in origins]).astype('float32'));wm=tensor(np.concatenate([v['present'][:,o] for o in origins]).astype('float32'));w=tensor(np.concatenate([v['u'][:,o,:,0] for o in origins]).astype('float32'));vm=tensor(np.concatenate([d['valid'][:,o] for o in origins]).astype('float32'));opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.1);qs=tensor(np.array([.1,.5,.9],dtype='float32'));hist=[]
    for ep in range(1,epochs+1):
        net.train();losses=[]
        for ix in torch.randperm(len(y),device=DEVICE).split(64):
            p,q,bal,_=net(*(a[ix] for a in xx));yy=y[ix];m=vm[ix];den=(yy*m).sum(1).clamp_min(1e-4);err=yy[:,:,None]-p;point=((p[:,:,1]-yy).abs()*m).sum(1)/den;pin=(torch.maximum(qs*err,(qs-1)*err)*m[:,:,None]).sum((1,2))/den/3;shrink=((p[:,:,1]-xx[3][ix]).abs()*m).sum(1)/den;mask=wm[ix] if cfg['use_cpu'] else wm[ix]*0;we=w[ix,:,None]-q;wd=(w[ix]*mask).sum(1).clamp_min(1e-4);wp=(torch.maximum(qs*we,(qs-1)*we)*mask[:,:,None]).sum((1,2))/wd/3;has=mask.sum(1)>0;aux=wp[has].mean() if has.any() else wp.sum()*0;loss=point.mean()+.1*pin.mean()+.05*aux+.3*shrink.mean()+.002*bal;assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(a.grad is None or torch.isfinite(a.grad).all() for a in net.parameters());torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach()))
        hist.append(dict(epoch=ep,loss=float(np.mean(losses))));print('FUSION',cfg,seed,hist[-1],flush=True)
        if callback and ep in [4,10,20]:callback(net,ep)
    return net,hist
