"""Synthetic implementation checks only; never a complete-source model score."""
import copy, json
import numpy as np
import torch
import complete_usage_models as ex

def main():
    d=ex.load();rr=np.load(ex.ROOT/'aggregated/resources_15min.npz')
    r=rr['resource'];cap=r[:,:,:,2:4]*rr['mapped_machines'][:,None,None,None]
    u=np.zeros((57,31,96,6));ix=d['cells']=='a';rng=np.random.default_rng(20261001)
    u[ix,:,:,:2]=rng.uniform(.5,100,size=(int(ix.sum()),31,96,2))
    u[ix,:,:,2:]=1;present=u[:,:,:,3]>0
    v=dict(u=u,r=r,cap=cap,present=present,
           ratios=np.divide(u[:,:,:,:2],cap,out=np.zeros_like(cap),where=cap>0))
    origins=[14];prior={14:ex.previous('dev')[:,0]}
    xx=ex.neural_inputs(d,v,origins,prior);checks={}
    checks['eleven_input_channels']=xx[0].shape==(57,672,11)
    state=torch.load(ex.B/'round5/models/fusion16_e10_dev.pt',weights_only=True,map_location=ex.DEVICE)['state']
    for name,cfg in [('full',dict(use_cpu=True,cross=True,freq=True)),
                     ('no_cpu',dict(use_cpu=False,cross=True,freq=True)),
                     ('no_cross',dict(use_cpu=True,cross=False,freq=True)),
                     ('no_frequency',dict(use_cpu=True,cross=True,freq=False))]:
        torch.manual_seed(20261001)
        net=ex.CapacityFusionMoE(width=16,**cfg).to(ex.DEVICE)
        loaded=net.load_state_dict(state,strict=False)
        checks[name+'_checkpoint_compatible']=not loaded.unexpected_keys and loaded.missing_keys==['resource_embed.weight']
        p,w,balance,gate=net(*xx)
        checks[name+'_finite_quantiles']=bool(torch.isfinite(p).all() and torch.isfinite(w).all())
        checks[name+'_power_bounds_order']=bool((p>=0).all() and (p<=1).all() and (p.diff(dim=-1)>=-1e-7).all())
        checks[name+'_cpu_nonnegative_order']=bool((w>=0).all() and (w.diff(dim=-1)>=-1e-7).all())
        loss=p.mean()+.01*w.mean()+.002*balance
        loss.backward()
        checks[name+'_finite_gradients']=all(z.grad is None or torch.isfinite(z.grad).all() for z in net.parameters())
        if name=='no_cpu':
            net.eval()
            perturb=xx[0].clone();perturb[:,:,1]*=100;perturb[:,:,4]+=1000
            with torch.inference_mode():
                pp=net(*xx)[0];qq=net(perturb,*xx[1:])[0]
            checks['no_cpu_control_power_invariant']=bool(torch.equal(pp,qq))
        del net
    dd={k:(z.copy() if isinstance(z,np.ndarray) else z) for k,z in d.items()}
    vv={k:z.copy() for k,z in v.items()}
    for key in ['y','prod','valid','pvalid']:dd[key][:,14:]=0
    for key in vv:vv[key][:,14:]=0
    yy=ex.neural_inputs(dd,vv,origins,prior)
    checks['future_power_cpu_resource_invariant']=all(torch.equal(a,b) for a,b in zip(xx,yy))
    try:
        ex.inputs()
        checks['incomplete_source_refused']=False
    except AssertionError as e:
        checks['incomplete_source_refused']='Refuse incomplete CPU export' in str(e)
    assert all(checks.values()),checks
    out=dict(synthetic_implementation_only=True,actual_complete_cpu_training=False,
             all_passed=True,checks=checks)
    ex.save_json(ex.ROOT/'results/complete_model_preflight.json',out)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
