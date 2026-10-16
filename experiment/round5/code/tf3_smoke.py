from lib5 import *
from timesfm3 import TimesFM3Forecaster
from timesfm3.torch.timesfm3_forecaster import linear_interpolation
from round2_lib import inputs
def main():
    d=load();f=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device=DEVICE,per_core_batch_size=4,local_files_only=True);model=f.model
    for p in model.parameters():p.requires_grad_(False)
    for p in model.output_head.parameters():p.requires_grad_(True)
    inp=inputs(d,14,'cellprod',context=672)[0];target=tensor(linear_interpolation(inp['target'])[None]);cov=inp.get('past_covariates',{});cv=tensor(linear_interpolation(np.stack(list(cov.values())))[None]) if cov else None
    model.eval()
    with torch.no_grad():a=model.decode(target,horizon=96,past_only_covariates=cv)
    q=type(model).decode.__wrapped__(model,target,horizon=96,past_only_covariates=cv)
    diff=(q-a).abs().max();loss=(q[:,:,:,4]-.7).abs().mean();loss.backward()
    out=dict(shape=list(q.shape),gradient_enabled=q.requires_grad,reference_difference=float(diff.detach()),finite_gradient=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.output_head.parameters()),max_gradient=max(float(p.grad.abs().max()) for p in model.output_head.parameters()))
    save_json(R/'results/tf3_gradient_smoke.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
