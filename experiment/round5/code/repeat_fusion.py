from lib5 import *
from train_fusion import CONFIGS,train,predict
def main():
    d=load();p=dict(np.load(R/'results/fusion_dev.npz'));y,m=truth(d,DEV);full={k:v for k,v in p.items() if k.startswith('fusion')};key=min(full,key=lambda k:metrics(y[:,:4],full[k][:,:4],m[:,:4])['wape']);name,ep=key.rsplit('_e',1);ep=int(ep)
    save_json(R/'results/repeat_plan.json',dict(chosen=key,selection='first four DEV days only',second_seed=20261002,aggregation='equal average of the two specified seeds; do not choose best seed',test_used=False))
    checkpoint=R/'models'/f'{key}_dev_seed20261002.pt'
    if checkpoint.exists():
        from fusion_model import FusionMoE
        saved=torch.load(checkpoint,map_location=DEVICE,weights_only=True);net=FusionMoE(**saved['config']).to(DEVICE);net.load_state_dict(saved['state']);hist=json.loads((R/'results/repeat_history.json').read_text())
    else:
        net,hist=train(d,14,CONFIGS[name],20261002,ep);torch.save(dict(state=net.state_dict(),config=CONFIGS[name],cutoff=14,epoch=ep,seed=20261002),checkpoint)
    q,w,g=predict(net,d,DEV)
    firstq=np.load(R/'results/fusion_quantiles_dev.npz')[key];np.savez_compressed(R/'results/fusion_repeat_quantiles_dev.npz',second_seed=q,equal_seed_mean=(firstq+q)/2)
    np.savez_compressed(R/'results/fusion_repeat_dev.npz',second_seed=q[:,:,:,1],equal_seed_mean=(p[key]+q[:,:,:,1])/2);save_json(R/'results/repeat_history.json',hist);print(key,metrics(y,q[:,:,:,1],m),metrics(y,(p[key]+q[:,:,:,1])/2,m),flush=True)
if __name__=='__main__':main()
