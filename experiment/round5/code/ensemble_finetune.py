import finetune_tf3 as ft
from lib5 import *
def main():
    cfg=dict(scope='head',lr=1e-4,objective='ensemble');name='tf3_ensemble_head_lr1e4';d=load();origins=DEV;y,m=truth(d,origins);path=R/'results/tf3_ensemble_dev.npz';p={};rows=[]
    save_json(R/'docs/ensemble_finetune_extension.json',dict(added_before_test=True,configuration=cfg,reason='Fine tune the existing fixed three-model ensemble objective, instead of only the TimesFM component objective',training_origins=TRAIN,checkpoints=ft.STEPS,teacher_anchor=.5,uses_existing_causal_forecasts=True,no_test_labels=True))
    def checkpoint(f,step):
        key=f'{name}_s{step}';p[key]=np.clip(ft.forecast(f,d,origins,ft.BASECFG['tf3_cellprod']),0,1);rows.append(dict(model=key,**metrics(y,p[key],m)));ft.save_delta(f,R/'models'/f'{key}_dev.pt',cfg,14,step);np.savez_compressed(path,**p);pd.DataFrame(rows).to_csv(R/'results/tf3_ensemble_dev.csv',index=False);print('RESULT',rows[-1],flush=True)
    f,hist=ft.train(d,14,cfg,300,checkpoint);save_json(R/'results/ensemble_finetune_history.json',hist)
if __name__=='__main__':main()
