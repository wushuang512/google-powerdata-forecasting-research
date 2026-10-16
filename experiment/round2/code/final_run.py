import json,gc
from round2_lib import *
from extras import EXTRA
from combine import history

def main():
    s=json.loads((R/'results/selection.json').read_text());d=load();base=base_chronos();configs={**CONFIGS,**EXTRA}
    p24=dict(np.load(R/'results/final_chronos24.npz')) if (R/'results/final_chronos24.npz').exists() else {}
    p7=dict(np.load(R/'results/final_chronos7.npz')) if (R/'results/final_chronos7.npz').exists() else {}
    for name in list(dict.fromkeys(s['needed']+['zero_cov'])):
        if name.startswith('tirex'):continue
        if name in p24 and name in p7:continue
        ft=None
        if name.startswith('history'):
            p24[name]=history(d,TEST,width=int(name[7:]));p7[name]=history(d,[22,23,24],672,int(name[7:]))
        else:
            if name.startswith('zero_'):pipe=base;kind=name[5:]
            else:
                cfg=configs[name];ckpt=R/'models'/f'{name}_d22'/'finetuned-ckpt'
                if ckpt.exists():
                    from chronos import Chronos2Pipeline
                    ft=Chronos2Pipeline.from_pretrained(str(ckpt),device_map='cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    print('FINAL TRAIN',name,flush=True);ft=train(base,d,22,cfg,name)
                pipe=ft;kind=cfg['kind']
            p24[name]=predict(pipe,d,TEST,kind);p7[name]=predict(pipe,d,[22,23,24],kind,h=672)
        np.savez_compressed(R/'results/final_chronos24.npz',**p24);np.savez_compressed(R/'results/final_chronos7.npz',**p7)
        print('SAVED',name,flush=True);del ft;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
