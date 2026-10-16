import sys,json,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent;R3=B/'round3'
sys.path.insert(0,str(R3/'code'))
from lib import load,truth,metrics,np,TRAIN,DEV,TEST
from timesfm3_run import forecast,CONFIGS,TimesFM3Forecaster
import pandas as pd
# Fixed model selected earlier; this is a post-selection robustness audit, never independent confirmation.
def main():
    d=load()
    if not (R3/'results/tf3_train.npz').exists():
        model=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device='cuda',per_core_batch_size=4,local_files_only=True)
        p=forecast(model,d,TRAIN,CONFIGS['tf3_cellprod']);np.savez_compressed(R3/'results/tf3_train.npz',tf3_cellprod=p)
    rows=[]
    for split,origins in [('train',TRAIN),('dev',DEV),('test',TEST)]:
        c=np.load(R3/f'results/chronos_{split}.npz')['native'];t=np.load(R3/f'results/tirex_{split}.npz')['native'];f=np.load(R3/f'results/tf3_{split}.npz')['tf3_cellprod'];old=(c+t)/2;new=(old+f)/2;y,m=truth(d,origins)
        for j,o in enumerate(origins):
            for k,z in [('old_equal_mix',old),('secondary_equal_mix',new)]:rows.append(dict(model=k,target_day=o+1,period=split,**metrics(y[:,j:j+1],z[:,j:j+1],m[:,j:j+1])))
    pd.DataFrame(rows).to_csv(R/'results/early_audit.csv',index=False)
    print(pd.DataFrame(rows).groupby(['period','model']).wape.mean(),flush=True)
if __name__=='__main__':main()
