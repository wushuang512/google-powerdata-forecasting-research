"""Partial-shard CPU/memory covariate exploration; never a total-workload estimate."""
import resource_experiment as ex
import numpy as np,datetime,json
from pathlib import Path
ex.R=ex.B/'round7'
original=ex.features
def loadr():
    r=np.load(ex.ROOT/'aggregated/resources_15min.npz')['resource'];u=np.load(ex.ROOT/'aggregated/usage_sample_15min.npz')['features'];return np.concatenate([r,u],-1)
def features(d,o,b,r,use):
    x,bb,nn=original(d,o,b,r[:,:,:,:8],True)
    if not use:return x,bb,nn
    u=r[:,:o,:,8:];extra=[];names=[]
    for k in [1,3,7]:
        h=u[:,-k:];mask=h[:,:,:,5];cnt=mask.sum((1,2)).clip(1)
        for j in range(5):
            mean=(h[:,:,:,j]*mask).sum((1,2))/cnt;std=np.sqrt((((h[:,:,:,j]-mean[:,None,None])**2)*mask).sum((1,2))/cnt)
            for a,s in [(mean,'mean'),(std,'std')]:extra.append(np.repeat(a[:,None],96,1).reshape(-1,1));names.append(f'partial_task_sample_{j}_{s}{k}')
        extra.append(np.repeat(mask.mean((1,2))[:,None],96,1).reshape(-1,1));names.append(f'partial_task_sample_availability{k}')
    for j in range(6):extra.append(u[:,-1,:,j].reshape(-1,1));names.append(f'partial_task_sample_lastslot_{j}')
    return np.concatenate([x,*extra],1),bb,nn+names
def main():
    ex.loadr=loadr;ex.features=features;ex.CONFIGS={k:v for k,v in ex.CONFIGS.items() if v['kind']=='ridge'}
    for sub in ['code','docs','models','results']:(ex.R/sub).mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registry=ex.CONFIGS,training_origins=ex.TRAIN,development_origins=ex.DEV,final_training_origins=ex.TRAIN+ex.DEV,test_origins=ex.TEST,selection='First four development days choose config, latter four alpha using paired-cell one-SE preferring alpha0. Freeze hash before test scoring.',resource_groups='control = complete machine event features; resource = same features plus partial task-window CPU/memory sample in cell a',partial_sample='Smallest official JSON shard selected by compressed size, not labels; only JOB records. Unknown inclusion probabilities and nonrepresentative coverage. Mean of observed windows, no claim of population workload.',forecast_inputs='Only completed records and past days; no future resource or workload values',independent_confirmation=False)
    ex.save_json(ex.R/'docs/protocol.json',protocol);ex.main()
    path=ex.R/'results/summary.json';s=json.loads(path.read_text());s['actual_task_usage_used']=True;s['actual_task_usage_scope']='Partial job-window sample, cell a only; not total workload';s['group_definition']='control = resources alone; resource = resources plus partial CPU/memory';ex.save_json(path,s)
if __name__=='__main__':main()
