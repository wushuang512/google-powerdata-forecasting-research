import argparse,json,gc,subprocess
from round2_lib import *
from extras import EXTRA
from combine import history,apply_weights
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--origin',type=int,default=31);ap.add_argument('--days',type=int,choices=[1,7],default=1);args=ap.parse_args()
    if not 14<=args.origin<=31:raise ValueError('origin must be 14..31 for this dataset')
    s=json.loads((R/'results/selection.json').read_text());d=load();configs={**CONFIGS,**EXTRA};p={};base=None
    for name in s['needed']:
        if name.startswith('tirex'):continue
        ft=None
        if name.startswith('history'):p[name]=history(d,[args.origin],args.days*96,int(name[7:]));continue
        if name.startswith('zero_'):
            if base is None:base=base_chronos()
            pipe=base;kind=name[5:]
        else:
            from chronos import Chronos2Pipeline
            ckpt=R/'models'/f'{name}_d22'/'finetuned-ckpt'
            ft=Chronos2Pipeline.from_pretrained(str(ckpt),device_map='cuda' if torch.cuda.is_available() else 'cpu');pipe=ft;kind=configs[name]['kind']
        p[name]=predict(pipe,d,[args.origin],kind,h=args.days*96)
        del pipe,ft;gc.collect();torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-u','-X','utf8',str(R/'code/tirex_run.py'),'--split','deploy','--origin',str(args.origin),'--days',str(args.days)],check=True)
    p.update(dict(np.load(R/f'results/tirex_deploy_{args.origin}_{args.days}day.npz')))
    p['old_equal_mix']=(p['zero_uni']+p['tirex_uni'])/2
    curve,peak=apply_weights(p,s,d,[args.origin]);recs=[]
    for i,name in enumerate(d['names']):
        for t in range(args.days*96):
            ts=pd.Timestamp('2019-05-01')+pd.Timedelta(days=args.origin,minutes=t*15)
            recs.append((name,str(d['cells'][i]),str(ts),float(curve[i,0,t]),float(peak[i,0,t])))
    path=R/f'results/deployment_origin{args.origin}_{args.days}day.csv'
    pd.DataFrame(recs,columns=['series','cell','local_trace_time_PT','curve_predicted_util','peak_predicted_util']).to_csv(path,index=False)
    print('Saved',path,flush=True)

if __name__=='__main__':main()
