import argparse,subprocess
from round2_lib import *
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--origin',type=int,default=31);ap.add_argument('--days',type=int,choices=[1,7],default=1);ap.add_argument('--tirex-only',action='store_true');args=ap.parse_args()
    d=load()
    if not args.tirex_only:
        pipe=base_chronos();p=predict(pipe,d,[args.origin],'uni',h=args.days*96)
    subprocess.run([sys.executable,'-u','-X','utf8',str(R/'code/tirex_run.py'),'--split','legacy','--origin',str(args.origin),'--days',str(args.days)],check=True)
    tirex=np.load(R/f'results/tirex_legacy_{args.origin}_{args.days}day.npz')['tirex_uni'];p=tirex if args.tirex_only else (p+tirex)/2;rec=[]
    for i,name in enumerate(d['names']):
        for t in range(args.days*96):rec.append((name,str(d['cells'][i]),str(pd.Timestamp('2019-05-01')+pd.Timedelta(days=args.origin,minutes=t*15)),float(p[i,0,t])))
    prefix='tirex' if args.tirex_only else 'legacy'
    path=R/f'results/{prefix}_deployment_origin{args.origin}_{args.days}day.csv';pd.DataFrame(rec,columns=['series','cell','local_trace_time_PT','predicted_util']).to_csv(path,index=False);print('Saved',path)

if __name__=='__main__':main()
