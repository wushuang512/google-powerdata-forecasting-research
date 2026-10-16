import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent;R3=B/'round3'
sys.path.insert(0,str(R3/'code'))
from lib import load,truth,metrics,np,TRAIN,DEV,TEST
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    d=load();rows=[]
    for split,origins in [('train',TRAIN),('dev',DEV),('test',TEST)]:
        old=(np.load(R3/f'results/chronos_{split}.npz')['native']+np.load(R3/f'results/tirex_{split}.npz')['native'])/2
        new=np.clip((old+np.load(R3/f'results/tf3_{split}.npz')['tf3_cellprod'])/2,0,1);y,m=truth(d,origins)
        pieces=[('May15-18',slice(0,4)),('May19-22',slice(4,8))] if split=='dev' else [(f'May{origins[0]+1}-{origins[-1]+1}',slice(None))]
        for label,sl in pieces:
            a=metrics(y[:,sl],old[:,sl],m[:,sl])['wape'];b=metrics(y[:,sl],new[:,sl],m[:,sl])['wape'];rows.append(dict(period=label,old=a,new=b,improvement_pp=a-b,relative_percent=(a-b)/a*100))
    pd.DataFrame(rows).to_csv(R/'results/time_blocks.csv',index=False)
    z=np.load(R3/'results/test_predictions.npz');y,m=truth(d,TEST);old=z['old_equal_mix'];new=z['secondary_equal_mix'];loo=[]
    for cell in np.unique(d['cells']):
        keep=d['cells']!=cell;a=metrics(y[keep],old[keep],m[keep])['wape'];b=metrics(y[keep],new[keep],m[keep])['wape'];loo.append(dict(excluded_cell=str(cell),old=a,new=b,improvement_pp=a-b))
    pd.DataFrame(loo).to_csv(R/'results/leave_one_cell_out_descriptive.csv',index=False)
    dayold=[metrics(y[:,j:j+1],old[:,j:j+1],m[:,j:j+1])['wape'] for j in range(9)];daynew=[metrics(y[:,j:j+1],new[:,j:j+1],m[:,j:j+1])['wape'] for j in range(9)]
    fig,ax=plt.subplots(figsize=(9,3.25));ax.plot(range(23,32),dayold,'o-',label='Previous best',color='#536574');ax.plot(range(23,32),daynew,'s-',label='New fixed ensemble',color='#157A83');ax.set(xlabel='Target day in May 2019',ylabel='Macro WAPE (%)',xticks=range(23,32));ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);fig.tight_layout();(R/'figures').mkdir(exist_ok=True);fig.savefig(R/'figures/test_daily.png',dpi=180);plt.close(fig)
    (R/'results/robustness.json').write_text(json.dumps(dict(time_blocks=rows,leave_one_cell_out=loo,post_selection_audit=True,independent_validation=False),indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
