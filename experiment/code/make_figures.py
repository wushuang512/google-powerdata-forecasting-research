import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
def main():
    d=load();s=json.loads((ROOT/'results/selection24.json').read_text());primary=s['primary24']
    table=pd.read_csv(ROOT/'results/test24_metrics.csv').set_index('model')
    labels={'last':'Previous day','mean3':'3-day mean','chronos_zero':'Chronos-2 zero-shot',s['families']['chronos_lora']:'Chronos-2 LoRA','chronos_joint_raw':'Joint components','chronos_joint_physical':'Physical reconciliation',s['families']['patch_physical']:'Patch model + constraints','ensemble':'Validation ensemble','ensemble_peak':'Ensemble + peak prior'}
    keys=[k for k in labels if k in table.index]
    fig,ax=plt.subplots(figsize=(8.6,4.0));values=table.loc[keys,'wape']
    bars=ax.barh([labels[k] for k in keys],values,color=['#117c83' if k==primary else '#8fa9bd' for k in keys])
    for b,v in zip(bars,values):ax.text(v+.02,b.get_y()+b.get_height()/2,f'{v:.3f}%',va='center',fontsize=9)
    ax.set_xlim(0,max(values)*1.17);ax.invert_yaxis();ax.set_xlabel('24-hour macro WAPE (%)');ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(ROOT/'figures/model_comparison.png');plt.close(fig)
    days=pd.read_csv(ROOT/'results/per_day_metrics.csv')
    fig,ax=plt.subplots(figsize=(8.6,3.5))
    for key,label,color in [('last','Previous day','#aab5bd'),('chronos_zero','Chronos-2 zero-shot','#4c78a8'),(primary,'Development-selected model','#117c83')]:
        a=days[days.model==key];ax.plot(a.target_day,a.wape,'o-',label=label,color=color)
    ax.set_xlabel('Target day in May 2019');ax.set_ylabel('Macro WAPE (%)');ax.set_xticks(range(23,32));ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(ROOT/'figures/daily_error.png');plt.close(fig)
    cells=pd.read_csv(ROOT/'results/per_cell_metrics.csv').pivot(index='cell',columns='model',values='wape')
    fig,ax=plt.subplots(figsize=(8.6,3.5));xx=np.arange(len(cells))
    ax.bar(xx-.18,cells['last'],.36,label='Previous day',color='#aab5bd');ax.bar(xx+.18,cells[primary],.36,label='Selected model',color='#117c83')
    ax.set_xticks(xx,cells.index);ax.set_xlabel('Cell');ax.set_ylabel('Mean domain WAPE (%)');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(ROOT/'figures/cell_error.png');plt.close(fig)
    # A transparent representative example: median domain improvement, first test day.
    domain=pd.read_csv(ROOT/'results/per_domain_metrics.csv').pivot(index='series',columns='model',values='wape')
    ranked=(domain['last']-domain[primary]).sort_values();name=ranked.index[len(ranked)//2];i=list(d['names']).index(name)
    pp=np.load(ROOT/'results/test24_predictions.npz');y,m=truth(d,TEST24)
    fig,ax=plt.subplots(figsize=(8.6,3.5));xx=np.arange(96)/4
    ax.plot(xx,y[i,0],label='Observed',color='#20252b',lw=1.5);ax.plot(xx,pp['last'][i,0],label='Previous day',color='#aab5bd',lw=1);ax.plot(xx,pp[primary][i,0],label='Selected model',color='#117c83',lw=1.5)
    ax.set_title(f'{name}, target day 23 (median improvement domain)',fontsize=10);ax.set_xlabel('Hour');ax.set_ylabel('Power utilization');ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/example_curve.png');plt.close(fig)
    data=dict(example_domain=str(name),example_day=23,selection_rule='median per-domain improvement over persistence, first test day')
    (ROOT/'results/figure_notes.json').write_text(json.dumps(data,indent=2))
    # Profile and variability use development data only.
    profile=d['y'][:,:14].mean(1);levels=d['y'][:,:14].mean((1,2));cv=d['y'][:,:14].std((1,2))/levels
    pd.DataFrame({'series':d['names'],'cell':d['cells'],'development_mean_util':levels,'development_cv':cv,'production_valid_fraction':d['pvalid'][:,:14].mean((1,2))}).to_csv(ROOT/'results/load_characteristics.csv',index=False)
    print('Four verified-data charts generated.')

if __name__=='__main__':main()
