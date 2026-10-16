from pathlib import Path
import sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
TOP=Path(sys.argv[1]).resolve();O=TOP/'experiment/model_comparison_20261005'
R=json.loads((O/'results/task_revision/summary.json').read_text());A={a['model']:a for a in R['models']};Q={a['model']:a for a in R['seven_day']}
F=O/'figures/task_revision';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
ids=['best','tf3','r4','equal','tirex','r2peak','last'];labels=[R['short_names'][i] for i in ids]
fig,ax=plt.subplots(1,2,figsize=(10.6,5.6),sharey=True)
for a,k,t in zip(ax,['wape','peak_mape'],['24小时完整曲线 WAPE %','24小时每日峰值 MAPE %']):
 vals=[A[i][k] for i in ids];a.barh(np.arange(len(ids)),vals,color='#3f6d91',height=.65);a.set_xlabel(t);a.set_xlim(0,max(vals)*1.23);a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
 for j,v in enumerate(vals):a.text(v+.04,j,f'{v:.4f}',va='center',fontsize=10)
ax[0].set_yticks(range(len(ids)),labels);ax[0].invert_yaxis();fig.tight_layout();fig.savefig(F/'24h_curve_peak.png',dpi=180);plt.close(fig)
colors=['#285b87','#ba603a','#469378','#82639c','#666666']; ids=['tf3_uni21_7','best_transfer7','tirex','r4','last']
for metric,filename,title in [('curve_wape','seven_curve_horizon.png','未来7天完整曲线 按预测日拆分的WAPE'),('daily_mean_wape','seven_mean_horizon.png','未来7天每日平均负荷 按预测日拆分的WAPE')]:
 fig,ax=plt.subplots(figsize=(10.5,3.2))
 for i,c in zip(ids,colors):
  a=sorted([x for x in R['horizons'] if x['model']==i],key=lambda x:x['forecast_day']);ax.plot(range(1,8),[v[metric] for v in a],marker='o',ms=4,label=R['short_names'][i].replace('\n',' '),color=c)
 ax.set_xlabel('预测第几天');ax.set_ylabel('WAPE %');ax.set_xticks(range(1,8));ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(0,1.30),fontsize=9,ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.89));fig.savefig(F/filename,dpi=180);plt.close(fig)
print(F)
