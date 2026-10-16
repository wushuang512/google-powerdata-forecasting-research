import json
from round2_lib import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10,'savefig.dpi':180})
LABELS={'last':'前一天同刻','zero_uni':'Chronos-2 原模型','zero_cov':'Chronos-2 生产特征','tirex_uni':'TiRex-2 原模型','old_equal_mix':'复现旧等权组合','round1_primary':'第一轮冻结组合','new_curve':'第二轮曲线组合','new_peak':'第二轮峰值修正'}

def main():
    summary=json.loads((R/'results/summary.json').read_text());scores=summary['horizon1'];keys=list(LABELS)
    fig,ax=plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios':[1.2,1]})
    vals=[scores[k]['wape'] for k in keys];colors=['#A7B4C0']*6+['#007A87','#52A8A5']
    ax[0].barh([LABELS[k] for k in keys],vals,color=colors);ax[0].invert_yaxis();ax[0].set_xlim(0,max(vals)*1.18);ax[0].set_xlabel('次日曲线 WAPE（%，越低越好）')
    for i,v in enumerate(vals):ax[0].text(v+.03,i,f'{v:.4f}',va='center')
    days=pd.read_csv(R/'results/per_day.csv')
    for k,c in [('old_equal_mix','#A36322'),('round1_primary','#6D7380'),('new_curve','#007A87')]:
        f=days[days.model==k];ax[1].plot(f.target_day,f.wape,marker='o',color=c,label=LABELS[k])
    ax[1].set_xlabel('2019 年 5 月预测目标日');ax[1].set_ylabel('日 WAPE（%）');ax[1].legend(frameon=False,fontsize=9);ax[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(R/'figures/accuracy.png');plt.close(fig)
    cell=pd.read_csv(R/'results/per_cell.csv').pivot(index='cell',columns='model',values='wape')
    fig,ax=plt.subplots(figsize=(8,3.2));x=np.arange(len(cell));w=.34
    ax.bar(x-w/2,cell['old_equal_mix'],w,label='复现旧等权组合',color='#A36322');ax.bar(x+w/2,cell['new_curve'],w,label='第二轮曲线组合',color='#007A87')
    ax.set_xticks(x,cell.index);ax.set_ylabel('域平均 WAPE（%）');ax.legend(frameon=False);fig.tight_layout();fig.savefig(R/'figures/cells.png');plt.close(fig)
    seven=summary['horizon7'];kk=['last','zero_uni','tirex_uni','old_equal_mix','round1_primary','new_curve','new_peak']
    fig,ax=plt.subplots(figsize=(8,3.4));v=[seven[k]['wape'] for k in kk];ax.barh([LABELS[k] for k in kk],v,color=['#A7B4C0']*5+['#007A87','#52A8A5']);ax.invert_yaxis();ax.set_xlim(0,max(v)*1.17)
    for i,z in enumerate(v):ax.text(z+.03,i,f'{z:.4f}',va='center')
    ax.set_xlabel('未来七天曲线 WAPE（%，重叠窗口探索性结果）');fig.tight_layout();fig.savefig(R/'figures/seven.png');plt.close(fig)

if __name__=='__main__':main()
