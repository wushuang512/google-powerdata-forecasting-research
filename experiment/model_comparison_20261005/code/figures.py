from pathlib import Path
import sys,json
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
TOP=Path(sys.argv[1]).resolve();O=TOP/'experiment/model_comparison_20261005';S=json.loads((O/'results/summary.json').read_text());A=S['models']
font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
ids=['best','tf3','r5','r4','equal','tirex','chronoscov','tf25','chronos','ridge','lgb','last']
fig,axs=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
labels=[A[i]['name'].replace('当前最佳 R8b','当前最佳').replace('Chronos 与 TiRex 等权组合','Chronos 与 TiRex').replace('Chronos 2 历史协变量','Chronos 2 协变量') for i in ids]
for ax,key,title in zip(axs,['wape','peak_mape'],['次日曲线 WAPE %','日峰值 MAPE %']):
    v=[A[i][key] for i in ids];ax.barh(np.arange(len(ids)),v,color=['#176785' if i=='best' else '#77A9A4' if i=='last' else '#AFBCC9' for i in ids]);ax.invert_yaxis();ax.set_yticks(np.arange(len(ids)),labels if key=='wape' else []);ax.set_title(title,fontproperties=font);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    for j,x in enumerate(v):ax.text(x+.025,j,f'{x:.3f}',va='center',fontsize=9)
    ax.set_xlim(0,max(v)*1.22)
fig.savefig(O/'figures/curve_and_peak.png',dpi=180);plt.close(fig)
df=pd.read_csv(O/'results/per_day.csv');fig,ax=plt.subplots(figsize=(10.8,3.7),layout='constrained')
for i,c in [('best','#176785'),('tf3','#B57A4E'),('equal','#5C8C71'),('last','#9CA5B2')]:
    x=df[df.model==i];ax.plot(range(23,32),x.wape,label=A[i]['name'],color=c,marker='o',ms=4,lw=1.7)
ax.set_ylabel('逐日全域曲线 WAPE %',fontproperties=font);ax.set_xlabel('2019年5月预测日',fontproperties=font);ax.set_xticks(range(23,32));ax.legend(prop=font,ncol=2,loc='upper center');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'figures/day_stability.png',dpi=180);plt.close(fig)
print('Created two scientific figures')
