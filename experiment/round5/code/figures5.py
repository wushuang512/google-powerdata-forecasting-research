from lib5 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    d=load();y,m=truth(d,TEST);p=np.load(R/'results/test_predictions.npz');old=[];new=[]
    for j in range(9):
        old.append(metrics(y[:,j:j+1],p['anchor'][:,j:j+1],m[:,j:j+1])['wape']);new.append(metrics(y[:,j:j+1],p['selected'][:,j:j+1],m[:,j:j+1])['wape'])
    fig,ax=plt.subplots(figsize=(9,3.2));ax.plot(range(23,32),old,'o-',label='Previous best',color='#586575');ax.plot(range(23,32),new,'s-',label='Frozen Round 5',color='#157A83');ax.set(xlabel='Target day in May 2019',ylabel='Macro WAPE (%)',xticks=range(23,32));ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);fig.tight_layout();fig.savefig(R/'figures/daily_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.2));dev=pd.read_csv(R/'results/fusion_dev.csv');names=['fusion32_e10','no_cross_e10','no_frequency_e10','dense_moe_e10'];values=[float(dev[dev.model==k].wape.iloc[0]) for k in names];ax.bar(['Full fusion','No cross-attention','No FFT branch','Dense experts'],values,color=['#157A83','#8596A3','#8596A3','#8596A3']);ax.set(ylabel='Development macro WAPE (%)',ylim=(0,3.5));ax.spines[['top','right']].set_visible(False)
    for j,v in enumerate(values):ax.text(j,v+.04,f'{v:.4f}',ha='center')
    fig.tight_layout();fig.savefig(R/'figures/ablation_development.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
