"""Diagnose the observed plateau; truth-assisted diagnostics are not forecasts."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent;B=TOP/'experiment'
sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,truth,metrics,TRAIN,DEV,TEST,save_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def weighted_lag_corr(d,p,origins):
    y,m=truth(d,origins);level=np.stack([d['y'][:,o-7:o].mean((1,2)) for o in origins],1).clip(.05)
    e=((y-p)*m).sum(2)/m.sum(2).clip(1)/level
    # Center within PDU so persistent differences cannot create pooled correlation.
    a=e[:,:-1];b=e[:,1:];a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    return float(np.corrcoef(a.ravel(),b.ravel())[0,1])

def main():
    dest=B/'accuracy_assessment'
    dest.mkdir(parents=True,exist_ok=True)
    d=load();y,m=truth(d,TEST);s5=read(B/'round5/results/summary.json');ft=read(B/'round8b/results/summary.json');gate=read(B/'round8c/results/summary.json')
    p=np.load(B/'round8b/results/test_predictions.npz')['selected'];aix=d['cells']=='a';den=(y*m).sum((1,2));norm=100/den/57;error=np.abs(y-p)*m
    baseline=np.stack([d['y'][:,o-1] for o in TEST],1)
    allscore=metrics(y,p,m)
    domains=[]
    for i,name in enumerate(d['names']):
        domains.append(dict(domain=str(name),cell=str(d['cells'][i]),wape=float(100*error[i].sum()/den[i]),contribution_pp=float(error[i].sum()*norm[i])))
    domains.sort(key=lambda z:z['contribution_pp'],reverse=True)
    cells=[]
    for c in sorted(set(d['cells'])):
        ix=d['cells']==c
        cells.append(dict(cell=str(c),domains=int(ix.sum()),**metrics(y[ix],p[ix],m[ix]),contribution_pp=float((error[ix].sum((1,2))*norm[ix]).sum())))
    cells.sort(key=lambda z:z['contribution_pp'],reverse=True)
    days=[]
    for j,o in enumerate(TEST):
        days.append(dict(date=f'2019-05-{o+1:02d}',**metrics(y[:,[j]],p[:,[j]],m[:,[j]]),contribution_pp=float((error[:,j].sum(1)*norm).sum())))
    med=np.zeros((57,9));oracle=p.copy()
    for i in range(57):
        for j in range(9):
            med[i,j]=np.median((y[i,j]-p[i,j])[m[i,j]])
            oracle[i,j]=np.clip(p[i,j]+med[i,j],0,1)
    ol=metrics(y,oracle,m)
    experts={k:np.load(B/f'round3/results/{k}_test.npz')['native'] for k in ['chronos','tirex']}
    experts['timesfm']=np.load(B/'round3/results/tf3_test.npz')['tf3_cellprod']
    names=list(experts)
    ee=np.stack([((z-y)/d['y'][:,:22].mean((1,2)).clip(.05)[:,None,None])[m] for z in experts.values()])
    corr=np.corrcoef(ee)
    # This uses actual future outcomes for hindsight selection, so cannot be deployed.
    raw=np.stack([p,*experts.values()]);idx=np.abs(raw-y).argmin(0);oracle_point=np.take_along_axis(raw,idx[None],axis=0)[0]
    selected=ft['selection'];devbase=.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000']
    from lib5 import anchor
    devbase+=.5*anchor('dev');devcand=np.load(B/'round8b/results/dev_predictions.npz')[selected['config']];devp=(1-selected['chosen']['alpha'])*devbase+selected['chosen']['alpha']*devcand
    lag_dev=weighted_lag_corr(d,devp,DEV);lag_test=weighted_lag_corr(d,p,TEST)
    rows=[dict(stage='原等权组合',wape=s5['scores']['old_equal_mix']['wape']),
          dict(stage='固定基础三模型',wape=s5['scores']['anchor']['wape']),
          dict(stage='第五轮残差组合',wape=s5['scores']['selected']['wape']),
          dict(stage='第八轮组件微调',wape=ft['scores']['selected']['wape']),
          dict(stage='最新权重微调对照',wape=gate['scores']['selected']['wape'])]
    source=read(ROOT/'results/projection_progress_a.json')
    out=dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             assessment='Practical plateau for current power/proxy inputs, not an identified irreducible forecasting limit',
             independent_confirmation=False,stage_scores=rows,baseline=metrics(y,baseline,m),current=allscore,
             current_cell_a=metrics(y[aix],p[aix],m[aix]),power_domains=57,total_days=31,training_origins=TRAIN,
             development_origins=DEV,test_origins=TEST,valid_test_points=int(m.sum()),
             recent_component_gain_pp=s5['scores']['selected']['wape']-allscore['wape'],
             recent_component_ci=ft['comparison_vs_previous']['day_and_cell_95'],
             latest_gate_gain_pp=gate['comparison_vs_previous']['difference_pp'],
             top10_domain_error_share=sum(x['contribution_pp'] for x in domains[:10])/allscore['wape'],
             top3_day_error_share=sum(sorted([x['contribution_pp'] for x in days],reverse=True)[:3])/allscore['wape'],
             cells=cells,days=days,domains=domains,expert_error_correlation=dict(names=names,matrix=corr.tolist()),
             normalized_daily_bias_lag1=dict(development=lag_dev,test=lag_test,note='Centered per PDU; few date pairs; descriptive, not proof of learnable signal'),
             hindsight_perfect_daily_offset=dict(score=ol,diagnostic_only=True,uses_future_truth=True,deployable=False),
             hindsight_per_point_expert=dict(score=metrics(y,oracle_point,m),diagnostic_only=True,uses_future_truth=True,deployable=False),
             complete_cpu_source_progress=source,complete_cpu_experiment_started=(B/'round9/results/dev_predictions.npz').exists(),
             data_npz_sha256=hashlib.sha256((B/'results/data.npz').read_bytes()).hexdigest(),
             conclusion='Stop broad repeated foundation tuning on the same May test; finish predeclared complete CPU matched experiment. Explore a causal low-capacity bias filter only if development residuals show persistence.')
    a_contribution=out['current_cell_a']['wape']*int(aix.sum())/57
    out['a_only_correction_bound']=dict(a_domains=int(aix.sum()),a_contribution_pp=a_contribution,
        all57_best_possible_wape_if_a_perfect=allscore['wape']-a_contribution,
        maximum_global_relative_reduction_percent=100*a_contribution/allscore['wape'],
        if_a_error_reduced_20_percent_global_gain_pp=.2*a_contribution,
        condition='Only cell-a predictions may change, all other52 exact prior. Optimistic mathematical bound, not an achievable forecast.')
    out['decision']=dict(broad_existing_input_tuning='Stop widening similar-model searches on reused May dates',
        complete_cpu_experiment='Continue registered matched experiment; added information has not yet been tested',
        extra_bias_filter='Not started: development centered lag correlation only 0.083; weak persistence provides no compelling basis for expanded online-filter tuning',
        independent_breakthrough='Cannot be established without unseen dates or sites; no independent data added')
    save_json(dest/'assessment.json',out)
    for name,table in [('per_domain',domains),('per_cell',cells),('per_day',days),('stage_scores',rows)]:pd.DataFrame(table).to_csv(dest/f'{name}.csv',index=False)
    tt='\n'.join(f'| {z["stage"]} | {z["wape"]:.6f} |' for z in rows)
    cc='\n'.join(f'| {z["cell"]} | {z["domains"]} | {z["wape"]:.4f} | {z["contribution_pp"]:.4f} |' for z in cells)
    text=f'''# 当前预测精度是否已到临界点

核对时间：{out['created_at']}。全部数字来自同一 57 域、九个测试日和相同有效掩码的实际预测。不是跨月独立验证。

## 判断

现有功率与生产功率代理输入下，已进入明显的工程平台期，继续宽泛地微调同类基础模型，预计收益低；目前不能确定理论最低误差，也不能断言真实算力输入没有上升空间。完整 CPU 实验尚未开始评分，不能把旧代理融合或局部 CPU 样本的失败当成完整真实工作量实验已经失败。

| 实际方案 | 测试 WAPE % |
|---|---:|
{tt}

前一天同刻基线 WAPE {out['baseline']['wape']:.6f}%，当前为 {allscore['wape']:.6f}%。第五轮以后组件微调只减少 {out['recent_component_gain_pp']:.9f} 个百分点，联合 95% 改进区间 [{out['recent_component_ci'][0]:.6f}, {out['recent_component_ci'][1]:.6f}] 跨零。最新权重微调反而增加 {-out['latest_gate_gain_pp']:.6f} 个百分点。实现检查通过不等于没有过拟合。

## 瓶颈证据

本地功率仅 31 天，开发模型训练监督起报日只有 7 个，最终拟合只有 15 个；57 域和 49,243 个有效测试点具有日期、场站相关，不能当作等量独立样本。开发期和同一五月测试期已多轮被查看，微小改进很难与选择波动区分。

Chronos 与 TiRex 的归一化误差相关 {corr[0,1]:.4f}，Chronos 与 TimesFM 为 {corr[0,2]:.4f}，TiRex 与 TimesFM 为 {corr[1,2]:.4f}。这些是同月描述性相关；相似模型可能共同遗漏同一种变化，继续改权重未必提供新信息。

当前最高十个供电域贡献了全部宏平均误差的 {out['top10_domain_error_share']*100:.2f}%，最高三个日期贡献 {out['top3_day_error_share']*100:.2f}%。逐域和逐日完整数据见 CSV。a 的五域当前 WAPE {out['current_cell_a']['wape']:.6f}%，新完整 CPU 输入目前只覆盖这五域，不可把局部改善放大为全部场站改善。

| 集群 | 域数 | 集群宏平均 WAPE % | 对全部 57 域误差贡献 pp |
|---|---:|---:|---:|
{cc}

## 上升空间的边界

如果事后知道每个域每天的最佳固定功率偏移，WAPE 可降到 {ol['wape']:.6f}%；事后逐点挑选最接近真值的专家，WAPE 为 {out['hindsight_per_point_expert']['score']['wape']:.6f}%。两者都使用未来真值，不能部署，也不代表可达到的预测精度，更不是数学上的不可约误差。这只说明部分残差属于整天的水平偏移、模型误差也并非完全一致，后续应检验能否由过去或真实负载提前识别。

每天的归一化偏差在域内中心化后，开发期一日滞后相关为 {lag_dev:.4f}，测试期为 {lag_test:.4f}。开发期持续性很弱，没有继续扩大在线误差过滤参数搜索的充分依据；因此本次不追加该搜索。测试相关只作已标注的事后诊断，没有据此改模型或调参数。

若新分支只改变 a 的五域，其他 52 域保持原预测，那么即便事后令 a 的误差为零，整体 WAPE 最多从 {allscore['wape']:.6f}% 降到 {allscore['wape']-a_contribution:.6f}%，整体最大相对误差下降 {100*a_contribution/allscore['wape']:.2f}%。这不是可实现精度，而是当前宏平均指标下局部修正的乐观数学界限。a 的误差实际降低 20% 时，整体仅下降 {.2*a_contribution:.6f} 个百分点。联合网络可能改变其他域，因此不适用此界限，但也不能把其他域变化都归因于 a 的 CPU 输入。

## 继续研究的取舍

1. 完成已登记的完整 CPU 与内存历史、预测未来负载、资源状态融合以及非负斜率响应对照；必须与同正则、同日期、同宽度的无 CPU 控制比较。使用未来真实 CPU 拟合功率不算有效预测突破。
2. 这次开发期历史偏差持续性弱，暂不扩展旧输入的在线校准搜索。任何后续小参数校准先登记范围，再做按时间划分的验证；未来功率、CPU 和未来预测误差均不得用于输入。无稳定开发期支持则保留原模型。
3. 不继续在这九天反复扩大模型和超参数搜索。完整 CPU 也未改善时，将结论写为当前信息条件下的平台期；若要证明进一步增益，需要新增月份、未参与选模场站，或起报时已知的排班、负载计划等信息。当前数据不能验证这些额外信息是否可用。

## 当前读取状态与来源

a 集群必要列读取 {source['completed']:,}/{source['expected']:,} 个对象，当前完整 CPU 模型分数仍为空。来源为 [Google 官方 PowerData2019](https://github.com/google/cluster-data/blob/master/PowerData2019.md) 与 [ClusterData2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)。完整输入范围、许可与提取审核见原桌面资料。该评估将纳入最终 Word 汇报，后续真实结果只会补充判断，不回写此前冻结选择。
'''
    (dest/'精度平台期评估.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.5),layout='constrained')
    xx=np.arange(4);vv=[z['wape'] for z in rows[:4]]
    axes[0].plot(xx,vv,'o-',color='#2563a6',label='Frozen main methods')
    axes[0].plot(4,rows[4]['wape'],'x',color='#bd3e3e',markersize=9,label='Gate candidate, not adopted')
    for j,z in enumerate(rows):axes[0].annotate(f'{z["wape"]:.6f}',(j,z['wape']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_xticks(range(5),['C+T','C+T+TF','Residual','Adapt TF','New gate'],rotation=20)
    axes[0].set_ylim(min(vv)-.012,max(vv)+.018);axes[0].set_ylabel('Macro WAPE (%)');axes[0].set_title('Diminishing gains, same May test');axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,loc='upper right')
    im=axes[1].imshow(corr,vmin=.8,vmax=1,cmap='Blues')
    axes[1].set_xticks(range(3),['Chronos','TiRex','TimesFM']);axes[1].set_yticks(range(3),['Chronos','TiRex','TimesFM']);axes[1].set_title('Highly correlated forecast errors')
    for i in range(3):
        for j in range(3):axes[1].text(j,i,f'{corr[i,j]:.3f}',ha='center',va='center',color='white' if corr[i,j]>.94 else 'black')
    fig.colorbar(im,ax=axes[1],shrink=.7)
    axes[2].bar([z['cell'] for z in cells],[z['wape'] for z in cells],color=['#ce8d34' if z['cell']=='a' else '#6491b7' for z in cells])
    axes[2].set_xlabel('Cell (only a has new complete CPU input)');axes[2].set_ylabel('Cell macro WAPE (%)');axes[2].set_title('Error varies across cells');axes[2].grid(axis='y',alpha=.2)
    fig.savefig(dest/'accuracy_plateau_diagnostics.png',dpi=150);plt.close(fig)
    print(json.dumps({k:out[k] for k in ['recent_component_gain_pp','a_only_correction_bound','decision']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
