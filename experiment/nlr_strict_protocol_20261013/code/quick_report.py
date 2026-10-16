"""Fast Markdown, full CSV and actual Python DataFrame tables, from new results only."""
from pathlib import Path
import argparse,shutil
import pandas as pd
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_strict_protocol_20261013'
audit=read(F/'results/independent_verification.json');assert audit['all_passed']
for rel,h in audit['bindings'].items():assert sha(R/rel)==h,rel
rows=read(F/'results/summary.json')['rows'];assert len(rows)==42;sel=read(F/'results/selection.json');q=read(F/'data/quality_report.json')
LABEL={'LightGBM':'LightGBM NLR重新拟合','timesfm3':'TimesFM3 原权重','toto2':'Toto2 原权重','moirai2':'Moirai2 原权重','HDSF':'HDSF LightGBM TimesFM3 Toto2 Moirai2 NLR单纯形适配','last':'前一天同刻','week':'上周同刻','mean3':'前三天同刻均值','Google_Chronos_TiRex_equal':'Chronos-2＋TiRex-2 等权','Google_three_fixed':'Chronos-2 25%＋TiRex-2 25%＋TimesFM3 50%','Google_R5_Ridge':'三专家固定组合＋Google第5轮Ridge校准','Google_R8b_finetuned':'三专家及Ridge＋Google第8轮TimesFM3微调差分','Google_HDSF_original':'原HDSF第44轮固定源权重迁移','Google_R48_robust':'第48轮HDSF固定源权重迁移'}
order=['last','week','mean3','LightGBM','timesfm3','toto2','moirai2','HDSF']+read(F/'docs/historical_controls.json')['schemes']
def task(H,cohort='daily_origins'):return [next(r for r in rows if r['horizon_days']==H and r['cohort']==cohort and r['model']==n) for n in order]
def value(x):return 'null' if x is None else f'{x:.4f}'
def md_table(rr,fields):
    headers=['具体模型']+[title for k,title in fields];out=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    out += ['| '+LABEL[r['model']]+' | '+' | '.join(value(r[k]) for k,t in fields)+' |' for r in rr]
    return '\n'.join(out)
def chosen(H):return LABEL[sel['tasks'][str(H)]['selected_method']]
text=f'''# NLR严格协议重跑评估简报

本轮所有十四种方案已在新的独立目录重新运行。按共同向前验证起报平均MAPE预先冻结的方案，24小时是 **{chosen(1)}**，7天是 **{chosen(7)}**。测试指标最低另列，不能据此回改选模。

本轮没有使用上轮预测缓存。原始数据重新处理，需拟合的LightGBM与单纯形融合重新拟合；基础神经权重冻结，旧Google参数作为预先登记的迁移对照。并非所有底层神经模型在NLR重新梯度训练。已暴露的测试时间段不会因重跑而成为独立盲测；时区仍未核实。

## 三项任务实际测试结果

WAPE与MAPE为百分数，越低越好。下表所有方法的测试日期、真值、缺失规则和计分代码相同。
'''
matrix=[]
for n in order:
    u=next(r for r in task(1) if r['model']==n);w=next(r for r in task(7) if r['model']==n)
    matrix.append(dict(model=n,h1=u['curve_wape_percent'],h7=w['curve_wape_percent'],daily=w['daily_wape_percent']))
text+=md_table(matrix,[('h1','24小时曲线WAPE %'),('h7','7天曲线WAPE %'),('daily','7天日均WAPE %')])+'\n\n'
for k,title in [('h1','24小时曲线'),('h7','7天曲线'),('daily','7天日均')]:
    best=min(matrix,key=lambda r:r[k]);text+=f'- {title}测试描述性最低：{LABEL[best["model"]]}，{best[k]:.6f}%。\n'
text+='\n第5及第8轮等源1天构型的7天应用是预先登记的扩展，不称原七天模型原样复现。原HDSF及第48轮也统一用14天历史与单设施适配。原Google部署仍保持不变。\n'
for H in [1,7]:
    text+=f'\n## {H}天曲线与日均详细评价\n\n'
    text+=md_table(task(H),[(k,t) for k,t in [('curve_mape_percent','曲线MAPE %'),('curve_rmse','曲线RMSE kW'),('curve_wape_percent','曲线WAPE %'),('daily_mape_percent','日均MAPE %'),('daily_rmse','日均RMSE kW'),('daily_wape_percent','日均WAPE %')]])+'\n'
text+='\n## 非重叠7天检查\n\n'+md_table(task(7,'nonoverlapping_calendar7day'),[('curve_mape_percent','曲线MAPE %'),('curve_rmse','曲线RMSE kW'),('curve_wape_percent','曲线WAPE %'),('daily_wape_percent','日均WAPE %')])+'\n'
text+='''
## 用户协议落实与必要边界

- 使用真实ts和outdoor字段，温度从华氏转摄氏，湿度为百分数。
- 五项各自15分钟有限均值合计为默认目标；缺任一分项则缺失，不等同直接设施总表。不另加2.67kW。
- 临时UTC解释并保留timezone_verified false；固定范围异常屏蔽、1970记录剔除、重复均值，15分钟左闭右开并标起点。
- 仅历史协变量最多前填4格；目标和历史目标不填，不后填、不双向插值。预测最多读到t减15分钟。
- 原始每格有限记录数另存，有效均值不等于15条原始分钟齐全；首尾完整目标日定边界，内部缺测留在轴上。
- 约60/20/20连续日期划分，验证至少8天、测试至少9天；预测完整标签在所属划分内，共享完整窗口并记录跳过。
- 统一历史14天1344点。LightGBM初拟合及标准化只用训练，配置按验证MAPE选；随后固定配置重拟合训练加验证。基础模型的缩放也只用当时已知训练或训练加验证，非容量值。
- HDSF及全部候选总体比较使用共同的向前验证交叉检验起报MAPE；融合拟合分数另列样本内。本轮全部十四方案、两个时距在首次TEST前冻结。
- 测试滚动更新已知历史，不重新拟合。未来只允许已知日历，不输入未来真实IT、制冷或天气。
- MAPE起报等权，阈值1e-8；排除数另存，零分母null。曲线WAPE池化绝对误差除以真值绝对和；本轮是单一设施。日均先聚合96点再评分。
- 每日一次起报；七天重叠成绩和固定日历每7天的非重叠成绩分别报告，非重叠仍非独立随机样本。
- CSV与Python DataFrame表同时输出曲线和日均MAPE、RMSE、WAPE及相对基线下降比例。24小时基线为前一天，七天基线为前三天同刻均值；负下降率代表退步。不用100减WAPE称准确率。

可选IT目标本轮未运行。基础模型预训练语料是否含NLR尚不明确，严格缺测筛选只覆盖可用日期。因此结论是单设施、已暴露时间段的按协议重复评估，不是多场站独立泛化证明。

## 保存位置与复核

本轮目录 experiment/nlr_strict_protocol_20261013，数据、模型、来源、每起报预测、日志、冻结时间与独立复核都单独保留。全指标42行及逐预测日210行以CSV提供。原始Parquet和旧报告保持原样。

时间序列验证参考 https://otexts.com/fpp3/toolbox.html#time-series-cross-validation
原始资料 https://data.nlr.gov/submissions/300
'''
report=R/'NLR严格协议重跑评估简报.md';report.write_text(text,encoding='utf-8')
for filename in ['新协议主要模型测试全指标.csv','新协议逐预测日测试指标.csv','验证时间交叉检验全指标.csv']:shutil.copyfile(F/'results'/filename,R/('NLR严格重跑_'+filename))
df=pd.DataFrame(rows);df['具体模型']=df['model'].map(LABEL);keys=['horizon_days','cohort','具体模型','curve_mape_percent','curve_rmse','curve_wape_percent','daily_mape_percent','daily_rmse','daily_wape_percent']
keys += [c for c in df.columns if c.startswith('relative_')]
python_table=df[keys].to_string(index=False,float_format=lambda x:f'{x:.6f}');(F/'results/Python_dataframe_table.txt').write_text(python_table+'\n',encoding='utf-8');shutil.copyfile(F/'results/Python_dataframe_table.txt',R/'NLR严格重跑_Python全指标表.txt')
save(F/'results/quick_report_verification.json',dict(all_passed=True,utc=utc(),markdown=str(report),markdown_sha256=sha(report),test_rows=42,python_table_rows=len(df),python_table_columns=keys,all_numbers_from_new_round=True,all_predicted_before_scoring=True,previous_test_exposure_disclosed=True))
print('New strict Markdown CSV and Python tables published',42)
