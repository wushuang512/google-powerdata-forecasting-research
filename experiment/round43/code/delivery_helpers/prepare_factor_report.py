from pathlib import Path
import sys,json
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round43';a=json.loads((R/'results/factor_comparison.json').read_text(encoding='utf-8'));v=json.loads((R/'results/independent_verification.json').read_text(encoding='utf-8'));assert len(a['rows'])==126 and len(a['unique_rows'])==98 and v['all_passed'];assert not any(s['adopt_shared_scheme'] for s in a['factor_rounds'].values())
labels={41:'Moirai2 15分钟PCA1共同因子10%\n当前方案90% 无日均协调',42:'Moirai2 三小时PCA3 最后2层及输出层5%\n当前方案95% 全域日均协调',43:'Moirai2 小时PCA1水平匹配控制5%\n当前方案95% 全域日均协调'}
for r in a['rows']+a['unique_rows']:
 for nr,label in labels.items():
  if r['model']==f'r{nr}_diagnostic_H{r["horizon_days"]}':r['report_name']=label;r['model_name']='Salesforce Moirai 2.0 R small '+label.replace('Moirai2 ','').replace('\n','；')
(R/'results/factor_comparison.json').write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
# Keep the scientific arrays and scores intact; only expand display names in the CSV.
import csv
for name in ['模型各项指标完整结果.csv','9月28日统一口径模型复算.csv','9月28日指标完整模型对照.csv']:
 p=T/name
 with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rr=list(reader)
 for row,record in zip(rr,a['rows']):row['model_name']=record['model_name'];row['report_name']=record['report_name']
 with p.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rr)
s=(B/'round40/code/report_continued_moirai.py').read_text(encoding='utf-8').replace("R=B/'round40'","R=B/'round43'").replace("Q/'continued_comparison.json'","Q/'factor_comparison.json'").replace("len(a['rows'])==114","len(a['rows'])==126").replace("qa['check_count']==140",f"qa['check_count']=={v['check_count']}").replace('114条','126条').replace('114 uniform records; 92 unique predictions; actual four rounds','126 uniform records; 98 unique predictions; actual three new rounds').replace('为92组','为98组').replace('92组不同预测','98组不同预测').replace('92组不同','98组不同')
old="p('本次继续探索Salesforce Moirai 2.0 R small，完成多分辨率输出头训练、季节变化量、相关域联合预测及真实CPU内存辅助输入四组实验。24小时曲线、7天曲线和7天日均的测试最低值都有更新，但分别来自不同配置；相对当前推荐改善约0.27%、0.46%、0.79%，没有达到大幅提升，也未找到同时稳健领跑两类曲线的统一方法。')"
new="p('本次新增共同因子预测、注意力及全参数训练、每日变化量匹配对照三轮Salesforce Moirai 2.0 R small研究，共35组结构或训练设置。6组开发训练及1组冻结最终重训均实际完成。三轮没有达到24小时和7天曲线共同升级条件，当前部署保留第32轮与第36轮。全部实测最低值仍来自此前不同研究配置，尚未达到单一方案全面、大幅领跑。')"
assert old in s;s=s.replace(old,new)
s=s.replace("keys1=['r32_selected_H1','last_source','ridge','lgb','chronos','tirex','tf25','equal']+[f'r{n}_diagnostic_H1' for n in range(37,41)]", "keys1=['r32_selected_H1','last_source','ridge','lgb','chronos','tirex','tf25','equal','r37_diagnostic_H1','r41_diagnostic_H1','r42_diagnostic_H1','r43_diagnostic_H1']")
s=s.replace("keys7=['r36_selected_H7','r31_selected_H7','mean3_source','chronos','tirex','tf25_7','tf3_uni21_7','equal']+[f'r{n}_diagnostic_H7' for n in range(37,41)]", "keys7=['r36_selected_H7','r31_selected_H7','mean3_source','chronos','tirex','tf25_7','r38_diagnostic_H7','r40_diagnostic_H7','r41_diagnostic_H7','r42_diagnostic_H7','r43_diagnostic_H7']")
start=s.index("h('四组新方向的开发证据和采用判断')");end=s.index("h('真实CPU内存对照与来源范围')",start)
block="""h('三轮共同因子与注意力训练的真实结果')
rr=[]
for nr,H in [(41,1),(41,7),(42,1),(42,7),(43,1),(43,7)]:
 r=row(H,f'r{nr}_diagnostic_H{H}');rr.append([f'{H}天 '+r['report_name'],n(r['development_wape_original_grid']),n(r['wape']),n(r['daily_wape']),n(r['peak_mape']),'未采用研究候选'])
table(['明确方法与跨度','开发曲线\\nWAPE %↓','测试曲线\\nWAPE %↓','日均\\nWAPE %↓','峰值\\nMAPE %↓','冻结采用结论'],rr,[4.40,1.00,1.00,1.00,1.00,1.96],9.6)
p('第41轮：仅在训练期拟合57域均值、标准差和共同因子，比较PCA1／3／6与15分钟／1小时／3小时共9组。第42轮：PCA3的一小时和三小时，各实际训练最后1层加输出层、最后2层加输出层及全参数，另有2个匹配原始控制。可训练参数为236万、413万或1139万；每组固定80步，最终仅重训开发冻结的三小时最后2层配置。原始权重和老师保持不变，派生权重保存。')
p('第43轮：9组每日变化量与9组同条件水平预测对照。变化量只用起报前7天差分及额外1天历史，未来变化由模型预测并从最后已知日递推。开发选择了小时PCA1水平控制，未选择每日变化量；不得将该控制结果称为差分方法改善。三轮均先用开发早段选一个共同结构，再用后段选两个跨度共享的比例及日均协调，未按测试给24小时和7天各挑赢家。')
p('三轮均未通过开发共同升级。训练模型7天测试略优于当前推荐，24小时变差；小时水平控制24小时略优、7天变差，均没有共同领跑。新的独立复核包括注意力参数实际更新、应冻结参数等值、数据与标签边界、缓存签名、预测及全部指标重放。每日差分额外历史已纳入缓存签名；已有预测和冻结选择保持不变，修复前后证据保留。')
"""
s=s[:start]+block+s[end:]
start=s.index("for nr in range(37,41):\n for H in [1,7]:",s.index("h('预测区间和最终研究结论')"));end=s.index("table(['区间方法'",start)
s=s[:start]+"""for nr,H in [(37,1),(40,7),(38,7),(41,1),(41,7),(42,1),(42,7),(43,1),(43,7)]:
 v=next(v for v in a['intervals'] if v.get('method','').startswith(f'R{nr} ') and v['horizon_days']==H)
 rr.append([f'{H}天 '+row(H,f'r{nr}_diagnostic_H{H}')['report_name'],80,f'{v["coverage_percent"]:.3f}',n(v['mean_width_normalized']),n(v['winkler80'])])
"""+s[end:]
s=s.replace("p('结论：Moirai2方向具有小幅改善线索，小时适配头、相关域及真实CPU输入各有局部优势；四轮没有得到“同一方案稳健领跑所有曲线并大幅提升”的证据。继续在同一31天上追逐测试最低值会加重选择偏差。所有训练、负面开发结果和测试预测保留，正式推荐保持不变；确认更高精度仍需要独立月份或未参与探索的同类场站数据。')", "p('结论：此前Moirai2配置有局部改善线索，新增共同因子、注意力与每日变化量三轮均未实现共同升级。不能宣称已找到同一方案稳健领跑所有曲线或大幅突破。31天数据及多轮查看过的同月TEST只支持探索；模型更大、训练损失更低均不保证预测更准。训练与负面结果保留，正式推荐不变，可靠确认仍需独立数据。')")
# Keep interval page readable: current recommendations plus the six new diagnostics.
s=s.replace("for nr,H in [(37,1),(40,7),(38,7),(41,1),(41,7),(42,1),(42,7),(43,1),(43,7)]:\n v=", "for nr,H in [(41,1),(41,7),(42,1),(42,7),(43,1),(43,7)]:\n v=")
compile(s,'report_factor_final.py','exec');(R/'code/report_factor_final.py').write_text(s,encoding='utf-8');print('Actual126 rows/98 unique; new negative training results and joint selection included in report builder')
