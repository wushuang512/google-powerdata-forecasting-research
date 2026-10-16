from pathlib import Path
import sys,json,hashlib,subprocess
T=Path(sys.argv[1]).resolve();R=T/'experiment/round44';Q=R/'results'
if '--resume' in sys.argv:
 raise SystemExit(subprocess.call([str(T/'runtime/python/python.exe'),'-u','-X','utf8',str(R/'code/run_hdsf_guarded.py'),str(T)]))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
s=read(Q/'summary.json');assert read(Q/'science_verification.json')['all_passed'];assert hashlib.sha256((Q/'selection.json').read_bytes()).hexdigest()==(Q/'selection.sha256').read_text()
print('HDSF半日单纯形融合：固定D14/D22训练，使用9.28统一评价。')
for H in [1,7]:
 x=s['scores'][str(H)]['hdsf'];print(f'{H}天：曲线WAPE {x["wape"]:.6f}%；每日均值WAPE {x["daily_wape"]:.6f}%；峰值MAPE {x["peak_mape"]:.6f}%')
print('报告：',T/'HDSF半日单纯形融合复现与统一评价汇报.docx');print('完整结果：',Q/'HDSF与此前模型全部指标比较.csv');print('这是历史测试复现，缺少独立确认月份。GPU验证另存，不改变CPU完整复现冻结结果。')
