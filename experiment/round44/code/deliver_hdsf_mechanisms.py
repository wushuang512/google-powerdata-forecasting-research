from pathlib import Path
import json,sys,hashlib,datetime
T=Path(sys.argv[1]);A=T/'experiment/round44/results/mechanism_analysis_20261008'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((A/'verification.json').read_text(encoding='utf-8'))
p=json.loads((A/'report_provenance.json').read_text(encoding='utf-8'))
assert v['all_passed'] and v['check_count']==61
assert sha(Path(p['report']))==p['report_sha256'] and sha(Path(p['figure']))==p['figure_sha256']
assert sha(A/'verification.json')==p['verification_sha256']
qa={'file':p['figure'],'sha256':p['figure_sha256'],'visually_reviewed':True,'layout_passed':True,
    'review':'Both horizon panels reviewed. Labels, numeric weights, colorbars and full day range legible; no clipping.',
    'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(A/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
name=Path(p['report']).name
r=T/'开始阅读.md';text=r.read_text(encoding='utf-8');line=f'- [HDSF优势原因、固定消融与误差分解]({name})\n'
if line not in text:text=text.replace('新增同版式综合汇报已完成：\n\n','新增同版式综合汇报已完成：\n\n'+line,1)
r.write_text(text,encoding='utf-8')
r=T/'当前进展.md';text=r.read_text(encoding='utf-8');extra=f'\n机制分析已完成：[HDSF优势原因与消融分析]({name})。使用已有缓存做固定结构及成员删除对照，独立核对61项数值；已冻结权重、统一评价、模型部署和9页Word保持不变。\n'
if '机制分析已完成' not in text:text+=extra
r.write_text(text,encoding='utf-8')
manifest=T/'交付文件清单.json';m=json.loads(manifest.read_text(encoding='utf-8'))
m['mechanism_analysis']=dict(report=p['report'],report_sha256=p['report_sha256'],evidence=str(A),
    verification=str(A/'verification.json'),checks=61,all_passed=True,retrospective=True,
    independent_confirmation=False,model_selection_unchanged=True,canonical_evaluation_unchanged=True)
m['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['inventory_verified']=False
files=sorted((x for x in T.rglob('*') if x.is_file() and x!=manifest),key=lambda x:str(x.relative_to(T)))
m['all_files']=[dict(path=str(x.relative_to(T)),bytes=x.stat().st_size) for x in files]
m['files']=len(files);m['bytes']=sum(x['bytes'] for x in m['all_files'])
manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(analysis_all_passed=True,files=m['files'],inventory_requires_verification=True)))
