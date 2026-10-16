"""Verify final evidence and refresh the complete inventory after automation stop."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
top = root.parent
b = top / 'experiment'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

summary = read(b / 'round9/results/summary.json')
source = read(root / 'results/complete_projection_audit.json')
qa = read(b / 'round9/docs/report_qa/verification.json')
automation = read(root / 'results/continuation_automation.json')
report = top / 'Google PowerData 2019 完整算力特征与微调研究汇报.docx'
checks = dict(source_all_passed=source['all_passed'], source_files12085=source['source_files']==12085,
              models_all_passed=summary['all_passed'], word_all_pages_reviewed=qa['all_pages_visually_reviewed'],
              word12pages=qa['pages']==12, word_hash=sha(report)==qa['document_sha256'],
              automation_paused=automation['status']=='PAUSED',
              frozen_selection_hash=sha(b/'round9/results/selection.json')==(b/'round9/results/selection.sha256').read_text().strip(),
              no_large_improvement_claim=summary['large_improvement'] is False,
              independent_confirmation_limit=summary['independent_confirmation'] is False)
for name in ['selected', 'fusion_candidate']:
    v = read(b/f'round9/results/{name}_deployment_verification.json')
    checks[name+'_deployment'] = all(v[k] for k in ['rows5472','power_finite_bounds','cpu_missing_outside_a','cpu_finite_nonnegative_in_a','interval_order','selection_hash'])
checks['desktop_entries'] = all((top/f).is_file() for f in ['01_研究用新组合_预测次日.cmd','09_现有最佳微调_预测次日.cmd','10_完整算力融合候选_预测次日.cmd','11_权重微调对照_预测次日.cmd','开始阅读.md','清理说明.md','资源占用说明.md'])
checks['progress_completed'] = '研究完成' in (root/'当前进展.md').read_text(encoding='utf-8') and '交付已完成' in (top/'当前进展.md').read_text(encoding='utf-8')
assert all(checks.values()), checks
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
evidence = read(root/'results/final_delivery_evidence.json')
evidence.update(automation_pause_pending=False, automation_status='PAUSED', automation_paused_utc=now)
write(root/'results/final_delivery_evidence.json', evidence)
audit_path = root/'results/final_delivery_verification.json'
write(audit_path, dict(verified_utc=now, checks=checks, all_passed=True,
      inventory_checked_against_actual_files=True, source_scope='Complete official cell-a selected usage columns; not all eight cells or all fields.',
      same_month_test_repeatedly_viewed=True, frozen_wape=summary['scores']['selected']['all57']['wape']))
inventory_path = top/'交付文件清单.json'
files = [dict(file=str(p.relative_to(top)), bytes=p.stat().st_size) for p in top.rglob('*') if p.is_file() and p != inventory_path]
files.sort(key=lambda x: x['file'])
manifest = dict(root=str(top), generated_at=now, files=files, count=len(files), bytes=sum(v['bytes'] for v in files),
       complete_cell_a_projected_usage=True, complete_all_eight_cell_usage=False,
       latest_frozen_wape=summary['scores']['selected']['all57']['wape'],
       final_report_pages=12, final_report_visually_verified=True, automation_status='PAUSED')
write(inventory_path, manifest)
listed = {v['file']:v['bytes'] for v in read(inventory_path)['files']}
actual = {str(p.relative_to(top)):p.stat().st_size for p in top.rglob('*') if p.is_file() and p != inventory_path}
assert listed == actual, 'Inventory differs from actual file set or byte sizes'
assert len(listed)==manifest['count'] and sum(actual.values())==manifest['bytes']
print(json.dumps(dict(all_passed=True, inventory_files=len(actual), bytes=manifest['bytes'],
        report_pages=12, report_bytes=report.stat().st_size, automation_status='PAUSED',
        frozen_wape=summary['scores']['selected']['all57']['wape']), ensure_ascii=False))
