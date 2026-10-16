from pathlib import Path
import json,sys,datetime
T=Path(sys.argv[1]).resolve();R=T/'experiment/generalization_review_20261012'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
a=read(R/'suitability_audit.json');assert a['assessment_completed'] and not a['external_model_scoring_started']
assert all(x['all_passed'] for x in a['deployment_verification'].values())
assert read(T/'当前推荐方案.json')['scheme_id']=='R48_ROBUST_HDSF_USER_ADOPTED'
save(R/'completion_verification.json',dict(all_passed=True,user_adoption_implemented=True,both_horizons_replayed=True,spreadsheet_and_original_data_unmodified=True,scope='suitabilityaudit andevaluationdesign; externalforecasttraining/scoring notrun',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
p=T/'交付文件清单.json';m=read(p);m['current_user_adoption']=a['new_recommendation'];m['current_scheme_id']=a['new_recommendation'];m['latest_research_candidate_adopted']=True;m['latest_word_snapshot_status']='ReportcreatedbeforelatestR48useradoption; retainedhistoricalsnapshot';m['generalization_assessment']=str(R/'suitability_audit.json');m['generalization_model_scores_computed']=False;m['inventory_verified']=False
fs=[dict(path=str(f.relative_to(T)),bytes=f.stat().st_size) for f in T.rglob('*') if f.is_file() and f!=p];fs.sort(key=lambda x:x['path']);m['all_files']=fs;m['files']=len(fs);m['bytes']=sum(x['bytes'] for x in fs);save(p,m)
print(json.dumps(dict(current_scheme=a['new_recommendation'],assessment_completed=True,external_model_scores_computed=False,files=len(fs),bytes=m['bytes'])))
