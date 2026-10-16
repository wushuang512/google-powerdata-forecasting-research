"""Repair only encoded source-path strings, after matching existing raw hashes."""
from pathlib import Path
import argparse,shutil
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_user_protocol_20261013';p=F/'data/quality_report.json';x=read(p);repairs=[]
for kind,v in x['source_audit'].items():
    actual=R/'nlr-hpc-pue/raw'/Path(v['file']).name
    assert actual.is_file() and sha(actual)==v['sha256']
    if str(actual)!=v['file']:
        repairs.append(dict(source=kind,previous_path=v['file'],correct_path=str(actual),raw_sha256=v['sha256'],only_display_metadata_changed=True));v['file']=str(actual)
if repairs:
    before=F/'docs/source_path_metadata_before_repair.json'
    if not before.exists():shutil.copyfile(p,before)
    oldsha=sha(p);save(p,x);save(F/'docs/source_path_metadata_repair.json',dict(utc=utc(),old_sha256=oldsha,new_sha256=sha(p),repairs=repairs,processed_arrays_not_changed=True,original_raw_files_not_changed=True,protocol_and_selection_not_changed=True))
print('Metadata path repairs',len(repairs))
