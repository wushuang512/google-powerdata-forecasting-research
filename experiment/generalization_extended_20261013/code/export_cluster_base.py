from pathlib import Path
import argparse,runpy,sys,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--phase',choices=['dev','test'],required=True);a=ap.parse_args();root=Path(a.root);folder=root/'experiment/generalization_extended_20261013'
sys.argv=[str(folder/'code/portfolio_stage.py'),str(root),'Google2019','--stage','freeze']
env=runpy.run_path(str(folder/'code/portfolio_stage.py'),run_name='portfolio_import');s=env['read'](folder/'results/Google2019_selection.json');out={}
for H in [1,7]:
    task=s['tasks'][str(H)];name=task['selected_method']
    if a.phase=='test':
        with np.load(folder/'results/Google2019_test_predictions.npz') as z:p=z[f'{name}_H{H}'].copy()
    else:
        bank=env['load_bank']('dev',H,task['checkpoints']);p=env['make_methods'](bank,env['origins']('dev',H),H,task['configurations'])[name]
    out[f'selected_H{H}']=p
env['np_save'](folder/'results'/f'Google2019_{a.phase}_portfolio_for_cluster.npz',**out)
print('Frozen portfolio base exported',a.phase,flush=True)
