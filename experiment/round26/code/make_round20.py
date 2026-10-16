from pathlib import Path
import sys,ast
T=Path(sys.argv[1]);S=Path(__file__).parent;R=T/'experiment/round20'
for name in ['learned_peak_experiment.py','source_metrics.py']:ast.parse((S/name).read_text(encoding='utf-8'))
(R/'code').mkdir(parents=True,exist_ok=True)
for name in ['learned_peak_experiment.py','source_metrics.py']:
 p=R/'code'/name;v=(S/name).read_bytes()
 if (R/'docs/protocol.json').exists():assert p.read_bytes()==v,'Cannot mutate frozen experiment'
 p.write_bytes(v)
g=(S/'run_joint_local_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round16'","R=T/'experiment/round20'").replace("['develop','test']","['develop','test','deploy']").replace('code/joint_local_experiment.py','code/learned_peak_experiment.py')
(R/'code/run_learned_peak_guarded.py').write_text(g,encoding='utf-8');print('Round20 prepared')
