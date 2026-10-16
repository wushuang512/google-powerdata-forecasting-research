from pathlib import Path
import sys,ast
T=Path(sys.argv[1]);S=Path(__file__).parent;R=T/'experiment/round22';(R/'code').mkdir(parents=True,exist_ok=True)
for name in ['point_residual_experiment.py','source_metrics.py']:
 v=(S/name).read_text(encoding='utf-8');ast.parse(v);p=R/'code'/name
 if (R/'docs/protocol.json').exists():assert p.read_text(encoding='utf-8')==v
 p.write_text(v,encoding='utf-8')
g=(S/'run_cache_bounded_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round21'","R=T/'experiment/round22'").replace("['develop','test','deploy']","['develop','test']").replace('code/localized_peak_experiment.py','code/point_residual_experiment.py')
(R/'code/run_point_residual_guarded.py').write_text(g,encoding='utf-8');print('Round22 compact residual learning ready with512MiB hard cap')
