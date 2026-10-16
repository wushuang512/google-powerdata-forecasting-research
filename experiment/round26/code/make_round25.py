from pathlib import Path
import sys,ast
T=Path(sys.argv[1]);S=Path(__file__).parent;R=T/'experiment/round25';(R/'code').mkdir(parents=True,exist_ok=True)
for name in ['level_shape_experiment.py','source_metrics.py']:
 v=(S/name).read_text(encoding='utf-8');ast.parse(v);p=R/'code'/name
 if (R/'docs/protocol.json').exists():assert p.read_text(encoding='utf-8')==v
 p.write_text(v,encoding='utf-8')
g=(S/'run_cache_bounded_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round21'","R=T/'experiment/round25'").replace("['develop','test','deploy']","['develop','test']").replace('code/localized_peak_experiment.py','code/level_shape_experiment.py')
(R/'code/run_level_shape_guarded.py').write_text(g,encoding='utf-8');print('R25 daily-level and intraday-shape trial ready')
