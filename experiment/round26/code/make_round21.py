from pathlib import Path
import sys,ast
T=Path(sys.argv[1]);S=Path(__file__).parent;R=T/'experiment/round21';(R/'code').mkdir(parents=True,exist_ok=True)
source=(S/'learned_peak_experiment.py').read_text(encoding='utf-8').replace("R=B/'round20'","R=B/'round21'").replace("round20/models/interval7_radius.npz","round21/models/interval7_radius.npz")
a=source.index("CONFIGS=[");b=source.index("\nALPHAS=",a);source=source[:a]+"CONFIGS=[dict(id='prior',strength=0.,gate=True,location='prior',support=8)]+[dict(id=f'localpeak_top{n}_s{int(s*100)}_{loc}',strength=s,gate=True,location=loc,support=n) for n in [8,16,32] for s in [.25,.5] for loc in ['prior','near_history']]"+source[b:]
a=source.index('def candidate(');b=source.index('\ndef prepare():',a);source=source[:a]+(S/'localized_candidate_body.txt').read_text(encoding='utf-8')+source[b:]
source=source.replace("Native TimesFM3 learned daily-peak head transferred conservatively to current base, with sum-preserving bounded daily projection and optional near-history peak location.","Native TimesFM3 learned peak head with sign-agreement gating and sparse correction of top8/16/32 forecast points; mean-preserving bounded projection and optional near-history peak location.")
source=source.replace("Target peak correction capped at1% observed7day mean then strengths.1/.25/.5. Preserve full96 mean and[0,1], optional peak index nearest historical last-day max among forecast points within0.3%historical mean of current forecast max and8 slots of original argmax. No future truth or validity masks.","Target peak correction capped at1%observed7day mean then strengths.25/.5. Sign-agreement gate mandatory. Only stable-ranked top8/16/32 points plus pin may change; outside points exactly fixed. Preserve full96 mean and[0,1]; requested peak is clipped to feasible active-mean and inactive-max floor. Optional peak location within0.3%historical mean and8slots. No future truth/mask. R20 DEV showed P95 and threshold tradeoffs; sparse support is prospectively motivated by DEV. TEST was repeatedly seen and no independence restored.")
# Bind prior negative evidence and the source implementation.
source=source.replace("files=['results/data.npz'","files=['round20/code/learned_peak_experiment.py','round20/results/dev_summary.json','round20/results/selection.json','results/data.npz'")
ast.parse(source)
p=R/'code/localized_peak_experiment.py'
if (R/'docs/protocol.json').exists():assert p.read_text(encoding='utf-8')==source
p.write_text(source,encoding='utf-8');(R/'code/source_metrics.py').write_bytes((S/'source_metrics.py').read_bytes())
g=(S/'run_joint_local_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round16'","R=T/'experiment/round21'").replace("['develop','test']","['develop','test','deploy']").replace('code/joint_local_experiment.py','code/localized_peak_experiment.py')
(R/'code/run_localized_peak_guarded.py').write_text(g,encoding='utf-8');print('Round21 prepared with unchanged strict adoption guards')

