"""Before any model selection, tighten chronological separation, preserving v1."""
from pathlib import Path
import datetime,hashlib,json,sys
R=Path(sys.argv[1])/'experiment/round46';p=R/'docs/protocol.json'
assert not (R/'results/selection.json').exists()
assert not list((R/'results').glob('*_test_complete.json'))
raw=p.read_bytes();j=json.loads(raw.decode('utf-8'))
archive=R/'docs/protocol_registered_v1.json';assert not archive.exists();archive.write_bytes(raw)
j['selection_amendment']={
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_protocol_sha256':hashlib.sha256(raw).hexdigest(),
 'before_any_family_or_fusion_selection':True,'TEST_not_run':True,'training_recipe_unchanged':True,
 'reason':'Avoid familyalpha selected on lateDEV labels then retrospectively scoring it in earlier forwardfusionfolds. Familyconfiguration andalpha now use only targetdates≤D18; fusionvalidation usesD19/D21 for24h andD19–D22 for7d. This reduces multi-level validation reuse; stillveryfewdates andexposedhistoricalTEST.'}
j['family_selection']='EarlyDEV H1 targetD15/D17 andH7 target≤D18 choose head/tail,40/80steps andalpha0/.25/.5/1 bysame perPDUmetrics andpaireddateoneSE. All learnedfamilyhyperparameters labelend≤D18. No newTEST score consulted.'
j['fusion_selection']='Fitweights on precedingDEV targetdates; validateonlyD19/D21 for24h andD19–D22 for7d, afterfamilyparameterlabelendD18. Originalfrozen reference and3registeredlayouts, alpha0/.25/.5/1; .005pp andpairedoneSEgain, positivefoldmajority, fullDEVcurve/daily nonworse,peakregression≤.1pp. OriginalTEST is historicallyexposed, noindependenceclaim.'
j['fusion_script_sha256']=hashlib.sha256((R/'code/hdsf_foundation_fusion.py').read_bytes()).hexdigest()
p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'amendment_recorded_before_selection':True,'training_recipe_unchanged':True,'family_label_end':18,'fusion_validation_after':18}))
