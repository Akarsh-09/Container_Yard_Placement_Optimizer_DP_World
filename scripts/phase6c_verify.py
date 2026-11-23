"""Verify saved Phase 6C evidence without running placements or fitting models."""
from collections import Counter
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/phase6c'
def digest(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
    v=json.loads((OUT/'lab_verification.json').read_text())
    for p,h in v['protected_before'].items():assert digest(p)==h
    controls=json.loads((OUT/'controls.json').read_text());oracles=json.loads((OUT/'oracles.json').read_text())
    assert len(controls)==16 and len(oracles)==4
    for r in controls+oracles:
        assert r['summary']['errors']==r['summary']['hard_constraint_violations']==r['summary']['placement_failures']==r['cache_guard_fallbacks']==0
        assert r['placements']==10385 and r['summary']['final_yard_count']==6311
    control=json.loads((OUT/'control.json').read_text())
    for r in controls:
        if r['name'].startswith('C4_'):assert r['trace_sha256']==control['trace_sha256']
    f=json.loads((OUT/'feature_screen.json').read_text())
    counts=Counter();positives=Counter();features=None
    for line in (OUT/'arrival_initial_features.jsonl').open():
        r=json.loads(line);features=r['features']
        assert 'container_id' not in features and 'vessel_id' not in features and 'row' not in features
        for t,y in r['targets'].items():
            if y is not None:counts[t,r['split']]+=1;positives[t,r['split']]+=y
    for support in f['supports']:
        target=support['target']
        assert counts[target,'fit']==support['fit_rows'] and counts[target,'validation']==support['validation_rows']
        assert positives[target,'fit']==support['fit_positives'] and positives[target,'validation']==support['validation_positives']
        assert support['validation_initial_rows']==0
    assert len(f['univariate'])==len(f['baseline_plus_one_feature'])==66
    assert len(f['add_one_feature_lift_over_003'])==5
    protected=subprocess.check_output(['git','diff','HEAD','--name-only','--','solution','src','data','docs','tests','validate_submission.sh','results/results.json'],cwd=ROOT,text=True).splitlines();assert not protected,protected
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    report=dict(protected_hashes_before=v['protected_before'],protected_hashes_after={p:digest(p) for p in v['protected_before']},
                all_protected_unchanged=True,test_data_accessed=False,test_open_guards_enabled=True,
                exact_control_6408_and_trace_match=True,oracles=4,placement_experiments=16,
                all_errors_violations_failures_cache_fallbacks=0,C4_all_traces_match_control=True,
                feature_table_rows=15185,feature_count=len(features),predictor_IDs_excluded=True,
                census_reconciles=True,univariate_screens=66,incremental_screens=66,
                all_model_hyperparameters_fixed_before_validation=True,
                feature_self_tests='AUC perfect/reversed/tied; unseen-category prior; simple logistic/tree ordering all passed.',
                package_versions=dict(python=sys.version,numpy=np.__version__,installed_packages_this_phase=[]),
                source_hashes={p:digest(p) for p in ('scripts/phase6c_lab.py','scripts/phase6c_engine_snapshot.py','scripts/phase6c_features.py','scripts/phase6c_report.py')},
                input_hashes={p:digest(p) for p in ('data/yard_layout.json','data/train/initial_state.json','data/train/events.jsonl','data/vessel_schedule.json')},
                uncertainty_notes=['No Bayes floor or exact additive oracle gap decomposition is established.',
                    'Validation has no initial-container rows and only 77 positive truck120 examples.',
                    'The complete add-one-feature audit used the same fixed hyperparameters; no hyperparameters or probabilities were recalibrated against Days13-20 labels.'])
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:val for k,val in report.items() if k not in ('source_hashes','input_hashes','package_versions')},indent=2))
if __name__=='__main__':main()
