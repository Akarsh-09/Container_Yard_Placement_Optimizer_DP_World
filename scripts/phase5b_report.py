"""Report the single defensive-fix evaluation without replaying any strategy."""
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.scoring import SimulationStats
OUT=ROOT/'results/phase5b'
def main():
    path=ROOT/'results/results.json';r=json.loads(path.read_text())
    assert r['errors']==r['hard_constraint_violations']==r['placement_failures']==0
    assert r['total_reshuffles']==6179
    s=SimulationStats(total_reshuffles=r['total_reshuffles'],retrieval_events=r['retrieval_events'])
    rate=s.reshuffles_per_retrieval;score=s.quantitative_score()
    old=json.loads((ROOT/'results/phase5/summary.json').read_text())
    greedy=old['greedy'];random=old['random_aggregate']
    saving=(greedy['total_reshuffles']-r['total_reshuffles'])/greedy['total_reshuffles']*100
    summary=dict(official=r,official_rate_exact=rate,score_exact=score,
                 greedy=greedy,random_aggregate=random,random_seeds=old['random_seeds'],
                 test_reshuffles_saved_vs_greedy=greedy['total_reshuffles']-r['total_reshuffles'],
                 test_percent_saved_vs_greedy=saving,baseline_replays_in_phase5b=0,
                 phase5b_strategy_test_runs=1,failed_run_archive='results/phase5/failed_initial_run.json',
                 official_output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    docpath=ROOT/'docs/design.md';doc=docpath.read_text()
    row=f"| Frozen MyStrategy | {r['total_reshuffles']:,} | {rate:.6f} | {score:.2f} | {r['elapsed_sec']:.2f}s | 0 / 0 / 0 |"
    doc,n=re.subn(r'^\| Frozen MyStrategy \| 10,055 .*$',row,doc,flags=re.M);assert n==1
    a=doc.index('**The official test run failed the error gate.**');b=doc.index('\n\n## 7.',a)
    disclosure='''The initial frozen test run failed with 9,861 exceptions from unhandled `""`-padded sparse tiers in `YardState`, absent from the training initial yard; its raw output is archived at `results/phase5/failed_initial_run.json`. A strictly defensive empty-slot and physical-height fix was applied with no algorithmic, scoring, cohort, or parameter changes, verified by the unchanged `survival_params.py` SHA-256 and identical 10,385-decision training trace SHA-256 `74a6945f533e5cbe55d16fecf93d09fdb0b886da151bd06bfa4fad464fb2f414`. The reported `results/results.json` is the single valid test execution afterward; unchanged baseline outputs were reused, with no additional instrumented test replay or post-fix tuning.'''
    interpretation=f'''On the valid test run, the policy saves {greedy['total_reshuffles']-r['total_reshuffles']:,} reshuffles ({saving:.2f}%) versus Greedy, compared with 20.26% on train. Its official rate moves from .627804 on train to {rate:.6f} on test. This supports a retained advantage on this dataset, while the disclosed post-exposure API repair means the evaluation is not an untouched first-attempt success; it does not establish broader generalization.'''
    doc=doc[:a]+disclosure+'\n\n'+interpretation+doc[b:]
    oldlim='The sparse-input failure exposes a missing production robustness test. Future work must distinguish physical stack height from occupied container entries and validate cache counts accordingly. No such fix was made after test opening.'
    assert oldlim in doc
    doc=doc.replace(oldlim,'Sparse-input handling now separates physical heights from real container IDs and is covered by targeted tests; the initial failure remains disclosed.')
    doc=doc.replace('Metadata is deleted on successful retrieval.','Metadata is deleted on retrieval callbacks and rehydrated if live refresh finds the target still present.')
    doc=doc.replace('A total-count mismatch triggers a full public-API refresh.','Physical heights include padded gaps; cached counts include only real IDs. A total-count mismatch triggers a full public-API refresh.')
    assert '[PENDING PHASE 5]' not in doc
    docpath.write_text(doc)
    # Reuse the development renderer with a separate destination; leave tracked
    # Phase 4/5 pagination artifacts and their source utility unchanged.
    renderer=ROOT/'scripts/phase4_render_design.py'
    source=renderer.read_text().replace("OUT=ROOT/'tmp/pdfs';OUT.mkdir(parents=True,exist_ok=True)","OUT=ROOT/'results/phase5b/design_review';OUT.mkdir(parents=True,exist_ok=True)")
    exec(compile(source,str(renderer),'exec'),{'__file__':str(renderer),'__name__':'__main__'})
    print('Official rate',rate,'score',score,'saved vs Greedy',saving)
if __name__=='__main__':main()
