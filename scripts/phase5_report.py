"""Aggregate the fixed test outputs; never execute a strategy or rewrite its JSON."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.scoring import SimulationStats
OUT=ROOT/'results/phase5'
def read(path):return json.loads((ROOT/path).read_text())
def enrich(r):
    r=dict(r)
    stats=SimulationStats(total_reshuffles=r['total_reshuffles'],retrieval_events=r['retrieval_events'],hard_constraint_violations=r['hard_constraint_violations'])
    r['official_rate_exact']=stats.reshuffles_per_retrieval
    r['score_exact']=stats.quantitative_score()
    return r
def main():
    official=enrich(read('results/results.json'))
    greedy=enrich(read('results/phase5/greedy.json'))
    random=[dict(enrich(read(f'results/phase5/random_{s}.json')),seed=s) for s in range(5)]
    aggregate={}
    for key,value in random[0].items():
        if key=='seed' or not isinstance(value,(int,float)):continue
        values=[r[key] for r in random]
        aggregate[key]=dict(mean=statistics.mean(values),std_population=statistics.pstdev(values),min=min(values),max=max(values))
    summary=dict(official=official,greedy=greedy,random_seeds=random,random_aggregate=aggregate,
                 official_validity_gate_passed=False,official_strategy_test_runs=1,
                 interpretation='Official errors=9861: every placement fell back to the simulator. No valid SurvivalRisk test performance claim is made.',
                 diagnostics='results/phase5/diagnostics.json')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    p=ROOT/'docs/design.md';doc=p.read_text()
    total=aggregate['total_reshuffles'];rate=aggregate['official_rate_exact'];score=aggregate['score_exact'];runtime=aggregate['elapsed_sec']
    old='| Random, seeds 0-4 mean +/- SD | [PENDING PHASE 5] | [PENDING PHASE 5] | [PENDING PHASE 5] | [PENDING PHASE 5] | [PENDING PHASE 5] |'
    new=f"| Random, seeds 0-4 mean +/- SD | {total['mean']:,.1f} +/- {total['std_population']:.1f} | {rate['mean']:.6f} +/- {rate['std_population']:.6f} | {score['mean']:.2f} +/- {score['std_population']:.0f} | {runtime['mean']:.3f}s +/- {runtime['std_population']:.3f}s | 0 / 0 / 0 |"
    assert old in doc;doc=doc.replace(old,new)
    for label,r in [('Greedy',greedy),('Frozen MyStrategy',official)]:
        old=f'| {label} | [PENDING PHASE 5] | [PENDING PHASE 5] | [PENDING PHASE 5] | [PENDING PHASE 5] | [PENDING PHASE 5] |'
        new=f"| {label} | {r['total_reshuffles']:,} | {r['official_rate_exact']:.6f} | {r['score_exact']:.2f} | {r['elapsed_sec']:.2f}s | {r['errors']:,} / {r['hard_constraint_violations']} / {r['placement_failures']} |"
        assert old in doc;doc=doc.replace(old,new)
    doc=doc.replace('All runs have zero errors, hard violations, and placement failures.','All training runs have zero errors, hard violations, and placement failures.')
    old='After Phase 5 approval, run the frozen strategy once via `python -m src.run --strategy solution.my_strategy.MyStrategy --data-dir data/test -o results/results.json -v`, then unchanged Greedy and seeded Random baselines. Exact commands and the pre-test source manifest are in `results/phase4/phase5_plan.txt`. No strategy or parameter changes follow test opening; only result reporting is completed.'
    new='''**The official test run failed the error gate.** Its 9,861 exceptions cover every arrival; reported reshuffles and score reflect simulator fallback placements, not successful SurvivalRisk decisions. The test snapshot has 313 sparse stacks with 352 empty tier slots. The frozen cache assumed every tier through stack height held a container; the public API returns an empty ID for padded gaps and `get_container_info("")` returns None, causing the logged exception. Thus training gains did not produce a valid test submission, and this run cannot establish survival-model generalization. Strategy and parameter hashes matched before and after; there was exactly one official test replay, with no repair, retuning, or second instrumented pass. Raw output remains unchanged in `results/results.json`; fixed-seed baselines and read-only diagnostics are in `results/phase5/`.'''
    assert old in doc;doc=doc.replace(old,new)
    doc=doc.replace('Initial-origin truck deliveries still cost','The sparse-input failure exposes a missing production robustness test. Future work must distinguish physical stack height from occupied container entries and validate cache counts accordingly. No such fix was made after test opening. Initial-origin truck deliveries in training still cost')
    assert '[PENDING PHASE 5]' not in doc
    p.write_text(doc)
    print('Random aggregate:')
    for k in ('total_reshuffles','official_rate_exact','score_exact','elapsed_sec'):print(k,aggregate[k])
    print('Greedy exact rate/score',greedy['official_rate_exact'],greedy['score_exact'])
    print('Official exact rate/score',official['official_rate_exact'],official['score_exact'])
if __name__=='__main__':main()
