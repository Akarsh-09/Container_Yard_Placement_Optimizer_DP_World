"""Seal the defensive-only change after train proof, before the test replay."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase5b'
def digest(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
    old=json.loads((ROOT/'results/phase4/frozen_manifest.json').read_text())
    params='solution/survival_params.py'
    assert digest(params)==old['files'][params]
    previous=subprocess.check_output(['git','show','HEAD:solution/my_strategy.py'],cwd=ROOT,text=True)
    assert hashlib.sha256(previous.encode()).hexdigest()==old['files']['solution/my_strategy.py']
    current=(ROOT/'solution/my_strategy.py').read_text()
    before=ast.parse(previous);after=ast.parse(current)
    def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    a,b=functions(before),functions(after)
    changed=sorted(k for k in a if a[k]!=b[k])
    allowed=['_finish_placement','_place_survival','_refresh_indices','_reset_caches','_survival_key','on_container_retrieved']
    assert changed==sorted(allowed),changed
    assert a['survival_block_probability']==b['survival_block_probability']
    normalized=current.replace('self._arrival_times.get(cid, self.current_time)','self._arrival_times[cid]')
    assert a['_survival_key']==functions(ast.parse(normalized))['_survival_key']
    assert a['_place_survival']==functions(ast.parse(current.replace('height = self._heights[i]','height = len(self._contents[i])')))['_place_survival']
    equivalence=json.loads((OUT/'equivalence.json').read_text())
    assert equivalence['identical_placements']==10385 and equivalence['phase3d_hash_match']
    for name in ('cached','live'):
        r=json.loads((OUT/(name+'.json')).read_text())
        assert r['summary']['total_reshuffles']==6408 and r['cache_guard_fallbacks']==0
        assert r['summary']['errors']==r['summary']['hard_constraint_violations']==r['summary']['placement_failures']==0
    assert '88 passed' in (OUT/'pytest_pre.log').read_text()
    failed_hash=json.loads((ROOT/'results/phase5/official_output_hash.json').read_text())['sha256']
    assert digest('results/results.json')==failed_hash
    archive=ROOT/'results/phase5/failed_initial_run.json'
    if archive.exists():assert hashlib.sha256(archive.read_bytes()).hexdigest()==failed_hash
    else:shutil.copyfile(ROOT/'results/results.json',archive)
    files={p:digest(p) for p in old['files']}
    baselines=['results/phase5/greedy.json']+[f'results/phase5/random_{s}.json' for s in range(5)]
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
                  baseline_files={p:digest(p) for p in baselines},failed_initial_run_sha256=failed_hash,
                  survival_parameters_unchanged=True,survival_math_AST_unchanged=True,
                  cohorts_unchanged_except_defensive_arrival_time_fallback=True,changed_methods=changed,
                  train_placement_sha256=equivalence['trace_sha256'],train_placements=10385,train_reshuffles=6408,
                  unit_tests_passed=88,authorised_change='Physical stack heights separated from real IDs; empty-slot filtering and defensive cache/metadata handling only.',
                  phase5b_test_runs_before_manifest=0)
    (OUT/'phase5b_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
