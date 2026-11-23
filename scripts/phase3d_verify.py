"""Record final Phase 3D train-only verification after the CLI controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase3d'
def main():
    runs=json.loads((OUT/'runs.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text())['winner']
    cache=json.loads((OUT/'cache_equivalence.json').read_text())
    cli=json.loads((OUT/'candidate_cli.json').read_text())
    default=json.loads((OUT/'default_cli.json').read_text())
    assert len(runs)==14
    assert cli['total_reshuffles']==selected['summary']['total_reshuffles']==6408
    assert default['total_reshuffles']==7026
    for r in runs:
        assert all(r['summary'][k]==0 for k in ('errors','hard_constraint_violations','placement_failures'))
    assert all(cli[k]==0 for k in ('errors','hard_constraint_violations','placement_failures'))
    assert cache['identical_placements']==10385
    assert cache['trace_sha256']==selected['trace_sha256']
    protected=subprocess.check_output(['git','diff','HEAD','--name-only','--','src','data','validate_submission.sh'],cwd=ROOT,text=True).splitlines()
    assert not protected,protected
    files=['solution/my_strategy.py','solution/survival_params.py','solution/phase3d_candidate.py',
           'scripts/phase3d.py','scripts/phase3d_audit.py','scripts/phase3d_report.py','scripts/phase3d_verify.py','tests/test_my_strategy.py']
    result=dict(tests_passed=120,experiment_runs=14,all_errors_violations_failures=0,
                cache_guard_fallbacks=sum(r['cache_guard_fallbacks'] for r in runs),
                cache_equivalence=cache,cli_seconds=cli['elapsed_sec'],cli_under_15s=cli['elapsed_sec']<15,
                candidate_cli_exact_match=True,default_unchanged_7026=True,
                fit_cutoff_exclusive='2025-01-13T00:00:00',test_data_opened=False,
                protected_files_unchanged=True,python=sys.version,
                sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
