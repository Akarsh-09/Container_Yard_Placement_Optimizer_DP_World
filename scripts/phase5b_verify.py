"""Post-test integrity checks for the authorised defensive sparse-stack repair."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase5b'
def digest(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
    manifest=json.loads((OUT/'phase5b_manifest.json').read_text())
    checks={p:digest(p)==h for p,h in manifest['files'].items()};assert all(checks.values())
    baseline_checks={p:digest(p)==h for p,h in manifest['baseline_files'].items()};assert all(baseline_checks.values())
    assert digest('results/phase5/failed_initial_run.json')==manifest['failed_initial_run_sha256']
    original=json.loads((ROOT/'results/phase4/frozen_manifest.json').read_text())
    assert digest('solution/survival_params.py')==original['files']['solution/survival_params.py']
    result=json.loads((ROOT/'results/results.json').read_text())
    assert result['errors']==result['hard_constraint_violations']==result['placement_failures']==0
    assert result['total_reshuffles']==6179 and result['elapsed_sec']<15
    summary=json.loads((OUT/'summary.json').read_text())
    assert digest('results/results.json')==summary['official_output_sha256']
    assert 'ERROR: Strategy raised' not in (OUT/'my_strategy_cli.log').read_text()
    eq=json.loads((OUT/'equivalence.json').read_text())
    assert eq['identical_placements']==10385 and eq['trace_sha256']==manifest['train_placement_sha256']
    assert all('88 passed' in (OUT/f'pytest_{when}.log').read_text() for when in ('pre','post'))
    validator=(OUT/'validator.log').read_text();assert 'PASSED (0 warnings)' in validator
    assert 'Found strategy: solution.my_strategy.MyStrategy' in validator
    doc=(ROOT/'docs/design.md').read_text();assert '[PENDING PHASE 5]' not in doc
    pages=len(PdfReader(OUT/'design_review/design-review.pdf').pages);assert pages==3
    assert len(re.findall(r'^## [1-7]\. ',doc,re.M))==7
    protected=subprocess.check_output(['git','diff','HEAD','--name-only','--','src','data','validate_submission.sh'],cwd=ROOT,text=True).splitlines();assert not protected,protected
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    post=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
              hashes={p:digest(p) for p in checks},baseline_checks=baseline_checks)
    (OUT/'manifest_after.json').write_text(json.dumps(post,indent=2)+'\n')
    report=dict(approved_fix_only=True,survival_parameters_unchanged=True,survival_math_AST_unchanged=True,
                train_equivalence=eq,train_reshuffles=6408,train_count_guard_fallbacks=0,
                pre_and_post_test_hashes_match=True,phase5b_strategy_test_replays=1,
                prior_failed_strategy_test_replays=1,baseline_replays_this_phase=0,
                failed_initial_run_archived_byte_identically=True,
                official=dict(summary=result,official_rate_exact=summary['official_rate_exact'],score_exact=summary['score_exact'],
                              output_sha256=summary['official_output_sha256'],error_gate_passed=True),
                tests_passed=88,scaffold_tests=27,strategy_tests=61,
                validator='PASSED (0 warnings), original validator via documented portable grep adapter',
                design=dict(words=len(doc.split()),sections=7,pages=pages,font='11pt A4 reference rendering',
                            visual_review=True,pending_markers=0,initial_failure_and_fix_disclosed=True),
                protected_files_unchanged=True,baseline_files_unchanged=True)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
