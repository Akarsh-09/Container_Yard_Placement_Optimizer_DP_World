"""Final integrity/report checks. Does not run MyStrategy on any dataset."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase5'
def main():
    manifest=json.loads((ROOT/'results/phase4/frozen_manifest.json').read_text())
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in manifest['files']}
    checks={p:hashes[p]==h for p,h in manifest['files'].items()}
    assert all(checks.values())
    post=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,hashes=hashes)
    (OUT/'manifest_after.json').write_text(json.dumps(post,indent=2)+'\n')
    result_path=ROOT/'results/results.json'
    original=json.loads((OUT/'official_output_hash.json').read_text())
    output_sha=hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert output_sha==original['sha256']
    result=json.loads(result_path.read_text())
    assert result['errors']==result['placement_events']==9861
    assert result['hard_constraint_violations']==result['placement_failures']==0
    assert result['total_reshuffles']==10055
    error_lines=sum('ERROR: Strategy raised' in s for s in (OUT/'my_strategy_cli.log').read_text().splitlines())
    assert error_lines==9861
    for name in ['greedy']+[f'random_{s}' for s in range(5)]:
        r=json.loads((OUT/(name+'.json')).read_text())
        assert r['errors']==r['hard_constraint_violations']==r['placement_failures']==0
    tests=(OUT/'pytest.log').read_text();assert '78 passed' in tests
    validator=(OUT/'validator.log').read_text();assert 'PASSED (0 warnings)' in validator
    assert 'Found strategy: solution.my_strategy.MyStrategy' in validator
    design=(ROOT/'docs/design.md').read_text();assert '[PENDING PHASE 5]' not in design
    pages=len(PdfReader(OUT/'design_review/design-review.pdf').pages);assert pages==3
    protected=subprocess.check_output(['git','diff','HEAD','--name-only','--','solution','src','data','tests','validate_submission.sh'],cwd=ROOT,text=True).splitlines()
    assert not protected,protected
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    report=dict(official_strategy_test_runs=1,official_result_byte_preserved=True,official_output_sha256=output_sha,
                manifest_before_all_match=True,manifest_after_all_match=True,hashes=hashes,
                official=dict(errors=result['errors'],hard_constraint_violations=result['hard_constraint_violations'],
                              placement_failures=result['placement_failures'],logged_exceptions=error_lines,
                              error_gate_passed=False),
                baseline_runs=dict(greedy=1,random_seeds=[0,1,2,3,4],errors_violations_failures=0),
                pytest_passed=78,validator='PASSED (0 warnings), via unchanged validator with documented Perl grep adapter',
                validator_scope='Train-only first-500-event smoke test; its pass does not negate the official test exceptions.',
                design=dict(words=len(design.split()),sections=len(re.findall(r'^## [1-7]\. ',design,re.M)),
                            reference_pages=pages,visual_review=True,pending_markers=0),
                protected_files_unchanged=True,strategy_or_parameter_changes_after_test=False,
                submission_success=False,
                reason='Frozen strategy fails on sparse initial stack tiers; all test arrivals use simulator fallback. No repair or rerun performed.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
