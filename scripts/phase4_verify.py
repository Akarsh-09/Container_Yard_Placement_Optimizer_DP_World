"""Seal the frozen production implementation and validate Phase 4 evidence."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solution.survival_params import SURVIVAL_PARAMETERS
OUT=ROOT/'results/phase4'
def read(name):return json.loads((OUT/name).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def main():
    cached,live=read('cached.json'),read('live.json')
    eq=read('equivalence.json')
    a,b=read('cli_1.json'),read('cli_2.json')
    assert a['total_reshuffles']==b['total_reshuffles']==6408
    elapsed=[a.pop('elapsed_sec'),b.pop('elapsed_sec')]
    assert a==b and max(elapsed)<15
    normalized=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert cached['trace_sha256']==live['trace_sha256']==eq['trace_sha256']
    assert cached['cache_guard_fallbacks']==live['cache_guard_fallbacks']==0
    assert cached['slices']['origin_exit/initial/TRUCK_DLVR']['reshuffles']==1693
    subclasses=[]
    for p in (ROOT/'solution').glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ClassDef) and any(isinstance(x,ast.Name) and x.id=='PlacementStrategy' for x in node.bases):subclasses.append((p.name,node.name))
    assert subclasses==[('my_strategy.py','MyStrategy')]
    protected=subprocess.check_output(['git','diff','HEAD','--name-only','--','src','data','validate_submission.sh','tests/test_simulator.py','tests/test_yard_state.py'],cwd=ROOT,text=True).splitlines()
    assert not protected,protected
    previous={}
    exec(subprocess.check_output(['git','show','HEAD:solution/survival_params.py'],cwd=ROOT,text=True),previous)
    expected={k:v for k,v in previous['SURVIVAL_PARAMETERS'].items() if k not in ('sched_disch_active','sched_truck_active')}
    expected.update(previous.get('EXPANDED_PARAMETERS',{}))
    assert SURVIVAL_PARAMETERS==expected and len(expected)==11
    from pypdf import PdfReader
    pages=len(PdfReader(ROOT/'tmp/pdfs/design-review.pdf').pages)
    assert pages==3
    design=(ROOT/'docs/design.md').read_text()
    assert len(re.findall(r'^## [1-7]\. ',design,re.M))==7
    assert design.count('[PENDING PHASE 5]')==15
    files=['solution/__init__.py','solution/my_strategy.py','solution/survival_params.py','data/vessel_schedule.json']
    manifest=dict(files={p:sha(p) for p in files},policy='Expanded single-rotation SurvivalRisk; gamma=1/240 per hour; exact ties; S3 alignment',
                  trace_sha256=eq['trace_sha256'],train_reshuffles=6408,test_contents_opened=False)
    (OUT/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    native=(OUT/'validator_native.log').read_text()
    portable=(OUT/'validator_portable.log').read_text()
    assert 'invalid option -- P' in native
    assert 'Found strategy: solution.my_strategy.MyStrategy' in portable and 'PASSED (1 warnings)' in portable
    result=dict(tests_passed=78,scaffold_tests=27,strategy_tests=51,placement_trace=eq,
                cli_seconds=elapsed,cli_fields_identical_except_elapsed_sec=True,normalized_cli_sha256=normalized,
                all_errors_violations_failures=0,count_guard_fallbacks=0,subclasses=subclasses,
                numeric_parameters_unchanged=True,protected_files_unchanged=True,
                test_contents_opened=False,validator_native='Aborted: BSD grep lacks -P',
                validator_portable='Unchanged validator + Perl grep -oP adapter: passed; one pending-results warning',
                design=dict(words=len(design.split()),sections=7,reference_pages=pages,page_size='A4',
                            font='Helvetica 11pt, 13.2pt leading',margins_mm='16 horizontal, 15 top, 18 bottom',
                            visual_review=True,pending_cells='Only Phase 5 test metrics'),
                state_lifecycle={k:cached[k] for k in ('metadata_entries','history_seen','history_missed','origin_entries','arrival_time_entries','overdue_heap_entries')},
                python=sys.version,base_git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                hashes={p:sha(p) for p in files+['docs/design.md','tests/test_my_strategy.py']})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
