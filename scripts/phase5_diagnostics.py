"""Read-only final input/identity audit; never instantiate or replay MyStrategy."""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.yard_state import YardState
from src.event_reader import read_events
OUT=ROOT/'results/phase5'
def main():
    initial=json.loads((ROOT/'data/test/initial_state.json').read_text())
    layout=json.loads((ROOT/'data/yard_layout.json').read_text())
    groups=defaultdict(list)
    for c in initial['containers']:
        p=c['position'];groups[(p['block'],p['bay'],p['row'])].append(p['tier'])
    holes=[]
    for coord,tiers in sorted(groups.items()):
        missing=sorted(set(range(1,max(tiers)+1))-set(tiers))
        if missing:holes.append(dict(stack=coord,occupied_tiers=sorted(tiers),missing_tiers=missing))
    yard=YardState(layout);yard.load_initial_state(initial)
    empty_slots=[];span=0
    for b,info in sorted(layout['blocks'].items()):
        for bay in range(1,info['bays']+1):
            for row in range(1,info['rows']+1):
                height=yard.get_stack_height(b,bay,row);span+=height
                for tier in range(1,height+1):
                    cid=yard.get_container_at(b,bay,row,tier)
                    if yard.get_container_info(cid) is None:
                        empty_slots.append(dict(block=b,bay=bay,row=row,tier=tier,cid=cid))
    active={c['container_id'] for c in initial['containers']}
    counts=Counter()
    for e in read_events(str(ROOT/'data/test/events.jsonl')):
        counts['events']+=1
        if e.type in ('DISCHARGE','TRUCK_RECV'):
            counts['arrivals']+=1;active.add(e.container_id)
        else:
            counts['retrievals']+=1
            if e.container_id in active:
                counts['found']+=1;active.remove(e.container_id)
            else:counts['absent']+=1
    r=dict(method='Read-only initial snapshot/public YardState inspection and identity ledger; no MyStrategy instance or simulator replay.',
           initial_containers=len(initial['containers']),stack_span=span,
           sparse_stacks=len(holes),missing_tier_slots=sum(len(h['missing_tiers']) for h in holes),
           sparse_stack_details=holes,public_API_empty_slots=empty_slots,
           first_empty_slot=empty_slots[0] if empty_slots else None,
           identity_counts=dict(counts),ledger_final_count=len(active),
           root_cause='The supplied loader pads sparse initial stacks with empty-string IDs. The frozen cache treats every tier through get_stack_height as a container; get_container_info(empty_string) returns None, then _remember(None) raises AttributeError before cache initialization completes.',
           result_interpretation='The official run used simulator fallback placements after each exception; its reshuffle count is not a successful SurvivalRisk evaluation. No fix or second strategy test run was performed.')
    (OUT/'diagnostics.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('sparse_stack_details','public_API_empty_slots')},indent=2))
if __name__=='__main__':main()
