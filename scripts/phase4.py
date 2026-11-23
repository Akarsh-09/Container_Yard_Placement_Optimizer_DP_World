"""Production-only train replay, frozen-trace equivalence, and provenance.

No historical experiment module is imported; pruned modes are intentionally gone.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solution.my_strategy import MyStrategy, parse_time
from src.event_reader import read_events
from src.simulator import Simulator
from src.yard_state import YardState
OUT=ROOT/'results/phase4'
FIT_END=parse_time('2025-01-13T00:00:00')

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

class Traced(MyStrategy):
    def initialize(self,layout,initial):
        super().initialize(layout,initial)
        self.trace=[]
        self.slices=defaultdict(lambda:dict(found=0,reshuffles=0))
    def on_event(self,event):
        self.event=event
        super().on_event(event)
    def place_container(self,yard,event):
        p=super().place_container(yard,event)
        self.trace.append((event.event_id,p.block,p.bay,p.row,p.tier))
        return p
    def on_container_retrieved(self,cid,pos,reshuffles):
        origin=self._origins[cid]
        period='early' if self.current_time<FIT_END else 'late'
        for key in (period,'origin/'+origin,'exit/'+self.event.type,period+'/'+self.event.type,
                    'origin_exit/'+origin+'/'+self.event.type):
            self.slices[key]['found']+=1
            self.slices[key]['reshuffles']+=reshuffles
        super().on_container_retrieved(cid,pos,reshuffles)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    layout=json.loads((ROOT/'data/yard_layout.json').read_text())
    initial=json.loads((ROOT/'data/train/initial_state.json').read_text())
    events=read_events(str(ROOT/'data/train/events.jsonl'))
    old=json.loads((ROOT/'results/phase3d/cache_equivalence.json').read_text())
    traces=[]
    results=[]
    for cached in (True,False):
        yard=YardState(layout);yard.load_initial_state(initial)
        strategy=Traced(use_cache=cached);strategy.initialize(layout,initial)
        started=time.perf_counter()
        stats=Simulator(yard,strategy).run(events)
        elapsed=time.perf_counter()-started
        trace_hash=hashlib.sha256(json.dumps(strategy.trace,separators=(',',':')).encode()).hexdigest()
        assert trace_hash==old['trace_sha256']
        assert len(strategy.trace)==10385 and stats.total_reshuffles==6408
        assert stats.errors==stats.hard_constraint_violations==stats.placement_failures==strategy.cache_guard_fallbacks==0
        r=dict(use_cache=cached,seconds=elapsed,trace_sha256=trace_hash,placements=len(strategy.trace),
               summary=stats.summary(),retrieval_not_found=stats.retrieval_not_found,
               official_rate=stats.reshuffles_per_retrieval,score=stats.quantitative_score(),
               slices=dict(strategy.slices),cache_guard_fallbacks=strategy.cache_guard_fallbacks,
               metadata_entries=len(strategy._meta),history_seen=len(strategy.seen_in_yard_or_arrived),
               history_missed=len(strategy.observed_missed_loads),origin_entries=len(strategy._origins),
               arrival_time_entries=len(strategy._arrival_times),overdue_heap_entries=len(strategy._overdue_heap))
        save('cached.json' if cached else 'live.json',r)
        print('cached' if cached else 'live',stats.total_reshuffles,elapsed,trace_hash,flush=True)
        results.append(r);traces.append(strategy.trace)
    assert traces[0]==traces[1]
    save('equivalence.json',dict(identical_placements=10385,phase3d_hash_match=True,
                                trace_sha256=results[0]['trace_sha256'],cached_seconds=results[0]['seconds'],
                                live_seconds=results[1]['seconds']))
if __name__=='__main__':main()
