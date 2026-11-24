"""Sparse-stack regressions for each paradigm, plus scalar/vector score equivalence."""
from common import *
from policies import Policy,THETA0
import importlib.util
import numpy as np

def main():
    spec=importlib.util.spec_from_file_location('phase7a_existing_tests',ROOT/'tests/test_my_strategy.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    temporary=OUT/'synthetic';temporary.mkdir(exist_ok=True);schedule=t.schedule_path.__wrapped__(temporary)
    configs=[('ADP_theta0','ADP',THETA0),('ADP_dense','ADP',[1.,.7,.2,-.1,.4,-.2,.1,.05,.1,-.1,.2,.3,-.2,-.1,.1,.05])]+[(m,m,THETA0) for m in ('CD1_aggressive_compact','CD2_drain_and_shield','CD3_sink_blocks','CD4_init_quarantine')]
    final=OUT/'final_weights.json'
    if final.exists():configs.append(('ADP_final','ADP',json.loads(final.read_text())['theta']))
    checks=[]
    for name,mode,theta in configs:
        def factory(schedule_path=None,use_cache=True):
            s=Policy(mode,theta);s.schedule_path=schedule_path;s.use_cache=use_cache;return s
        t.MyStrategy=factory
        for cached in (True,False):
            t.test_sparse_initial_height_and_real_container_count(cached)
            t.test_full_sparse_stack_is_skipped(cached)
            t.test_gap_above_target_failed_removal_rehydrates_metadata(cached,schedule)
        t.test_sparse_top_retrieval_leaves_ghost_stack(False)
        t.test_unannounced_live_container_has_defensive_arrival_time(schedule)
        t.test_sparse_cached_live_multistep_equivalence(schedule)
        checks.append(dict(policy=name,sparse_checks=9,passed=True))
    # Independently coded scalar feature policy vs vector aggregation on 96
    # closed-loop real-yard placements, spanning overdue and rotation transitions.
    scalar=Policy.__mro__[1]
    class Audited(Policy):
        def __init__(self,theta):super().__init__(theta=theta);self.compared=0
        def _place_survival(self,e):
            finish=self._finish_placement
            self._finish_placement=lambda i:i
            a=super()._place_survival(e)
            if self.compared<96:
                b=scalar._place_survival(self,e);assert a==b,(e.event_id,a,b);self.compared+=1
            self._finish_placement=finish
            return finish(a)
    layout,initial,events=inputs()
    for name,mode,theta in [configs[1]]:
        s=Audited(theta);yard=YardState(layout);yard.load_initial_state(initial);s.initialize(layout,initial)
        # 96 actual arrivals are enough for an independent aggregation check.
        prefix=[];n=0
        for e in events:
            prefix.append(e);n+=e.type in ARRIVALS
            if n==96:break
        stats=Simulator(yard,s).run(prefix);assert stats.errors==stats.hard_constraint_violations==stats.placement_failures==s.cache_guard_fallbacks==0
        checks.append(dict(policy=name,scalar_vector_identical_placements=s.compared,passed=True))
    save('sparse_verification.json',dict(checks=checks,passed=True))
    print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
