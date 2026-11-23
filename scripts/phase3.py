"""Train-only diagnostics, exact cache controls, and bounded Phase 3 sweeps.

Run: .venv/bin/python scripts/phase3.py --stage all
Stages can run separately: preflight, diagnostics, grid. Selection uses only
Jan 1-12 found counts. Jan 13-20 is a previously explored continuation period.
"""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase1 import DiagnosticSimulator, ObservedBaseline, quantiles
from phase2 import records_and_diagnostics
from solution.my_strategy import MyStrategy, parse_time
from src.baseline_greedy import GreedyStrategy
from src.event_reader import read_events
from src.yard_state import YardState

OUT=ROOT/"results/phase3"
FIT_END=parse_time("2025-01-13T00:00:00")
ARRIVALS={"DISCHARGE","TRUCK_RECV"}
TIMINGS=("raw","s2_rollover","s3_next_feasible")
BINS=("<-72","[-72,-24)","[-24,-6)","[-6,0)","0","(0,6]","(6,24]","(24,72]",">72")


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def load_inputs():
    OUT.mkdir(parents=True,exist_ok=True)
    layout=json.loads((ROOT/"data/yard_layout.json").read_text())
    initial=json.loads((ROOT/"data/train/initial_state.json").read_text())
    events=read_events(str(ROOT/"data/train/events.jsonl"))
    schedule=json.loads((ROOT/"data/vessel_schedule.json").read_text())
    # phase2's helper writes diagnostics into its module's OUT. Redirect only its
    # artifact destination; the simulator and strategy are never monkey-patched.
    import phase2
    original_out=phase2.OUT
    phase2.OUT=OUT
    try:
        records=records_and_diagnostics(initial,events,schedule)
    finally:
        phase2.OUT=original_out
    return layout,initial,events,records


class Traced(ObservedBaseline):
    def __init__(self,base,records):
        super().__init__(base,records)
        self.trace=[]

    def place_container(self,yard,event):
        p=self.base.place_container(yard,event)
        self.trace.append((event.event_id,p.block,p.bay,p.row,p.tier))
        return p


def evaluate(name,base,inputs,config=None):
    layout,initial,events,records=inputs
    yard=YardState(layout)
    yard.load_initial_state(initial)
    observer=Traced(base,records)
    observer.initialize(layout,initial)
    started=time.perf_counter()
    stats=DiagnosticSimulator(yard,observer).run(events)
    elapsed=time.perf_counter()-started
    assert stats.errors==stats.placement_failures==stats.hard_constraint_violations==0
    assert stats.final_yard_count==6311 and stats.retrieval_not_found==1333
    assert observer.bound_checked==1645 and stats.total_reshuffles>=1325
    slices={k:dict(v,found_rate=v["reshuffles"]/v["found"] if v["found"] else None,
                   official_rate=v["reshuffles"]/v["retrievals"] if v["retrievals"] else None)
            for k,v in observer.buckets.items()}
    assert slices["period/1-12"]["found"]==5042 and slices["period/13-20"]["found"]==3832
    result={"name":name,"config":config,"summary":stats.summary(),"wall_sec":elapsed,
            "official_rate_exact":stats.reshuffles_per_retrieval,"score_exact":stats.quantitative_score(),
            "excess_above_LB":stats.total_reshuffles-1325,"excess_per_found":(stats.total_reshuffles-1325)/8874,
            "slices":slices,"rehandled_blocker_moves":observer.rehandled_blocker_moves,
            "trace_sha256":hashlib.sha256(json.dumps(observer.trace,separators=(",",":")).encode()).hexdigest(),
            "placements":len(observer.trace),"cache_guard_fallbacks":getattr(base,"cache_guard_fallbacks",None),
            "global_invalidations":getattr(base,"conservative_global_invalidations",None),
            "priority_updates":getattr(base,"priority_transition_updates",None),
            "strategy_sha256":hashlib.sha256((ROOT/"solution/my_strategy.py").read_bytes()).hexdigest()}
    save(name+".json",result)
    print(name,stats.total_reshuffles,"early",slices["period/1-12"]["reshuffles"],
          "late",slices["period/13-20"]["reshuffles"],"sec",round(elapsed,2),flush=True)
    return result,observer.trace


def preflight(inputs):
    config=dict(mode="expected_blocking",timing_key="s3_next_feasible",epsilon=0.15,sigma_hours=48,lambda_empty=0.0,lambda_gap=0.0)
    cached,trace=evaluate("equivalence_cached",MyStrategy(**config),inputs,config)
    live,other=evaluate("equivalence_live",MyStrategy(**config,use_cache=False),inputs,dict(config,use_cache=False))
    assert trace==other, "Cached and uncached placement sequences differ"
    assert cached["summary"]["total_reshuffles"]==live["summary"]["total_reshuffles"]
    endpoint=dict(mode="expected_blocking",timing_key="raw",epsilon=0.5,sigma_hours=48,lambda_empty=0.0,lambda_gap=0.0)
    limit,a=evaluate("greedy_limit",MyStrategy(**endpoint),inputs,endpoint)
    greedy,b=evaluate("greedy_control",GreedyStrategy(),inputs)
    assert a==b and limit["summary"]["total_reshuffles"]==8036
    save("equivalence.json",{"cached_live_identical":True,"placement_count":len(trace),
                             "cached_trace_sha256":cached["trace_sha256"],
                             "live_trace_sha256":live["trace_sha256"],
                             "cached_sec":cached["wall_sec"],"live_sec":live["wall_sec"],
                             "greedy_limit_identical_sequence":True,"greedy_total":8036,
                             "greedy_trace_sha256":greedy["trace_sha256"]})


class HindsightLDT(MyStrategy):
    """Offline train-only diagnostic; NEVER imported by solution/."""
    def __init__(self,actual):
        self.actual=actual
        super().__init__(mode="raw_strict")

    def _resident_priority(self,cid):
        return self.actual.get(cid,float("inf"))

    def _incoming_priority(self,event,meta):
        return self.actual.get(event.container_id,float("inf"))


def bin_index(delta):
    if delta < -72: return 0
    if delta < -24: return 1
    if delta < -6: return 2
    if delta < 0: return 3
    if delta == 0: return 4
    if delta <= 6: return 5
    if delta <= 24: return 6
    if delta <= 72: return 7
    return 8


def diagnostics(inputs):
    layout,initial,events,records=inputs
    actual={cid:parse_time(r["retrieval"].isoformat()) for cid,r in records.items() if r["retrieval"] is not None}
    evaluate("H1_hindsight_LDT",HindsightLDT(actual),inputs)
    arrival={e.container_id:e for e in events if e.type in ARRIVALS}
    active=set(c["container_id"] for c in initial["containers"])
    missed=set()
    found_order={}
    for index,e in enumerate(events):
        if e.type in ARRIVALS:
            active.add(e.container_id)
        elif e.container_id in active:
            active.remove(e.container_id)
            if parse_time(e.timestamp)<FIT_END:
                found_order[e.container_id]=index
        elif e.type=="LOAD":
            missed.add(e.container_id)

    def profile(ids):
        arrivals=[arrival[cid] for cid in ids]
        return {"n":len(arrivals),"ETD_lead_hours":quantiles([(parse_time(e.departure_time)-parse_time(e.timestamp))/3600 for e in arrivals]),
                "ETD_by_end_Day12":sum(parse_time(e.departure_time)<FIT_END for e in arrivals)}
    fit_arrivals={cid for cid,e in arrival.items() if parse_time(e.timestamp)<FIT_END}
    censored=fit_arrivals-set(found_order)
    save("D1_profiles.json",{"fit_arrivals_unretrieved_end_Day12":profile(censored),
                             "missed_LOAD_arrivals_by_Day12":profile(missed & fit_arrivals),
                             "all_1333_missed_LOAD_arrivals_train_only_descriptive":profile(missed),
                             "fit_cutoff_exclusive":"2025-01-13T00:00:00",
                             "warning":"Unretrieved at Day12 is right-censored, not a permanent-storage label. Full missed cohort is descriptive only, not used for fitting."})

    estimators={timing:MyStrategy(mode="expected_blocking",timing_key=timing) for timing in TIMINGS}
    for estimator in estimators.values():
        estimator.initialize(layout,initial)
    live={c["container_id"]:c for c in initial["containers"]}
    initial_ids=set(live)
    rng=random.Random(303)
    counts=collections.defaultdict(lambda:[[0,0] for _ in BINS])
    sampled=eligible=excluded=0
    for index,e in enumerate(events):
        now=parse_time(e.timestamp)
        if now>=FIT_END: break
        for estimator in estimators.values():
            estimator.on_event(e)
        if e.type in ARRIVALS:
            pairs=rng.sample(list(live),min(32,len(live)))
            next_ro={timing:m._next_rotations() for timing,m in estimators.items()}
            for cid in pairs:
                sampled+=1
                tx=found_order.get(cid,float("inf"))
                tc=found_order.get(e.container_id,float("inf"))
                if tx==tc==float("inf"):
                    excluded+=1
                    continue
                eligible+=1
                outcome=int(tx<tc)
                origin="initial" if cid in initial_ids else "new_arrival"
                overdue="overdue" if parse_time(live[cid].get("departure_time",""))<now else "not_overdue"
                for timing,m in estimators.items():
                    resident=m._meta[cid]
                    incoming_meta=m._meta[e.container_id]
                    delta=(m._priority(resident,next_ro[timing])-m._priority(incoming_meta,next_ro[timing],resident=False))/3600
                    bucket=bin_index(delta)
                    for group in ("all",origin,overdue,origin+"/"+overdue):
                        cell=counts[timing+"/"+group][bucket]
                        cell[0]+=1
                        cell[1]+=outcome
            live[e.container_id]=vars(e)
        else:
            live.pop(e.container_id,None)
    tables={key:[{"bin":label,"n":n,"resident_first":yes,"probability":yes/n if n else None}
                 for label,(n,yes) in zip(BINS,values)] for key,values in counts.items()}
    save("D2_pairwise.json",{"seed":303,"residents_sampled_per_arrival":32,"sampled_pairs":sampled,
                             "eligible_pairs":eligible,"excluded_both_censored":excluded,"tables":tables,
                             "overdue_definition":"Resident declared ETD < current event time.",
                             "warning":"Order is known for eligible pairs, but eligibility conditions on at least one observed departure by Day12. These are conditional curves, not unbiased full-population probabilities."})


def grid(inputs):
    configs=[dict(mode="expected_blocking",timing_key=t,epsilon=e,sigma_hours=s,lambda_empty=0.0,lambda_gap=0.0)
             for t,e,s in itertools.product(TIMINGS,(0.05,0.15,0.25,0.35),(12,48,168))]
    stage_a=[]
    for i,config in enumerate(configs):
        result,_=evaluate(f"A_{i:02d}",MyStrategy(**config),inputs,config)
        stage_a.append(result)
        save("stage_A.json",stage_a)
    # Stable config order breaks ties, never late-period or full-train outcomes.
    ranked=sorted(stage_a,key=lambda r:r["slices"]["period/1-12"]["reshuffles"])
    top1,top2=ranked[:2]
    stage_b=[top1]
    for i,(empty,gap) in enumerate(itertools.product((0.0,0.20,0.40),(0.0,0.05,0.15))):
        if empty==gap==0: continue
        config=dict(top1["config"],lambda_empty=empty,lambda_gap=gap)
        result,_=evaluate(f"B_{i:02d}",MyStrategy(**config),inputs,config)
        stage_b.append(result)
        save("stage_B.json",stage_b)
    best_b=min(stage_b,key=lambda r:r["slices"]["period/1-12"]["reshuffles"])
    best_pair={k:best_b["config"].get(k,0.0) for k in ("lambda_empty","lambda_gap")}
    transferred=dict(top2["config"],**best_pair)
    transfer,_=evaluate("B_transfer",MyStrategy(**transferred),inputs,transferred)
    save("stage_B_transfer.json",transfer)
    candidates=stage_a+stage_b[1:]+[transfer]
    winner=min(candidates,key=lambda r:r["slices"]["period/1-12"]["reshuffles"])
    save("selection.json",{"top2_A":[r["name"] for r in ranked[:2]],"best_B_pair":best_pair,
                           "winner":winner,"selection_rule":"Minimum Days1-12 reshuffle count; stable predeclared evaluation order for ties. No late-period tie-breaking.",
                           "both_periods_beat_greedy":winner["slices"]["period/1-12"]["reshuffles"]<4138 and winner["slices"]["period/13-20"]["reshuffles"]<3898,
                           "holdout_caveat":"Days13-20 was explored in earlier phases; it is a continuation evaluation, not an untouched final holdout."})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("all","preflight","diagnostics","grid"),default="all")
    args=parser.parse_args()
    inputs=load_inputs()
    if args.stage in ("all","preflight"): preflight(inputs)
    if args.stage in ("all","diagnostics"): diagnostics(inputs)
    if args.stage in ("all","grid"): grid(inputs)
    save("provenance_"+args.stage+".json",{"command":f".venv/bin/python scripts/phase3.py --stage {args.stage}",
                                          "python":sys.version,"test_data_opened":False,
                                          "strategy_sha256":hashlib.sha256((ROOT/"solution/my_strategy.py").read_bytes()).hexdigest(),
                                          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=="__main__": main()
