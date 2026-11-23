"""Seventeen isolated train ablations; M5 is gated by entry-time accuracy.

Run: .venv/bin/python scripts/phase3b.py
Default MyStrategy stays unchanged. No combinations or test-set access.
"""
import hashlib
import argparse
import itertools
import json
from math import isinf
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import phase3
from phase3 import FIT_END, ARRIVALS, Traced
from phase3b_diag import load_train_inputs
from solution.my_strategy import MyStrategy, parse_time

OUT=ROOT/"results/phase3b"
BASE=dict(mode="expected_blocking",timing_key="s3_next_feasible",epsilon=0.05,sigma_hours=12,
          lambda_empty=0.0,lambda_gap=0.0)


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


class PeriodExitObserver(Traced):
    def on_event(self,event):
        super().on_event(event)
        if event.type not in ARRIVALS:
            key="period_exit/"+("early/" if parse_time(event.timestamp)<FIT_END else "late/")+event.type
            self.buckets_for_event.append(key)
            self.buckets[key]["retrievals"]+=1


def display_config(config):
    return {key:"infinity" if isinstance(value,float) and isinf(value) else value for key,value in config.items()}


def verify_caches():
    inputs=load_train_inputs()
    phase3.OUT=OUT
    checks=[]
    for name,flag in [("M2a_0.20",{"expired_overdue_prob":0.20}),
                      ("M4_inf",{"missed_load_offset_hours":float("inf")})]:
        config=dict(BASE,**flag)
        cached,a=phase3.evaluate("verify_"+name+"_cached",MyStrategy(**config),inputs,
                                display_config(config),observer_class=PeriodExitObserver)
        live,b=phase3.evaluate("verify_"+name+"_live",MyStrategy(**config,use_cache=False),inputs,
                              dict(display_config(config),use_cache=False),observer_class=PeriodExitObserver)
        original=json.loads((OUT/(name+".json")).read_text())
        assert a==b
        assert cached["trace_sha256"]==original["trace_sha256"]==live["trace_sha256"]
        checks.append({"name":name,"identical_placements":len(a),"trace_sha256":cached["trace_sha256"],
                       "cached_seconds":cached["wall_sec"],"live_seconds":live["wall_sec"]})
    save("cache_equivalence.json",checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-caches",action="store_true")
    args=parser.parse_args()
    if args.verify_caches:
        verify_caches()
        return
    OUT.mkdir(parents=True,exist_ok=True)
    diag=json.loads((OUT/"diagnostics.json").read_text())
    assert not diag["A4"]["M5_gate"], "M5 gate changed; review its explicit definition before adding a hybrid"
    inputs=load_train_inputs()
    phase3.OUT=OUT
    baseline,trace=phase3.evaluate("baseline",MyStrategy(),inputs,BASE,observer_class=PeriodExitObserver)
    previous=json.loads((ROOT/"results/phase3/A_24.json").read_text())
    assert baseline["summary"]["total_reshuffles"]==7026
    assert baseline["trace_sha256"]==previous["trace_sha256"]
    configs=[]
    for value in (0.10,0.15,0.20,0.25,0.30):
        configs.append((f"M1_{value:.2f}",dict(BASE,initial_epsilon=value)))
    for prefix,flag in (("M2a","expired_overdue_prob"),("M2b","all_overdue_prob")):
        for value in (0.10,0.20,0.30): configs.append((f"{prefix}_{value:.2f}",dict(BASE,**{flag:value})))
    for intra,cross in itertools.product((0.5,2.0),(12,48)):
        configs.append((f"M3_{intra:g}_{cross}",dict(BASE,intra_rotation_sigma_hours=intra,sigma_hours=cross)))
    configs.extend([("M4_14d",dict(BASE,missed_load_offset_hours=336)),
                    ("M4_inf",dict(BASE,missed_load_offset_hours=float("inf")))])
    assert len(configs)==17
    runs=[]
    for name,config in configs:
        result,_=phase3.evaluate(name,MyStrategy(**config),inputs,display_config(config),observer_class=PeriodExitObserver)
        early=result["slices"]["period/1-12"]["reshuffles"]
        late=result["slices"]["period/13-20"]["reshuffles"]
        result["early_lift"]=3423-early
        result["late_lift"]=3603-late
        result["fit_improves"]=early<3423
        result["continuation_nonregression"]=late<=3603
        if name.startswith("M3"):
            result["M3_strict_cohort_gate"]=all(
                result["slices"][f"period_exit/{p}/LOAD"]["reshuffles"]<baseline["slices"][f"period_exit/{p}/LOAD"]["reshuffles"]
                and result["slices"][f"period_exit/{p}/TRUCK_DLVR"]["reshuffles"]<=baseline["slices"][f"period_exit/{p}/TRUCK_DLVR"]["reshuffles"]
                for p in ("early","late"))
        runs.append(result)
        save(name+".json",result)
        save("ablations.json",runs)
    save("summary.json",{"baseline":baseline,"runs":runs,"M5_skipped":True,
                         "best_fit":min(runs,key=lambda r:r["slices"]["period/1-12"]["reshuffles"])["name"],
                         "default_unchanged":True,"no_combinations":True,
                         "selection_rule":"Early counts only; late continuation is a non-regression gate, not a fitting target."})
    save("provenance.json",{"python":sys.version,"test_data_opened":False,
                            "command":".venv/bin/python scripts/phase3b.py",
                            "strategy_sha256":hashlib.sha256((ROOT/"solution/my_strategy.py").read_bytes()).hexdigest(),
                            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=="__main__": main()
