"""Eight core survival runs, two fit-gated expansions, two simple controls.

All fitted constants come from the Day1-12 audit and are numeric solution
defaults. Selection is by early found-retrieval count; late results are gates.
"""
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import phase3
from phase3b import PeriodExitObserver, display_config, BASE
from phase3b_diag import load_train_inputs
from solution.my_strategy import MyStrategy

OUT=ROOT/"results/phase3c"


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def evaluate(name,config,inputs):
    return phase3.evaluate(name,MyStrategy(**config),inputs,display_config(config),observer_class=PeriodExitObserver)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=load_train_inputs()
    phase3.OUT=OUT
    audit=json.loads((OUT/"audit.json").read_text())
    core=[]
    configs={}
    for index,(horizon,sigma) in enumerate(itertools.product((120,240,480,float("inf")),(0,2))):
        config=dict(mode="survival_risk",timing_key="s3_next_feasible",
                    survival_horizon_hours=horizon,survival_rotation_sigma_hours=sigma)
        name=f"S_{index:02d}"
        configs[name]=config
        r,_=evaluate(name,config,inputs)
        core.append(r)
        save("core.json",core)
    top=sorted(core,key=lambda r:r["slices"]["period/1-12"]["reshuffles"])[:2]
    expanded=[]
    if audit["fit"]["expanded_feature"]:
        for index,parent in enumerate(top):
            config=dict(configs[parent["name"]],use_expanded_survival=True)
            name=f"E_{index:02d}"
            configs[name]=config
            r,_=evaluate(name,config,inputs)
            expanded.append(r)
            save("expanded.json",expanded)
    controls=[]
    for label,offset in (("14d",336),("inf",float("inf"))):
        config=dict(BASE,expired_overdue_prob=.20,missed_load_offset_hours=offset)
        name="control_"+label
        r,_=evaluate(name,config,inputs)
        controls.append(r)
        save("controls.json",controls)
    winner=min(core+expanded,key=lambda r:r["slices"]["period/1-12"]["reshuffles"])
    config=configs[winner["name"]]
    save("selection.json",{"best_survival":winner,"top2_core":[r["name"] for r in top],
                           "expanded_feature":audit["fit"]["expanded_feature"],
                           "beats_baseline_both":winner["slices"]["period/1-12"]["reshuffles"]<3423 and winner["slices"]["period/13-20"]["reshuffles"]<3603,
                           "beats_M2a_both":winner["slices"]["period/1-12"]["reshuffles"]<3421 and winner["slices"]["period/13-20"]["reshuffles"]<3350,
                           "default_unchanged":True,"selection":"Early reshuffle count only; stable evaluation order breaks ties."})
    cached,a=evaluate("winning_cached",config,inputs)
    live,b=evaluate("winning_live",dict(config,use_cache=False),inputs)
    assert a==b
    assert cached["trace_sha256"]==winner["trace_sha256"]
    save("cache_equivalence.json",{"identical_placements":len(a),"cached_seconds":cached["wall_sec"],
                                    "live_seconds":live["wall_sec"],"trace_sha256":cached["trace_sha256"],
                                    "winner":winner["name"]})


if __name__=="__main__": main()
