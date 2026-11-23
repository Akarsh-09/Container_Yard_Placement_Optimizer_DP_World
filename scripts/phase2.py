"""Train-only six-mode experiment. No fitting and no access to data/test.

Run: .venv/bin/python scripts/phase2.py
Offline outcome labels go only to the observer, never to MyStrategy.
"""
import collections
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase1 import DiagnosticSimulator, ObservedBaseline
from solution.my_strategy import MODES, MyStrategy, parse_time
from src.event_reader import read_events
from src.yard_state import YardState

OUT = ROOT / "results/phase2"
ARRIVALS = {"DISCHARGE","TRUCK_RECV"}
PHASE2_MODES = tuple(mode for mode in MODES if mode != "expected_blocking")


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_chart():
    runs=json.loads((OUT/"experiments.json").read_text())["runs"]
    baseline=json.loads((ROOT/"results/phase1/baselines.json").read_text())
    rows=[]
    for r in [baseline["runs"][0]]+runs:
        rows.append({"name":r.get("mode","Greedy"),"total":r["summary"]["total_reshuffles"],
                     "early":r["slices"]["period/1-12"]["reshuffles"]/5042,
                     "late":r["slices"]["period/13-20"]["reshuffles"]/3832})
    randoms=baseline["runs"][1:]
    rows.append({"name":"Random mean (5 seeds)","total":sum(r["summary"]["total_reshuffles"] for r in randoms)/5,
                 "early":sum(r["slices"]["period/1-12"]["reshuffles"] for r in randoms)/5/5042,
                 "late":sum(r["slices"]["period/13-20"]["reshuffles"] for r in randoms)/5/3832})
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
    body{margin:0;font-family:var(--font-sans);color:var(--foreground)}label{font-size:13px}select{font:inherit;background:var(--card);color:var(--foreground);border:1px solid var(--border);border-radius:var(--radius);padding:6px;margin-left:8px}svg{width:100%;height:350px}text{fill:var(--foreground);font-size:12px}.base{fill:var(--chart-2)}.candidate{fill:var(--chart-1)}.best{fill:var(--chart-3)}p{font-size:12px;color:var(--muted-foreground);margin:0;line-height:1.5}
    </style></head><body><label>Metric<select id="metric"><option value="total">Full-train reshuffles</option><option value="early">Days 1–12 / found retrieval</option><option value="late">Days 13–20 / found retrieval</option></select></label><svg id="chart" viewBox="0 0 660 350" role="img" aria-label="Phase 2 training comparison"></svg><p>Lower is better. Period rates exclude absent targets. All candidates use fixed rules; no test data was accessed.</p><script>
    const rows=DATA;const ns='http://www.w3.org/2000/svg';function add(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;document.querySelector('#chart').appendChild(e);return e}function render(){document.querySelector('#chart').replaceChildren();const key=document.querySelector('#metric').value;const limit=Math.max(...rows.map(r=>r[key]))*1.13;rows.forEach((r,i)=>{let y=12+i*42;add('text',{x:0,y:y+15},r.name);const bar=add('rect',{x:175,y:y+1,width:r[key]/limit*420,height:21,rx:3,class:r.name==='Greedy'?'base':r.name==='s3_next_feasible'?'best':'candidate'});const title=document.createElementNS(ns,'title');title.textContent=r.name+': '+r[key];bar.appendChild(title);add('text',{x:182+r[key]/limit*420,y:y+16},key==='total'?r[key].toLocaleString('en-US',{maximumFractionDigits:1}):r[key].toFixed(4))})}document.querySelector('#metric').addEventListener('change',render);render();
    </script></body></html>'''.replace('DATA',json.dumps(rows))
    (OUT/"comparison.html").write_text(html)


def records_and_diagnostics(initial,events,schedule):
    records={c["container_id"]:dict(c,origin="initial",retrieval=None) for c in initial["containers"]}
    active=dict(records)
    found=[]
    for e in events:
        if e.type in ARRIVALS:
            record=dict(vars(e),origin=e.type,retrieval=None)
            records[e.container_id]=active[e.container_id]=record
        else:
            record=active.pop(e.container_id,None)
            if record is not None:
                from datetime import datetime
                record["retrieval"]=datetime.fromisoformat(e.timestamp)
                found.append((e,record))
    vessels={v["vessel_id"]:v for v in schedule["vessels"]}
    mismatch=collections.defaultdict(collections.Counter)
    truck=collections.defaultdict(collections.Counter)
    for e,r in found:
        rotations=vessels.get(r["vessel_id"],{}).get("rotations",[])
        if not rotations:
            continue
        nearest=min(rotations,key=lambda ro:abs(parse_time(ro["etd"])-parse_time(r["departure_time"])))
        gap=abs(parse_time(nearest["etd"])-parse_time(r["departure_time"]))
        if e.type=="TRUCK_DLVR":
            for origin in ("all",r["origin"]):
                truck[origin]["n"]+=1
                truck[origin]["within_1h"]+=gap<=3600
                truck[origin]["within_24h"]+=gap<=86400
                truck[origin]["exact"]+=gap==0
        elif not parse_time(nearest["load_start"])<=parse_time(e.timestamp)<=parse_time(nearest["load_end"]):
            actual=[ro for ro in rotations if parse_time(ro["load_start"])<=parse_time(e.timestamp)<=parse_time(ro["load_end"])]
            assert len(actual)==1
            direction="later_rotation" if parse_time(actual[0]["etd"])>parse_time(nearest["etd"]) else "earlier_rotation"
            for origin in ("all",r["origin"]):
                mismatch[origin][direction]+=1
    save("diagnostics.json",{"rotation_mismatched_LOAD":dict(mismatch),
                             "scheduled_vessel_TRUCK_DLVR_ETD_matches":dict(truck)})
    return records


def run(modes=PHASE2_MODES):
    OUT.mkdir(parents=True,exist_ok=True)
    layout=json.loads((ROOT/"data/yard_layout.json").read_text())
    initial=json.loads((ROOT/"data/train/initial_state.json").read_text())
    events=read_events(str(ROOT/"data/train/events.jsonl"))
    schedule=json.loads((ROOT/"data/vessel_schedule.json").read_text())
    records=records_and_diagnostics(initial,events,schedule)
    runs=[]
    for mode in modes:
        yard=YardState(layout)
        yard.load_initial_state(initial)
        base=MyStrategy(mode)
        observer=ObservedBaseline(base,records)
        observer.initialize(layout,initial)
        start=time.perf_counter()
        stats=DiagnosticSimulator(yard,observer).run(events)
        elapsed=time.perf_counter()-start
        assert stats.errors==stats.hard_constraint_violations==stats.placement_failures==0
        assert stats.retrieval_not_found==1333
        assert stats.final_yard_count==6311
        assert observer.bound_checked==1645
        assert observer.buckets["all"]["reshuffles"]==stats.total_reshuffles
        slices={k:dict(v,official_rate=v["reshuffles"]/v["retrievals"] if v["retrievals"] else 0,
                       found_rate=v["reshuffles"]/v["found"] if v["found"] else None)
                for k,v in observer.buckets.items()}
        result={"mode":mode,"summary":stats.summary(),"official_rate_exact":stats.reshuffles_per_retrieval,
                "score_exact":stats.quantitative_score(),"excess_above_LB":stats.total_reshuffles-1325,
                "excess_per_found":(stats.total_reshuffles-1325)/8874,
                "wall_sec":elapsed,"retrieval_not_found":stats.retrieval_not_found,"slices":slices,
                "rehandled_blocker_moves":observer.rehandled_blocker_moves,
                "unique_blockers_ever_moved":len(observer.moved_before),
                "strategy_sha256":hashlib.sha256((ROOT/"solution/my_strategy.py").read_bytes()).hexdigest(),
                "final_metadata_cache_size":len(base._meta)}
        runs.append(result)
        save(f"{mode}.json",result)
        print(mode,"reshuffles",stats.total_reshuffles,"rate",round(stats.reshuffles_per_retrieval,6),
              "early",round(slices["period/1-12"]["found_rate"],6),
              "late",round(slices["period/13-20"]["found_rate"],6),"seconds",round(elapsed,2),flush=True)
    runs=[json.loads((OUT/f"{mode}.json").read_text()) for mode in PHASE2_MODES if (OUT/f"{mode}.json").exists()]
    save("experiments.json",{"runs":runs,"fit":"No fitted parameters; rules specified before evaluation. S2 uses resident-only rollover as requested.",
                             "holdout":"Jan 13-20 warm-up continuation; exploratory comparison, not an untouched final holdout.",
                             "lower_bound":1325,"found_retrievals":8874})
    write_chart()
    paths=[ROOT/"solution/my_strategy.py",ROOT/"scripts/phase1.py",ROOT/"scripts/phase2.py",
           ROOT/"data/yard_layout.json",ROOT/"data/vessel_schedule.json",
           ROOT/"data/train/initial_state.json",ROOT/"data/train/events.jsonl"]
    save("provenance.json",{"python":sys.version,"command":".venv/bin/python scripts/phase2.py",
                            "sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                            "test_data_opened":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes",nargs="+",choices=PHASE2_MODES,default=list(PHASE2_MODES))
    run(parser.parse_args().modes)
