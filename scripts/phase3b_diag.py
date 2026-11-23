"""Train-only blocker provenance and fit-period feature diagnostics.

Run: .venv/bin/python scripts/phase3b_diag.py
Outcome labels never reach placement logic. Late-period statistics are autopsy
context only. Rotation-accuracy predictions are made prospectively at entry.
"""
import collections
from datetime import datetime
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import phase3
from phase3 import FIT_END, ARRIVALS, BINS, bin_index
from solution.my_strategy import MyStrategy, parse_time
from src.yard_state import YardState
from src.simulator import Simulator
from src.event_reader import read_events

OUT=ROOT/"results/phase3b"
END=parse_time("2025-01-21T00:00:00")
START=parse_time("2025-01-01T00:00:00")


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def load_train_inputs():
    """Pure loader: labels are for observers/diagnostics, never the strategy."""
    layout=json.loads((ROOT/"data/yard_layout.json").read_text())
    initial=json.loads((ROOT/"data/train/initial_state.json").read_text())
    events=read_events(str(ROOT/"data/train/events.jsonl"))
    records={c["container_id"]:dict(c,origin="initial",retrieval=None) for c in initial["containers"]}
    active=dict(records)
    for event in events:
        if event.type in ARRIVALS:
            record=dict(vars(event),origin=event.type,retrieval=None)
            records[event.container_id]=active[event.container_id]=record
        else:
            record=active.pop(event.container_id,None)
            if record is not None:
                record["retrieval"]=datetime.fromisoformat(event.timestamp)
    return layout,initial,events,records


def period(now):
    return "Days1-12" if now<FIT_END else "Days13-20"


def state(meta,now,next_rotations):
    if meta.departure >= now:
        return "active"
    return "scheduled_overdue" if meta.vessel in next_rotations else "no_future_rot_overdue"


class AutopsyStrategy(MyStrategy):
    def __init__(self,records,initial):
        super().__init__()
        self.records=records
        self.initial_ids={c["container_id"] for c in initial["containers"]}
        self.initial_slots={c["container_id"]:tuple(c["position"][k] for k in ("block","bay","row")) for c in initial["containers"]}
        self.landings={cid:{"source":"initial_unmoved"} for cid in self.initial_ids}
        self.seen=set(self.initial_ids)
        self.missed=set()
        self.missed_at_arrival={}
        self.cells=collections.Counter()
        self.sources=collections.defaultdict(collections.Counter)
        self.source_exit=collections.defaultdict(collections.Counter)
        self.exposure=collections.defaultdict(collections.Counter)

    def on_event(self,event):
        self.event=event
        if event.type=="LOAD" and event.container_id not in self.seen:
            self.missed.add(event.container_id)
        if event.type in ARRIVALS:
            self.missed_at_arrival[event.container_id]=event.container_id in self.missed
            self.seen.add(event.container_id)
        super().on_event(event)

    def place_container(self,yard,event):
        position=super().place_container(yard,event)
        assert yard.is_position_valid(position)
        now=self.current_time
        p=period(now)
        exposure=self.exposure[p]
        for i,(_,_,_,max_tiers) in enumerate(self._stacks):
            if len(self._contents[i])>=max_tiers:
                continue
            for cid in self._contents[i]:
                meta=self._meta[cid]
                exposure["scanned_residents"]+=1
                if meta.vessel not in self._next_by_vessel:
                    group=("truck_only" if meta.vessel in {"VSL019","VSL020"}
                           else "scheduled_past_final" if self._rotations.get(meta.vessel) else "other")
                    exposure["no_future_rotation"]+=1
                    exposure[group]+=1
                    if meta.departure<now:
                        exposure["no_future_rot_overdue"]+=1
                        exposure[group+"_overdue"]+=1
        incoming=self._incoming_priority(event,self._meta[event.container_id])
        underneath={}
        for tier in range(1,position.tier):
            cid=yard.get_container_at(position.block,position.bay,position.row,tier)
            meta=self._meta[cid]
            delta=self._eff[cid]-incoming
            underneath[cid]={"belief":"safe" if delta>0 else "tie" if delta==0 else "inversion",
                             "target_origin":"initial" if cid in self.initial_ids else "new",
                             "target_state":state(meta,now,self._next_by_vessel)}
        self.landings[event.container_id]={"source":"placed_by_strategy","underneath":underneath,
                                           "blocker_origin":event.type}
        return position

    def record_blocker(self,target,blocker):
        now=self.current_time
        p=period(now)
        landing=self.landings[blocker]
        source=landing["source"]
        self.sources[p][source]+=1
        self.source_exit[p][source+"/"+self.event.type]+=1
        if source=="initial_unmoved":
            assert target in self.initial_ids
            assert self.initial_slots[target]==self.initial_slots[blocker]
        if source=="placed_by_strategy":
            assert target in landing["underneath"], "Target could not have arrived below an existing blocker"
            attributes=landing["underneath"][target]
            retrieved=self.records[blocker]["retrieval"]
            horizon=FIT_END if p=="Days1-12" else END
            censored=retrieved is None or parse_time(retrieved.isoformat())>=horizon
            key=(p,attributes["belief"],landing["blocker_origin"],self.missed_at_arrival.get(blocker,False),
                 censored,attributes["target_origin"],attributes["target_state"],self.event.type)
            self.cells[key]+=1


class AutopsySimulator(Simulator):
    def _do_retrieval(self,target_id,above):
        for cid in above:
            self.strategy.record_blocker(target_id,cid)
        super()._do_retrieval(target_id,above)
        for cid in above:
            self.strategy.landings[cid]={"source":"placed_by_relocation"}


def autopsy(inputs):
    layout,initial,events,records=inputs
    yard=YardState(layout)
    yard.load_initial_state(initial)
    strategy=AutopsyStrategy(records,initial)
    strategy.initialize(layout,initial)
    stats=AutopsySimulator(yard,strategy).run(events)
    assert stats.total_reshuffles==7026
    assert sum(sum(c.values()) for c in strategy.sources.values())==7026
    assert sum(c.get("initial_unmoved",0) for c in strategy.sources.values())>=1325
    columns=("period","belief","blocker_origin","missed_load_at_arrival","blocker_unretrieved_at_period_end",
             "target_origin","target_state_at_placement","target_exit")
    cells=[dict(zip(columns,key),count=count) for key,count in sorted(strategy.cells.items())]
    for p in strategy.sources:
        assert sum(x["count"] for x in cells if x["period"]==p)==strategy.sources[p]["placed_by_strategy"]
    return {"A2":{"sources":dict(strategy.sources),"source_by_target_exit":dict(strategy.source_exit),
                  "strategy_placement_cells":cells,
                  "interpretation":"Most recent landing provenance, not counterfactual causal responsibility. Fit-period censor labels stop at Jan13; late labels stop at Jan21."},
            "A3_exposure":dict(strategy.exposure)}


def signals(inputs):
    layout,initial,events,records=inputs
    models={key:MyStrategy(mode="expected_blocking",timing_key=key) for key in ("raw","s2_rollover","s3_next_feasible")}
    for model in models.values(): model.initialize(layout,initial)
    initial_ids={c["container_id"] for c in initial["containers"]}
    found_order={}
    live=set(initial_ids)
    fit_found=[]
    for index,event in enumerate(events):
        if parse_time(event.timestamp)>=FIT_END: break
        if event.type in ARRIVALS:
            live.add(event.container_id)
        elif event.container_id in live:
            live.remove(event.container_id)
            found_order[event.container_id]=index
            fit_found.append(event)

    def assigned(model,meta,now,incoming):
        previous=model.current_time
        model.current_time=now
        future=model._next_rotations().get(meta.vessel)
        model.current_time=previous
        if model._timing=="s3_next_feasible": return future
        rotation=meta.matched_rotation
        if incoming or rotation is None or now<=rotation.end: return rotation
        return future if future is not None else rotation

    predicted={}
    for c in initial["containers"]:
        predicted[c["container_id"]]={key:assigned(m,m._meta[c["container_id"]],START,False)
                                      for key,m in models.items() if key!="raw"}
    live={c["container_id"]:c for c in initial["containers"]}
    seen=set(live)
    missed=set()
    missed_arrivals=set()
    rng=random.Random(303)
    curves=collections.defaultdict(lambda:[[0,0] for _ in BINS])
    original_counts=collections.defaultdict(lambda:[[0,0] for _ in BINS])
    samples=eligible=0
    known_at_fit=dict(live)
    for index,e in enumerate(events):
        now=parse_time(e.timestamp)
        if now>=FIT_END: break
        for m in models.values(): m.on_event(e)
        if e.type in ARRIVALS:
            predicted[e.container_id]={key:assigned(m,m._meta[e.container_id],now,True)
                                       for key,m in models.items() if key!="raw"}
            if e.container_id in missed: missed_arrivals.add(e.container_id)
            pairs=rng.sample(list(live),min(32,len(live)))
            next_ro={key:m._next_rotations() for key,m in models.items()}
            for cid in pairs:
                samples+=1
                tx=found_order.get(cid,float("inf"))
                tc=found_order.get(e.container_id,float("inf"))
                if tx==tc==float("inf"): continue
                eligible+=1
                outcome=int(tx<tc)
                s3=models["s3_next_feasible"]
                mx,mc=s3._meta[cid],s3._meta[e.container_id]
                rx=next_ro["s3_next_feasible"].get(mx.vessel)
                rc=next_ro["s3_next_feasible"].get(mc.vessel)
                same=mx.vessel==mc.vessel and rx is not None and rx==rc
                groups=["same_S3_rotation" if same else "different_or_fallback"]
                if mx.departure<now and rx is None: groups.append("no_future_rot_overdue")
                for key,m in models.items():
                    delta=(m._priority(m._meta[cid],next_ro[key])-m._priority(m._meta[e.container_id],next_ro[key],resident=False))/3600
                    bucket=bin_index(delta)
                    original_counts[key][bucket][0]+=1
                    original_counts[key][bucket][1]+=outcome
                    for group in groups:
                        cell=curves[key+"/"+group][bucket]
                        cell[0]+=1
                        cell[1]+=outcome
            live[e.container_id]=vars(e)
            known_at_fit[e.container_id]=vars(e)
            seen.add(e.container_id)
        else:
            if e.type=="LOAD" and e.container_id not in seen: missed.add(e.container_id)
            live.pop(e.container_id,None)

    previous=json.loads((ROOT/"results/phase3/D2_pairwise.json").read_text())
    assert samples==previous["sampled_pairs"] and eligible==previous["eligible_pairs"]
    for key,counts in original_counts.items():
        for old,(n,yes) in zip(previous["tables"][key+"/all"],counts):
            assert (old["n"],old["resident_first"])==(n,yes)
    curve_tables={key:[dict(bin=b,n=n,resident_first=yes,probability=yes/n if n else None)
                       for b,(n,yes) in zip(BINS,cells)] for key,cells in curves.items()}
    accuracy=collections.defaultdict(collections.Counter)
    s3=models["s3_next_feasible"]
    for e in fit_found:
        if e.type!="LOAD": continue
        origin=records[e.container_id]["origin"]
        now=parse_time(e.timestamp)
        actual=[r for r in s3._rotations[e.vessel_id] if r.start<=now<=r.end]
        assert len(actual)==1
        for key in ("s2_rollover","s3_next_feasible"):
            accuracy[origin][key+"_correct"]+=predicted[e.container_id][key]==actual[0]
        accuracy[origin]["n"]+=1
    initial_acc=accuracy["initial"]
    s2_new=sum(accuracy[o]["s2_rollover_correct"] for o in ("DISCHARGE","TRUCK_RECV"))
    s3_new=sum(accuracy[o]["s3_next_feasible_correct"] for o in ("DISCHARGE","TRUCK_RECV"))
    gate=initial_acc["s3_next_feasible_correct"]>initial_acc["s2_rollover_correct"] and s2_new>s3_new
    outcomes={e.container_id:e.type for e in fit_found}
    attributes=collections.defaultdict(collections.Counter)
    for cid,r in known_at_fit.items():
        outcome=outcomes.get(cid,"unretrieved_at_Day12")
        membership=r.get("port_of_discharge","") in s3._ports.get(r.get("vessel_id",""),())
        attributes["weight/"+r.get("weight_class","MEDIUM")][outcome]+=1
        attributes["port_in_vessel/"+str(membership)][outcome]+=1
        attributes["joint/"+r.get("weight_class","MEDIUM")+"/"+str(membership)][outcome]+=1
    return {"missed_LOAD_fit":{"arrived_by_Day12":len(missed_arrivals),
                               "retrieved_by_Day12":sum(cid in found_order for cid in missed_arrivals)},
            "pairwise":{"sampled_pairs":samples,"eligible_pairs":eligible,"same_seed_sample_verified":True,"tables":curve_tables,
                        "warning":"Known-order conditional sample; excludes pairs with both containers censored at Day12."},
            "entry_time_rotation_accuracy":dict(accuracy),"M5_gate":gate,
            "M5_gate_rule":"S3 strictly better for initial; S2 strictly better for combined DISCHARGE+TRUCK_RECV at entry, fit-period found LOADs only.",
            "attribute_contingencies":dict(attributes)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=load_train_inputs()
    result=autopsy(inputs)
    a1={}
    for name,path in [("Greedy","greedy_control.json"),("Selected","A_24.json"),("H1","H1_hindsight_LDT.json")]:
        r=json.loads((ROOT/"results/phase3"/path).read_text())
        a1[name]={key:value for key,value in r["slices"].items() if key.startswith("five_day/")}
    result["A1"]=a1
    schedule=json.loads((ROOT/"data/vessel_schedule.json").read_text())
    rotations=[r for v in schedule["vessels"] for r in v["rotations"]]
    result["A3_schedule"]={"earliest_eta":min(r["eta"] for r in rotations),"latest_etd":max(r["etd"] for r in rotations),
                           "rotation_ETD_counts":dict(collections.Counter("Days1-12" if parse_time(r["etd"])<FIT_END else "Days13-20" if parse_time(r["etd"])<END else "Days21+" for r in rotations)),
                           "load_end_counts":dict(collections.Counter("Days1-12" if parse_time(r["load_end"])<FIT_END else "Days13-20" if parse_time(r["load_end"])<END else "Days21+" for r in rotations))}
    result["A4"]=signals(inputs)
    save("diagnostics.json",result)
    print("Autopsy sources",result["A2"]["sources"],flush=True)
    print("M5 gate",result["A4"]["M5_gate"],result["A4"]["entry_time_rotation_accuracy"],flush=True)


if __name__=="__main__": main()
