"""Fit-only survival audit plus separately labelled full-train landing context."""
from bisect import bisect_left
import collections
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase3 import FIT_END, ARRIVALS, BINS, bin_index
from phase3b_diag import START, load_train_inputs
from solution.my_strategy import MyStrategy, parse_time
from src.event_reader import iter_events
from src.models import Container
from src.simulator import Simulator
from src.yard_state import YardState

OUT=ROOT/"results/phase3c"
GROUPS=("missed_load","expired_no_rot","unscheduled_active","sched_overdue",
        "sched_init_active","sched_disch_active","sched_truck_active")
FEATURES=("etd_vs_s3","lead_bucket","in_discharge_window")


def load_fit():
    layout=json.loads((ROOT/"data/yard_layout.json").read_text())
    initial=json.loads((ROOT/"data/train/initial_state.json").read_text())
    records={c["container_id"]:dict(c,origin="initial",entry=START,end=FIT_END,exit_type=None,
                                   missed=False,exit_order=float("inf")) for c in initial["containers"]}
    active=dict(records)
    seen=set(records)
    missed=set()
    events=[]
    for event in iter_events(str(ROOT/"data/train/events.jsonl")):
        t=parse_time(event.timestamp)
        if t>=FIT_END: break
        index=len(events)
        events.append(event)
        if event.type in ARRIVALS:
            r=dict(vars(event),origin=event.type,entry=t,end=FIT_END,exit_type=None,
                   missed=event.container_id in missed,exit_order=float("inf"))
            records[event.container_id]=active[event.container_id]=r
            seen.add(event.container_id)
        elif event.container_id in active:
            r=active.pop(event.container_id)
            r.update(end=t,exit_type=event.type,exit_order=index)
        elif event.type=="LOAD" and event.container_id not in seen:
            missed.add(event.container_id)
    schedule=json.loads((ROOT/"data/vessel_schedule.json").read_text())
    return layout,initial,events,records,schedule


def rotation_at(model,vessel,t):
    ends=model._ends.get(vessel,())
    index=bisect_left(ends,t)
    return model._rotations[vessel][index] if index<len(ends) else None


def group_at(r,t,rotation):
    if r["missed"]: return "missed_load"
    overdue=parse_time(r.get("departure_time",""))<t
    if rotation is None: return "expired_no_rot" if overdue else "unscheduled_active"
    if overdue: return "sched_overdue"
    return {"initial":"sched_init_active","DISCHARGE":"sched_disch_active","TRUCK_RECV":"sched_truck_active"}[r["origin"]]


def features_at(r,rotation,vessels):
    raw=parse_time(r.get("departure_time",""))
    diff=(raw-rotation.etd)/3600 if rotation else None
    relation=("match" if diff is not None and abs(diff)<=24 else "later" if diff is not None and diff>24 else "earlier_or_other")
    lead=(raw-r["entry"])/3600
    bucket="<24h" if lead<24 else "24-72h" if lead<=72 else ">72h"
    inside=any(parse_time(ro["discharge_start"])<=r["entry"]<=parse_time(ro["discharge_end"])
               for ro in vessels.get(r.get("vessel_id",""),{}).get("rotations",[]))
    return {"etd_vs_s3":relation,"lead_bucket":bucket,"in_discharge_window":str(inside)}


def cell():
    return dict(hours=0.0,background_exits=0,truck_exits=0,off_s3_LOADs=0,
                rotation_opportunities=0,rotation_LOADs=0)


def finalize(row):
    return dict(row,alpha=row["rotation_LOADs"]/row["rotation_opportunities"] if row["rotation_opportunities"] else 0.0,
                mu_per_hour=row["background_exits"]/row["hours"] if row["hours"] else 0.0)


def fit_tables(layout,initial,events,records,schedule):
    model=MyStrategy()
    model.initialize(layout,initial)
    vessels={v["vessel_id"]:v for v in schedule["vessels"]}
    tables={g:cell() for g in GROUPS}
    sub=collections.defaultdict(cell)
    def keys(r,t,rotation):
        g=group_at(r,t,rotation)
        out=[tables[g]]
        if g in ("sched_disch_active","sched_truck_active"):
            f=features_at(r,rotation,vessels)
            for feature in FEATURES:
                if feature=="in_discharge_window" and g!="sched_disch_active": continue
                out.append(sub[(g,feature,f[feature])])
        return out
    for r in records.values():
        vessel=r.get("vessel_id","")
        a,b=r["entry"],r["end"]
        points={a,b}
        raw=parse_time(r.get("departure_time",""))
        if a<raw<b: points.add(raw)
        for rotation in model._rotations.get(vessel,()):
            if a<rotation.end<b: points.add(rotation.end)
        points=sorted(points)
        for start,end in zip(points,points[1:]):
            t=(start+end)/2
            ro=rotation_at(model,vessel,t)
            for row in keys(r,t,ro): row["hours"]+=(end-start)/3600
        if r["exit_type"] is not None:
            ro=rotation_at(model,vessel,b)
            own_s3=(r["exit_type"]=="LOAD" and ro is not None and ro.start<=b<=ro.end and not r["missed"])
            if not own_s3:
                for row in keys(r,b,ro):
                    row["background_exits"]+=1
                    row["truck_exits"]+=r["exit_type"]=="TRUCK_DLVR"
                    row["off_s3_LOADs"]+=r["exit_type"]=="LOAD"
        for ro in model._rotations.get(vessel,()):
            if ro.end>FIT_END or not a<=ro.start<=b: continue
            # All outcomes for this opportunity are observed through load_end.
            rows=keys(r,ro.start,ro)
            success=r["exit_type"]=="LOAD" and ro.start<=r["end"]<=ro.end
            for row in rows:
                row["rotation_opportunities"]+=1
                row["rotation_LOADs"]+=success
    final={g:finalize(row) for g,row in tables.items()}
    subfinal=[dict(cohort=g,feature=f,level=level,**finalize(row)) for (g,f,level),row in sorted(sub.items())]
    screens=[]
    for g in ("sched_disch_active","sched_truck_active"):
        for f in FEATURES:
            rows=[r for r in subfinal if r["cohort"]==g and r["feature"]==f]
            ar=[r["alpha"] for r in rows if r["rotation_opportunities"]>=25]
            mr=[r["mu_per_hour"] for r in rows if r["hours"]>=1000]
            alpha_span=max(ar)-min(ar) if len(ar)>1 else 0
            hazard_ratio=(max(mr)/min(mr) if min(mr)>0 else None) if len(mr)>1 else 1
            hazard_qualifies=len(mr)>1 and max(mr)>2*min(mr)
            screens.append(dict(cohort=g,feature=f,alpha_span=alpha_span,hazard_ratio=hazard_ratio,
                                zero_hazard_level=bool(len(mr)>1 and min(mr)==0),
                                qualifies=alpha_span>0.15 or hazard_qualifies))
    qualifying=[s for s in screens if s["qualifies"]]
    # One expanded feature only, chosen using fit data. Prefer alpha separation,
    # then finite hazard separation; deterministic feature/cohort ordering.
    best=max(qualifying,key=lambda s:(s["alpha_span"],s["hazard_ratio"] or 1e6)) if qualifying else None
    feature=best["feature"] if best else None
    parameters={g:[row["alpha"] if g.startswith("sched_") else 0.0,row["mu_per_hour"]] for g,row in final.items()}
    expanded={}
    if feature:
        for row in subfinal:
            if row["feature"]!=feature: continue
            g=row["cohort"]
            expanded[g+"|"+row["level"]]=[
                row["alpha"] if row["rotation_opportunities"] else parameters[g][0],
                row["mu_per_hour"] if row["hours"] else parameters[g][1]]
    return dict(cohorts=final,subfeatures=subfinal,screen=screens,expanded_feature=feature,
                parameters=parameters,expanded_parameters=expanded,
                screening_support="Alpha comparisons require >=25 opportunities/level; hazard comparisons require >=1000 container-hours/level. All raw counts also reported.",
                estimation_note="Alpha is the requested closed-window LOAD incidence fraction, not a joint latent-jump MLE corrected for competing background exits. Mu is exits/person-hours. Non-S3 means outside the contemporaneous next S3 loading window.")


def pair_audit(layout,initial,events,records):
    model=MyStrategy()
    model.initialize(layout,initial)
    live={c["container_id"]:records[c["container_id"]] for c in initial["containers"]}
    rng=random.Random(303)
    counts=[[0,0,0] for _ in BINS]
    shares=collections.defaultdict(collections.Counter)
    for e in events:
        now=parse_time(e.timestamp)
        model.on_event(e)
        if e.type in ARRIVALS:
            pairs=rng.sample(list(live),min(32,len(live)))
            incoming=records[e.container_id]
            rc=rotation_at(model,e.vessel_id,now)
            gc=group_at(incoming,now,rc)
            ck=model._incoming_priority(e,model._meta[e.container_id])
            for cid in pairs:
                r=live[cid]
                rx=rotation_at(model,r.get("vessel_id",""),now)
                gx=group_at(r,now,rx)
                tx,tc=r["exit_order"],incoming["exit_order"]
                known=tx!=float("inf") or tc!=float("inf")
                y=int(tx<tc)
                key=model._priority(model._meta[cid],model._next_by_vessel)
                j=bin_index((key-ck)/3600)
                counts[j][0]+=1
                counts[j][1]+=known
                counts[j][2]+=y
                if j in (1,7):
                    shares[BINS[j]+"/resident"][gx]+=1
                    shares[BINS[j]+"/incoming"][gc]+=1
            live[e.container_id]=incoming
        else: live.pop(e.container_id,None)
    rows=[dict(bin=b,n=n,known_n=known,resident_first=yes,horizon_rate=yes/n if n else None,
               conditional_rate=yes/known if known else None) for b,(n,known,yes) in zip(BINS,counts)]
    assert sum(r["n"] for r in rows)==218528
    assert sum(r["known_n"] for r in rows)==128693
    return dict(rows=rows,cohort_shares=dict(shares),
                note="All sampled pairs included; both-censored pairs contribute zero to the finite-Day12-cutoff indicator. This is not an uncensored lifetime-order probability.")


class LandingStrategy(MyStrategy):
    def initialize(self,layout,initial):
        super().initialize(layout,initial)
        self.last={c["container_id"]:{"source":"initial"} for c in initial["containers"]}
    def on_event(self,e):
        self.event=e
        super().on_event(e)
    def place_container(self,yard,event):
        p=super().place_container(yard,event)
        self.last[event.container_id]={"source":"strategy"}
        return p


class LandingSimulator(Simulator):
    def __init__(self,yard,strategy):
        super().__init__(yard,strategy)
        self.landings=collections.Counter()
    def _do_retrieval(self,target,above):
        for cid in above:
            landing=self.strategy.last[cid]
            if landing["source"]=="relocation":
                data=landing["underneath"][target]
                p="Days1-12" if self.strategy.current_time<FIT_END else "Days13-20"
                self.landings[(p,)+data]+=1
        super()._do_retrieval(target,above)
    def _find_reshuffle_position(self,block,container):
        pos=super()._find_reshuffle_position(block,container)
        if pos is not None:
            s=self.strategy
            height=pos.tier-1
            underneath={}
            for tier in range(1,pos.tier):
                cid=self.yard.get_container_at(pos.block,pos.bay,pos.row,tier)
                meta=s._meta[cid]
                r=dict(origin="initial" if cid in s._initial_ids else "DISCHARGE",
                       missed=cid in s.observed_missed_loads,departure_time=meta.departure)
                # group_at expects ISO input; use the already parsed metadata.
                if r["missed"]: g="missed_load"
                elif meta.vessel not in s._next_by_vessel: g="expired_no_rot" if meta.departure<s.current_time else "unscheduled_active"
                elif meta.departure<s.current_time: g="sched_overdue"
                else: g="active"
                underneath[cid]=(height,tier,"initial" if cid in s._initial_ids else "new",g)
            s.last[container.container_id]={"source":"relocation","underneath":underneath}
        return pos


def landing_audit():
    layout,initial,events,_=load_train_inputs()
    yard=YardState(layout)
    yard.load_initial_state(initial)
    strategy=LandingStrategy()
    strategy.initialize(layout,initial)
    sim=LandingSimulator(yard,strategy)
    stats=sim.run(events)
    assert stats.total_reshuffles==7026 and sum(sim.landings.values())==1750
    return [dict(period=p,height_before_landing=h,victim_tier=t,origin=o,state=g,count=n)
            for (p,h,t,o,g),n in sorted(sim.landings.items())]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    layout,initial,events,records,schedule=load_fit()
    fit=fit_tables(layout,initial,events,records,schedule)
    audit=dict(fit_cutoff_exclusive="2025-01-13T00:00:00",fit_events=len(events),fit=fit,
               pairwise=pair_audit(layout,initial,events,records))
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    audit["relocation_landings_full_train_context_only"]=landing_audit()
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("FIT",fit["cohorts"],flush=True)
    print("EXPANSION",fit["expanded_feature"],flush=True)


if __name__=="__main__": main()
