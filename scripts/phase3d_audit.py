"""Fit-only remaining-feature and repeat-rotation audit; no Bayes ceiling claim."""
import collections
import json
from math import sqrt
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase3c_audit import load_fit, rotation_at, group_at, features_at, cell, finalize, FIT_END
from solution.my_strategy import MyStrategy, parse_time

OUT=ROOT/"results/phase3d"
ACTIVE=("sched_init_active","sched_disch_active","sched_truck_active")


def age_bucket(hours):
    return "<24h" if hours<24 else "24-72h" if hours<72 else "72-168h" if hours<168 else ">=168h"


def spread(rows,value,weight,min_support):
    rows=[r for r in rows if r[weight]>=min_support]
    if not rows: return dict(levels=0)
    total=sum(r[weight] for r in rows)
    mean=sum(r[value]*r[weight] for r in rows)/total
    return dict(levels=len(rows),minimum=min(r[value] for r in rows),maximum=max(r[value] for r in rows),
                weighted_mean=mean,weighted_std=sqrt(sum(r[weight]*(r[value]-mean)**2 for r in rows)/total),
                support=total)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    layout,initial,events,records,schedule=load_fit()
    model=MyStrategy()
    model.initialize(layout,initial)
    vessels={v["vessel_id"]:v for v in schedule["vessels"]}
    counters=collections.defaultdict(cell)
    second=collections.Counter()
    by_origin=collections.defaultdict(collections.Counter)
    def targets(r,t,rotation):
        g=group_at(r,t,rotation)
        if g not in ACTIVE: return []
        f=features_at(r,rotation,vessels)
        f.update(vessel_id=r.get("vessel_id",""),port_of_discharge=r.get("port_of_discharge",""),
                 resident_age=age_bucket((t-r["entry"])/3600))
        return [counters[(g,key,value)] for key,value in f.items()
                if key in ("etd_vs_s3","vessel_id","port_of_discharge","resident_age")
                or (key=="in_discharge_window" and g=="sched_disch_active")]
    for r in records.values():
        a,b=r["entry"],r["end"]
        vessel=r.get("vessel_id","")
        rotations=model._rotations.get(vessel,())
        raw=parse_time(r.get("departure_time",""))
        points={a,b}
        for t in [raw]+[ro.end for ro in rotations]+[a+h*3600 for h in (24,72,168)]:
            if a<t<b: points.add(t)
        points=sorted(points)
        for start,end in zip(points,points[1:]):
            t=(start+end)/2
            for row in targets(r,t,rotation_at(model,vessel,t)):row["hours"]+=(end-start)/3600
        if r["exit_type"] is not None:
            ro=rotation_at(model,vessel,b)
            own=r["exit_type"]=="LOAD" and ro is not None and ro.start<=b<=ro.end and not r["missed"]
            if not own:
                for row in targets(r,b,ro):
                    row["background_exits"]+=1
                    row["truck_exits"]+=r["exit_type"]=="TRUCK_DLVR"
                    row["off_s3_LOADs"]+=r["exit_type"]=="LOAD"
        for ro in rotations:
            if ro.end>FIT_END or not a<=ro.start<=b: continue
            for row in targets(r,ro.start,ro):
                row["rotation_opportunities"]+=1
                row["rotation_LOADs"]+=r["exit_type"]=="LOAD" and ro.start<=b<=ro.end
        first=rotation_at(model,vessel,a)
        if first is None or first.end>FIT_END or b<=first.end: continue
        index=rotations.index(first)+1
        if index>=len(rotations): continue
        nxt=rotations[index]
        if nxt.end>FIT_END or not a<=nxt.start<=b: continue
        for row in (second,by_origin[r["origin"]]):
            row["opportunities"]+=1
            row["LOADs"]+=r["exit_type"]=="LOAD" and nxt.start<=b<=nxt.end
    rows=[dict(cohort=g,feature=f,level=v,**finalize(c)) for (g,f,v),c in sorted(counters.items())]
    summaries=[]
    for group in ACTIVE:
        for feature in ("etd_vs_s3","in_discharge_window","vessel_id","port_of_discharge","resident_age"):
            values=[r for r in rows if r["cohort"]==group and r["feature"]==feature]
            if not values: continue
            alpha=spread(values,"alpha","rotation_opportunities",25)
            mu=spread(values,"mu_per_hour","hours",1000)
            summaries.append(dict(cohort=group,feature=feature,alpha=alpha,mu=mu,
                                  alpha_span=(alpha.get("maximum",0)-alpha.get("minimum",0)),
                                  vessel_specific=feature=="vessel_id"))
    result=dict(fit_cutoff_exclusive="2025-01-13T00:00:00",rows=rows,summary=summaries,
                second_rotation=dict(second,alpha=second["LOADs"]/second["opportunities"] if second["opportunities"] else None),
                second_rotation_by_origin={k:dict(v,alpha=v["LOADs"]/v["opportunities"] if v["opportunities"] else None) for k,v in by_origin.items()},
                notes=["Initial resident age is observed age since the initial snapshot, not unknown true dwell age.",
                       "Weighted spread is support-weighted standard deviation (opportunities for alpha, container-hours for mu).",
                       "Feature heterogeneity does not identify a Bayes predictability ceiling; port/vessel associations may be confounded.",
                       "Second opportunities require the full second loading window before the fit cutoff and presence at its start."])
    (OUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("SECOND",result["second_rotation"],flush=True)
    print("NON-ID ALPHA SPLITS >=.25",[(s["cohort"],s["feature"],s["alpha_span"]) for s in summaries if not s["vessel_specific"] and s["alpha_span"]>=.25],flush=True)


if __name__=="__main__":main()
