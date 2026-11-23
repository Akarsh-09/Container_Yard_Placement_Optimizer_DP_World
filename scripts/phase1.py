"""Reproducible, train-only diagnostics. Never imported by submission strategies.

Run from repository root: .venv/bin/python scripts/phase1.py
Outputs raw evidence to results/phase1/. No test files are opened.
"""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline_greedy import GreedyStrategy
from src.baseline_random import RandomStrategy
from src.event_reader import read_events
from src.placement_interface import PlacementStrategy
from src.simulator import Simulator
from src.yard_state import YardState

OUT = ROOT / "results/phase1"
START = dt.datetime(2025, 1, 1)
END = dt.datetime(2025, 1, 21)
SPLIT = dt.datetime(2025, 1, 13)
ARRIVALS = {"DISCHARGE", "TRUCK_RECV"}
WEIGHTS = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}


def stamp(s):
    return dt.datetime.fromisoformat(s) if s else None


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def quantiles(values):
    a = sorted(values)
    if not a:
        return {"n": 0}
    result = {"n": len(a), "min": a[0], "max": a[-1]}
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        x = (len(a) - 1) * p / 100
        i = int(x)
        result[f"p{p}"] = a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (x - i)
    return result


def hour_delta(a, b):
    return (a - b).total_seconds() / 3600


def slot_key(p):
    return p.block, p.bay, p.row


def initial_bound(initial, records):
    """Certified lower bound: initial blockers above each stack's first target.

    Before the first retrieval of an initial container from an original stack,
    incoming-container retrievals cannot move any of its initial containers.
    Every initial container above that first target must therefore be moved.
    Count only this first retrieval per original stack to avoid double counting.
    """
    stacks = collections.defaultdict(list)
    for c in initial["containers"]:
        p = c["position"]
        stacks[p["block"], p["bay"], p["row"]].append(c)
    targets = {}
    for stack in stacks.values():
        fetched = [c for c in stack if records[c["container_id"]]["retrieval"] is not None]
        if fetched:
            first = min(fetched, key=lambda c: records[c["container_id"]]["retrieval"])
            targets[first["container_id"]] = sum(c["position"]["tier"] > first["position"]["tier"] for c in stack)
    return targets


def analyse(initial, events, schedule):
    vessels = {v["vessel_id"]: v for v in schedule["vessels"]}
    records = {}
    active = {}
    found = []
    absent = []
    for c in initial["containers"]:
        r = dict(c, origin="initial", arrival=START, retrieval=None, exit=None)
        records[c["container_id"]] = active[c["container_id"]] = r
    arrival_events = {e.container_id: e for e in events if e.type in ARRIVALS}
    for e in events:
        if e.type in ARRIVALS:
            r = dict(vars(e), origin=e.type, arrival=stamp(e.timestamp), retrieval=None, exit=None)
            records[e.container_id] = active[e.container_id] = r
        else:
            r = active.pop(e.container_id, None)
            if r:
                r.update(retrieval=stamp(e.timestamp), exit=e.type)
                found.append((e, r))
            else:
                later = arrival_events.get(e.container_id)
                absent.append({"event_id": e.event_id, "container_id": e.container_id,
                               "timestamp": e.timestamp, "type": e.type,
                               "arrival_timestamp": later.timestamp if later else None,
                               "later_arrival": bool(later and later.event_id > e.event_id)})

    def rotation_for(r):
        rotations = vessels.get(r["vessel_id"], {}).get("rotations", [])
        departure = stamp(r.get("departure_time"))
        if not rotations or departure is None:
            return None
        # Match using only known metadata, never actual retrieval time.
        return min(rotations, key=lambda x: abs((stamp(x["etd"]) - departure).total_seconds()))

    errors = collections.defaultdict(list)
    aligned = collections.defaultdict(list)
    tails = []
    for e, r in found:
        departure = stamp(r.get("departure_time"))
        if departure is None:
            continue
        err = hour_delta(r["retrieval"], departure)
        errors[f'{r["origin"]}/{e.type}'].append(err)
        rotation = rotation_for(r)
        if e.type == "LOAD" and rotation:
            midpoint = stamp(rotation["load_start"]) + (stamp(rotation["load_end"]) - stamp(rotation["load_start"])) / 2
            aligned[r["origin"]].append(hour_delta(r["retrieval"], midpoint))
        if abs(err) > 24:
            tails.append({"container_id": e.container_id, "origin": r["origin"], "exit": e.type,
                          "vessel_id": r["vessel_id"], "error_hours": err,
                          "multi_rotation": len(vessels.get(r["vessel_id"], {}).get("rotations", [])) > 1})
    integrity = collections.defaultdict(lambda: collections.Counter(total=0, missing_departure=0, past_at_arrival_or_start=0))
    for r in records.values():
        integrity[r["origin"]]["total"] += 1
        departure = stamp(r.get("departure_time"))
        if departure is None:
            integrity[r["origin"]]["missing_departure"] += 1
        elif departure < r["arrival"]:
            integrity[r["origin"]]["past_at_arrival_or_start"] += 1
    tail_summary = {
        "n": len(tails), "by_origin": dict(collections.Counter(x["origin"] for x in tails)),
        "by_exit": dict(collections.Counter(x["exit"] for x in tails)),
        "multi_rotation": sum(x["multi_rotation"] for x in tails),
        "truck_only_vessel": sum(x["vessel_id"] in {"VSL019", "VSL020"} for x in tails),
        "worst_20": sorted(tails, key=lambda x: abs(x["error_hours"]), reverse=True)[:20],
    }
    save("etd.json", {"raw_error_hours": {k: quantiles(v) for k, v in errors.items()},
                      "schedule_midpoint_error_hours_by_origin_LOAD": {k: quantiles(v) for k, v in aligned.items()},
                      "integrity": integrity, "tails_abs_over_24h": tail_summary})

    initial_stacks = collections.defaultdict(list)
    for c in initial["containers"]:
        p = c["position"]
        initial_stacks[p["block"], p["bay"], p["row"]].append(c)
    inversions = 0
    stacks_with_inversions = 0
    for stack in initial_stacks.values():
        stack.sort(key=lambda c: c["position"]["tier"])
        count = 0
        for i, lower in enumerate(stack):
            rt = records[lower["container_id"]]["retrieval"]
            for upper in stack[i+1:]:
                ut = records[upper["container_id"]]["retrieval"]
                count += bool(rt is not None and (ut is None or rt < ut))
        inversions += count
        stacks_with_inversions += count > 0
    overdue = []
    for hours in (24, 48, 72):
        for origin in ("all", "initial", "DISCHARGE", "TRUCK_RECV"):
            eligible = []
            for r in records.values():
                if origin != "all" and r["origin"] != origin:
                    continue
                departure = stamp(r.get("departure_time"))
                if departure is None:
                    continue
                landmark = max(r["arrival"], departure + dt.timedelta(hours=hours))
                if landmark < END and (r["retrieval"] is None or r["retrieval"] > landmark):
                    eligible.append((r, landmark))
            retrieved = [(r, t) for r, t in eligible if r["retrieval"] is not None]
            overdue.append({"threshold_hours": hours, "origin": origin, "at_risk_n": len(eligible),
                            "later_retrieved_by_end": len(retrieved),
                            "unretrieved_at_end_censored": len(eligible)-len(retrieved),
                            "followup_hours": quantiles([hour_delta(END, t) for r, t in eligible]),
                            "later_wait_hours_observed": quantiles([hour_delta(r["retrieval"], t) for r, t in retrieved])})
    save("initial_overdue.json", {"initial_retrieved": sum(r["origin"] == "initial" and r["retrieval"] is not None for r in records.values()),
                                 "initial_unretrieved": sum(r["origin"] == "initial" and r["retrieval"] is None for r in records.values()),
                                 "initial_inversion_pairs_horizon_censored": inversions,
                                 "initial_stacks_with_inversions": stacks_with_inversions,
                                 "certified_first_initial_retrieval_lower_bound":sum(initial_bound(initial,records).values()),
                                 "lower_bound_target_count":len(initial_bound(initial,records)),
                                 "lower_bound_proof":initial_bound.__doc__,
                                 "overdue_landmark_analysis": overdue,
                                 "warning": "Unretrieved is right-censored at Jan 21; inversion pairs are NOT an achievable cost lower bound."})

    contingency = collections.defaultdict(collections.Counter)
    for r in records.values():
        vessel = r.get("vessel_id", "")
        vessel_class = "empty" if not vessel else "scheduled" if vessels.get(vessel, {}).get("rotations") else "unscheduled"
        departure = stamp(r.get("departure_time"))
        lead = hour_delta(departure, r["arrival"]) if departure else None
        bucket = "missing" if lead is None else "past" if lead < 0 else "0-24h" if lead < 24 else "24-72h" if lead < 72 else "72-168h" if lead < 168 else ">=168h"
        feature = (r["origin"], vessel_class, "port_present" if r.get("port_of_discharge") else "port_empty", bucket)
        contingency["/".join(feature)][r["exit"] or "unretrieved_censored"] += 1
    save("exit_contingency.json", dict(contingency))

    def describe(e, r):
        v = vessels.get(r["vessel_id"], {})
        ro = rotation_for(r)
        windows = [x["rotation_number"] for x in v.get("rotations", [])
                   if stamp(x["load_start"]) <= stamp(e.timestamp) <= stamp(x["load_end"])]
        return {"event_id": e.event_id, "timestamp": e.timestamp, "container_id": e.container_id,
                "origin": r["origin"], "port": r["port_of_discharge"], "weight": r["weight_class"],
                "stored_etd": r["departure_time"], "vessel": r["vessel_id"],
                "metadata_rotation": ro["rotation_number"] if ro else None,
                "actual_window_rotations": windows}

    stored_groups = collections.defaultdict(list)
    actual_groups = collections.defaultdict(list)
    mismatched_rotation = 0
    for e, r in found:
        if e.type != "LOAD":
            continue
        d = describe(e, r)
        stored_groups[r["vessel_id"], r["departure_time"]].append(d)
        actual_groups[r["vessel_id"], tuple(d["actual_window_rotations"])].append(d)
        mismatched_rotation += d["metadata_rotation"] not in d["actual_window_rotations"]
    restarts = []
    for key, sequence in stored_groups.items():
        for i, (a, b) in enumerate(zip(sequence, sequence[1:])):
            if a["port"] > b["port"]:
                restarts.append({"group": key, "before": a, "after": b,
                                 "context": sequence[max(0,i-1):i+3]})
    order_stats = []
    for (vessel, windows), seq in actual_groups.items():
        ports = vessels[vessel]["ports"]
        rank = {p: i for i, p in enumerate(ports)}
        order_stats.append({"vessel": vessel, "actual_windows": windows, "n": len(seq),
                            "lex_port_decreases": sum(a["port"] > b["port"] for a,b in zip(seq,seq[1:])),
                            "schedule_forward_decreases": sum(rank.get(a["port"],999) > rank.get(b["port"],999) for a,b in zip(seq,seq[1:])),
                            "schedule_reverse_decreases": sum(rank.get(a["port"],999) < rank.get(b["port"],999) for a,b in zip(seq,seq[1:])),
                            "same_port_weight_reversals": sum(a["port"]==b["port"] and WEIGHTS[a["weight"]]>WEIGHTS[b["weight"]] for a,b in zip(seq,seq[1:]))})
    all_load_order = collections.defaultdict(list)
    event_vs_container_vessel_mismatches = 0
    for e in events:
        if e.type != "LOAD":
            continue
        v = vessels[e.vessel_id]
        windows = tuple(x["rotation_number"] for x in v["rotations"]
                        if stamp(x["load_start"]) <= stamp(e.timestamp) <= stamp(x["load_end"]))
        all_load_order[e.vessel_id, windows].append(e)
        event_vs_container_vessel_mismatches += e.vessel_id != records[e.container_id]["vessel_id"]
    all_order = []
    for (vessel, windows), seq in all_load_order.items():
        all_order.append({"vessel":vessel, "windows":windows, "n":len(seq),
                          "lex_port_decreases":sum(a.port_of_discharge>b.port_of_discharge for a,b in zip(seq,seq[1:])),
                          "same_port_weight_reversals":sum(a.port_of_discharge==b.port_of_discharge and WEIGHTS[a.weight_class]>WEIGHTS[b.weight_class] for a,b in zip(seq,seq[1:]))})
    matched_errors = collections.defaultdict(list)
    for e,r in found:
        if e.type != "LOAD":
            continue
        rotation = rotation_for(r)
        matched = stamp(rotation["load_start"]) <= r["retrieval"] <= stamp(rotation["load_end"])
        midpoint = stamp(rotation["load_start"]) + (stamp(rotation["load_end"])-stamp(rotation["load_start"])) / 2
        matched_errors[f'{"matched" if matched else "mismatched"}/raw'].append(hour_delta(r["retrieval"],stamp(r["departure_time"])))
        matched_errors[f'{"matched" if matched else "mismatched"}/midpoint'].append(hour_delta(r["retrieval"],midpoint))
    save("load_order.json", {"stored_group_restarts": restarts, "actual_window_order_stats": order_stats,
                             "all_LOAD_including_absent_order":all_order,
                             "event_vs_container_vessel_mismatches":event_vs_container_vessel_mismatches,
                             "errors_by_rotation_match":{k:quantiles(v) for k,v in matched_errors.items()},
                             "found_LOAD_metadata_rotation_mismatch": mismatched_rotation,
                             "note": "Actual windows use event vessel and event time only for offline diagnosis; metadata mapping uses stored ETD."})
    save("absent.json", {"n": len(absent), "later_arrival": sum(x["later_arrival"] for x in absent),
                         "later_valid_retrieval_by_end":sum(records[x["container_id"]]["retrieval"] is not None for x in absent),
                         "no_train_arrival": sum(x["arrival_timestamp"] is None for x in absent), "events": absent})
    return records


class ObservedBaseline(PlacementStrategy):
    """Observers collect evidence; delegate all placement decisions unchanged."""
    def __init__(self, base, records, scan_samples=0):
        self.base = base
        self.records = records
        self.buckets = collections.defaultdict(lambda: collections.Counter(retrievals=0, found=0, reshuffles=0))
        self.moved_before = set()
        self.rehandled_blocker_moves = 0
        self.scan_samples = scan_samples
        self.scans = collections.defaultdict(list)
        self.containers_seen = []

    def initialize(self, layout, initial):
        self.base.initialize(layout, initial)
        self.bound_targets = initial_bound(initial,self.records)
        self.bound_checked = 0

    def on_event(self, event):
        self.event = event
        self.base.on_event(event)
        if event.type not in ARRIVALS:
            self.buckets_for_event = ["all", f"exit/{event.type}",
                                     "period/1-12" if stamp(event.timestamp) < SPLIT else "period/13-20",
                                     f"five_day/{(stamp(event.timestamp).day-1)//5*5+1}-{(stamp(event.timestamp).day-1)//5*5+5}"]
            origin = self.records.get(event.container_id, {}).get("origin", "absent_exogenous")
            self.buckets_for_event += [f"origin/{origin}", f"exit_origin/{event.type}/{origin}"]
            for k in self.buckets_for_event:
                self.buckets[k]["retrievals"] += 1

    def place_container(self, yard, event):
        if len(self.scans["public_stack_scan_sec"]) < self.scan_samples:
            if len(self.containers_seen) % 2 == 0:
                self.scan_stack(yard)
                self.scan_dict(yard)
            else:
                self.scan_dict(yard)
                self.scan_stack(yard)
            self.containers_seen.append(yard.total_containers())
        return self.base.place_container(yard, event)

    def scan_stack(self, yard):
        start = time.perf_counter()
        count = 0
        for block, bi in yard.blocks.items():
            for bay in range(1, bi.bays+1):
                for row in range(1, bi.rows+1):
                    h = yard.get_stack_height(block,bay,row)
                    for tier in range(1,h+1):
                        cid = yard.get_container_at(block,bay,row,tier)
                        info = yard.get_container_info(cid)
                        count += bool(info.departure_time)
        self.scans["public_stack_scan_sec"].append(time.perf_counter()-start)
        return count

    def scan_dict(self, yard):
        start = time.perf_counter()
        all_containers = yard.get_all_containers()
        copied = time.perf_counter()
        grouped = collections.defaultdict(list)
        for cid, (pos, info) in all_containers.items():
            grouped[slot_key(pos)].append((pos.tier, info.departure_time))
        for stack in grouped.values():
            stack.sort()
        self.scans["get_all_copy_sec"].append(copied-start)
        self.scans["get_all_and_group_sec"].append(time.perf_counter()-start)

    def on_container_retrieved(self, cid, pos, reshuffles):
        for k in self.buckets_for_event:
            self.buckets[k]["found"] += 1
            self.buckets[k]["reshuffles"] += reshuffles
        self.base.on_container_retrieved(cid, pos, reshuffles)


class DiagnosticSimulator(Simulator):
    """Instrumentation only: use original simulator for every yard mutation."""
    def _do_retrieval(self, target_id, above):
        if target_id in self.strategy.bound_targets:
            actual_initial_above = sum(self.strategy.records[cid]["origin"] == "initial" for cid in above)
            assert actual_initial_above >= self.strategy.bound_targets[target_id]
            self.strategy.bound_checked += 1
        self.strategy.rehandled_blocker_moves += sum(cid in self.strategy.moved_before for cid in above)
        self.strategy.moved_before.update(above)
        super()._do_retrieval(target_id, above)


def benchmark(layout, initial, events, records):
    results = []
    for name, seed in [("greedy", None)] + [("random", s) for s in range(5)]:
        random.seed(seed if seed is not None else 0)
        yard = YardState(layout)
        yard.load_initial_state(initial)
        observer = ObservedBaseline(GreedyStrategy() if name == "greedy" else RandomStrategy(), records)
        observer.initialize(layout, initial)
        start = time.perf_counter()
        sim = DiagnosticSimulator(yard, observer)
        stats = sim.run(events)
        assert observer.bound_checked == len(observer.bound_targets)
        assert stats.total_reshuffles >= sum(observer.bound_targets.values())
        assert stats.errors == stats.hard_constraint_violations == stats.placement_failures == 0
        assert observer.buckets["all"]["reshuffles"] == stats.total_reshuffles
        assert observer.buckets["all"]["found"] + stats.retrieval_not_found == stats.retrieval_events
        wall = time.perf_counter()-start
        slices = {}
        for k,v in observer.buckets.items():
            slices[k] = dict(v, rate=v["reshuffles"]/v["retrievals"] if v["retrievals"] else 0)
        result = {"strategy": name, "seed": seed, "summary": stats.summary(),
                  "retrieval_not_found": stats.retrieval_not_found, "wall_sec": wall, "slices": slices,
                  "rehandled_blocker_moves": observer.rehandled_blocker_moves,
                  "unique_blockers_ever_moved": len(observer.moved_before)}
        results.append(result)
        save(f"baseline_{name}{'' if seed is None else '_'+str(seed)}.json", result)
        print(name,seed,stats.total_reshuffles,round(stats.reshuffles_per_retrieval,6),round(wall,2),flush=True)
    randoms = results[1:]
    aggregates = {}
    for key in randoms[0]["summary"]:
        values = [r["slices"]["all"]["rate"] if key == "reshuffles_per_retrieval" else r["summary"][key] for r in randoms]
        aggregates[key] = {"mean": statistics.mean(values), "std_population": statistics.pstdev(values),
                           "min": min(values), "max": max(values)}
    slice_aggregates = {}
    for key in randoms[0]["slices"]:
        slice_aggregates[key] = {}
        for metric in ("retrievals","found","reshuffles","rate"):
            values = [r["slices"][key][metric] for r in randoms]
            slice_aggregates[key][metric] = {"mean":statistics.mean(values),"std_population":statistics.pstdev(values),"min":min(values),"max":max(values)}
    save("baselines.json", {"runs": results, "random_summary_aggregate": aggregates,
                            "random_slice_aggregate":slice_aggregates})

    # Distributed samples: 50 evenly spaced arrivals per day, via observer selection.
    sampled_ids = set()
    by_day = collections.defaultdict(list)
    for e in events:
        if e.type in ARRIVALS:
            by_day[e.timestamp[:10]].append(e.event_id)
    for ids in by_day.values():
        sampled_ids.update(ids[round(i*(len(ids)-1)/49)] for i in range(50))
    class Sampled(ObservedBaseline):
        def place_container(self, yard, event):
            self.scan_samples = 1000 if event.event_id in sampled_ids else 0
            return super().place_container(yard,event)
    yard = YardState(layout)
    yard.load_initial_state(initial)
    observer = Sampled(GreedyStrategy(), records)
    observer.initialize(layout,initial)
    stats = Simulator(yard,observer).run(events)
    assert stats.total_reshuffles == results[0]["summary"]["total_reshuffles"]
    save("scan_benchmark.json", {"sampled_arrivals":len(observer.containers_seen),
                                 "inventory":quantiles(observer.containers_seen),
                                 "timings_seconds":{k:quantiles(v) for k,v in observer.scans.items()},
                                 "estimated_added_seconds_10385_arrivals":{k:statistics.mean(v)*10385 for k,v in observer.scans.items()},
                                 "note":"1,000 evenly spaced arrival samples, 50/day; alternating method order. Includes all stacks, including full ones. Dict grouping excludes explicit empty-stack enumeration."})


def write_chart():
    b = json.loads((OUT / "baselines.json").read_text())
    greedy = b["runs"][0]
    rows = []
    for label, key in [("Full train", "all"), ("Days 1–12", "period/1-12"), ("Days 13–20", "period/13-20")]:
        g = greedy["slices"][key]["rate"]
        r = b["random_slice_aggregate"][key]["rate"]
        rows.append((label,g,r["mean"],r["std_population"]))
    marks = []
    for i,(label,g,r,s) in enumerate(rows):
        y = 30+i*95
        marks.append(f'<text x="0" y="{y}" class="label">{label}</text>')
        for j,value in enumerate((g,r)):
            yy=y+10+j*23
            marks.append(f'<rect x="125" y="{yy}" width="{value*360:.2f}" height="17" rx="3" class="s{j}"/>')
            marks.append(f'<text x="{132+value*360:.2f}" y="{yy+13}" class="value">{value:.4f}</text>')
            if j==1:
                lo,hi=125+(value-s)*360,125+(value+s)*360
                marks.append(f'<path d="M{lo} {yy+8}H{hi} M{lo} {yy+3}V{yy+13} M{hi} {yy+3}V{yy+13}" class="whisker"/>')
    html = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
    body{color:var(--foreground);font-family:var(--font-sans);margin:0}.legend{display:flex;gap:18px;font-size:13px;margin-bottom:8px}.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}.s0,.g{fill:var(--chart-1);background:var(--chart-1)}.s1,.r{fill:var(--chart-2);background:var(--chart-2)}svg{width:100%;height:310px}.label,.value{fill:var(--foreground);font-size:13px}.value{font-family:var(--font-mono)}.whisker{stroke:var(--foreground);stroke-width:1.5;fill:none}.note{color:var(--muted-foreground);font-size:12px;line-height:1.5;margin:0}
    </style></head><body><div class="legend"><span><i class="key g"></i>Greedy</span><span><i class="key r"></i>Random · mean ± 1 SD, seeds 0–4</span></div><svg viewBox="0 0 620 310" role="img" aria-label="Reshuffles per retrieval: full train Greedy 0.7873, Random 0.8788; days 1 to 12 Greedy 0.6498, Random 0.7383; days 13 to 20 Greedy 1.0154, Random 1.1119">'''+''.join(marks)+'''</svg><p class="note">Reshuffles per scheduled retrieval; lower is better. Absent targets remain in the denominator. The initial-yard lower bound is 0.1298 for full train.</p></body></html>'''
    (OUT / "benchmark.html").write_text(html)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT/"data/yard_layout.json", ROOT/"data/train/initial_state.json",
              ROOT/"data/train/events.jsonl", ROOT/"data/vessel_schedule.json"]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    layout = json.loads(inputs[0].read_text())
    initial = json.loads(inputs[1].read_text())
    events = read_events(str(inputs[2]))
    schedule = json.loads(inputs[3].read_text())
    records = analyse(initial,events,schedule)
    benchmark(layout,initial,events,records)
    write_chart()
    save("provenance.json", {"python":sys.version, "input_sha256":hashes,
                             "command":".venv/bin/python scripts/phase1.py", "random_seeds":list(range(5)),
                             "test_data_opened":False, "split":"Fit through Jan 12; holdout Jan 13-20. This phase fits no models."})


if __name__ == "__main__":
    main()
