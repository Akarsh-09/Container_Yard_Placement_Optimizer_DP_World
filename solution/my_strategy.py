"""Online expected-blocking placement with retained LDT reference ablations.

Inspired by Borgman, van Asperen & Dekker (2010), Online rules for container
stacking, https://doi.org/10.1007/s00291-010-0205-4. Full-stack compatibility,
strict tie handling and schedule variants are adaptations for this simulator.
Only the supplied schedule, current event, initial state and live yard are read.
"""
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import heapq
import json
from math import exp, expm1, isfinite
from pathlib import Path
from typing import Optional

from src.models import Position
from src.placement_interface import PlacementStrategy
from .survival_params import SURVIVAL_PARAMETERS, EXPANSION_FEATURE, EXPANDED_PARAMETERS


MODES = ("raw_strict", "raw_nonstrict", "raw_top_only", "s1_stored_order",
         "s2_rollover", "s3_next_feasible", "expected_blocking", "survival_risk")
WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
UNKNOWN_TIME = 253402300799.0
INFINITY = float("inf")


def survival_block_probability(alpha_x, mu_x, tau_x, alpha_c, mu_c, tau_c,
                               horizon_hours=INFINITY, rotation_sigma_hours=0.0,
                               discount_rate=0.0):
    """Independent exponential backgrounds with one Bernoulli jump each.

    All times/rates use hours. The background integral is the algebraic expansion
    of the <=3 piecewise exponential intervals. Coin-flip ordering handles equal
    jump times. sigma>0 softens only jump ordering and is an approximation to the
    exact step-survival model (it does not randomize the continuous survival).
    """
    rate = mu_x + mu_c + discount_rate
    jump_x = alpha_x > 0 and isfinite(tau_x) and tau_x <= horizon_hours
    jump_c = alpha_c > 0 and isfinite(tau_c) and tau_c <= horizon_hours
    ex = exp(-rate * tau_x) if jump_x and rate else 1.0
    ec = exp(-rate * tau_c) if jump_c and rate else 1.0
    background = 0.0
    if mu_x > 0:
        end_survival = exp(-rate * horizon_hours) if isfinite(horizon_hours) else 0.0
        integral = -expm1(-rate * horizon_hours) if isfinite(horizon_hours) else 1.0
        if jump_x: integral -= alpha_x * (ex - end_survival)
        if jump_c: integral -= alpha_c * (ec - end_survival)
        if jump_x and jump_c:
            integral += alpha_x * alpha_c * (min(ex, ec) - end_survival)
        background = mu_x / rate * integral
    atom = 0.0
    if jump_x:
        precedes = 0.0
        if alpha_c > 0 and isfinite(tau_c):
            delta = tau_x - tau_c
            if rotation_sigma_hours:
                precedes = 1.0 / (1.0 + exp(max(-30.0,min(30.0,-delta/rotation_sigma_hours))))
            else:
                precedes = 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
        atom = alpha_x * ex * (1.0 - alpha_c * precedes)
    return min(1.0,max(0.0,background + atom))


def multi_survival_block_probability(mu_x, jumps_x, mu_c, jumps_c,
                                     horizon_hours=INFINITY, discount_rate=0.0):
    """Exact piecewise integration for conditional Bernoulli rotation jumps.

    jumps are sorted (hours-from-now, conditional jump probability) tuples.
    At most four public calls per container give at most eight breakpoints.
    """
    xs=tuple((t,a) for t,a in jumps_x if a>0 and isfinite(t) and t<=horizon_hours)
    cs=tuple((t,a) for t,a in jumps_c if a>0 and isfinite(t) and t<=horizon_hours)
    if len(xs)<=1 and len(cs)<=1:
        tx,ax=xs[0] if xs else (INFINITY,0.0)
        tc,ac=cs[0] if cs else (INFINITY,0.0)
        return survival_block_probability(ax,mu_x,tx,ac,mu_c,tc,horizon_hours,0,discount_rate)
    events={}
    for side,jumps in enumerate((xs,cs)):
        for t,alpha in jumps:
            pair=events.setdefault(t,[0.0,0.0])
            pair[side]=1-(1-pair[side])*(1-alpha)
    rate=mu_x+mu_c+discount_rate
    fx=fc=1.0
    previous_exp=1.0
    probability=0.0
    for t,(ax,ac) in sorted(events.items()):
        at=exp(-rate*t) if rate else 1.0
        if mu_x:
            probability+=mu_x/rate*fx*fc*(previous_exp-at)
        probability+=fx*ax*at*fc*(1-.5*ac)
        fx*=1-ax
        fc*=1-ac
        previous_exp=at
    if mu_x:
        end=exp(-rate*horizon_hours) if isfinite(horizon_hours) else 0.0
        probability+=mu_x/rate*fx*fc*(previous_exp-end)
    return min(1.0,max(0.0,probability))


def parse_time(value):
    """Interpret timezone-free terminal timestamps consistently as UTC."""
    if not value:
        return UNKNOWN_TIME
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError):
        return UNKNOWN_TIME


@dataclass(frozen=True)
class Rotation:
    etd: float
    start: float
    end: float


@dataclass(frozen=True)
class ContainerMeta:
    departure: float
    vessel: str
    port: str
    weight_rank: int
    fraction: float
    matched_rotation: Optional[Rotation]
    s1_time: float


class MyStrategy(PlacementStrategy):
    """Expected-blocking policy selected on Days 1-12; LDT modes remain usable.

    Public-API stack contents refresh only when dirtied by arrivals/retrievals.
    Effective priority changes are processed at public schedule end transitions.
    Scoring costs O(S*H) per placement, plus dirty refresh and transition work.
    Storage is O(N + S + schedule size + 128*(K+S)), where K counts distinct
    resident priority values in retained tables. At most 128 incoming-priority
    risk/stack-score tables are retained. use_cache=False fully refreshes
    public stack contents each placement as an independent synchronization check.
    """
    def __init__(self, mode="expected_blocking", schedule_path=None,
                 timing_key="s3_next_feasible", epsilon=0.05, sigma_hours=12,
                 lambda_empty=0.0, lambda_gap=0.0, use_cache=True,
                 initial_epsilon=None, expired_overdue_prob=None,
                 all_overdue_prob=None, intra_rotation_sigma_hours=None,
                 missed_load_offset_hours=None, survival_horizon_hours=240,
                 survival_rotation_sigma_hours=0.0, use_expanded_survival=False,
                 survival_parameters=None, survival_discount_hours=None,
                 use_multi_rotation=False):
        if mode not in MODES:
            raise ValueError(f"Unknown strategy mode: {mode}")
        if timing_key not in ("raw", "s2_rollover", "s3_next_feasible"):
            raise ValueError("Unknown timing key")
        if not 0 <= epsilon <= 0.5 or not sigma_hours > 0:
            raise ValueError("Require 0 <= epsilon <= 0.5 and sigma_hours > 0")
        if lambda_empty < 0 or lambda_gap < 0:
            raise ValueError("Opportunity costs must be nonnegative")
        if initial_epsilon is not None and not 0 <= initial_epsilon <= 0.5:
            raise ValueError("initial_epsilon must lie in [0, 0.5]")
        for probability in (expired_overdue_prob, all_overdue_prob):
            if probability is not None and not 0 <= probability <= 1:
                raise ValueError("Overdue probabilities must lie in [0, 1]")
        if intra_rotation_sigma_hours is not None and not intra_rotation_sigma_hours > 0:
            raise ValueError("Intra-rotation scale must be positive")
        if missed_load_offset_hours is not None and not missed_load_offset_hours >= 0:
            raise ValueError("Missed-LOAD offset must be nonnegative or infinity")
        if not survival_horizon_hours > 0 or not survival_rotation_sigma_hours >= 0:
            raise ValueError("Survival horizon must be positive and smoothing nonnegative")
        if survival_discount_hours is not None and not survival_discount_hours > 0:
            raise ValueError("Discount scale must be positive")
        if use_multi_rotation and survival_rotation_sigma_hours:
            raise ValueError("Multi-rotation integration uses exact jump boundaries")
        self.survival_horizon_hours = survival_horizon_hours
        self.survival_rotation_sigma_hours = survival_rotation_sigma_hours
        self.use_expanded_survival = use_expanded_survival
        self.survival_discount_hours = survival_discount_hours
        self._discount_rate = 1.0/survival_discount_hours if survival_discount_hours is not None else 0.0
        self.use_multi_rotation = use_multi_rotation
        self.survival_parameters = dict(SURVIVAL_PARAMETERS if survival_parameters is None else survival_parameters)
        for alpha, mu in self.survival_parameters.values():
            if not 0 <= alpha <= 1 or not mu >= 0 or not isfinite(mu):
                raise ValueError("Invalid survival parameter")
        self.mode = mode
        self.timing_key = timing_key
        self.epsilon = epsilon
        self.sigma_hours = sigma_hours
        self.lambda_empty = lambda_empty
        self.lambda_gap = lambda_gap
        self.use_cache = use_cache
        self.initial_epsilon = initial_epsilon
        self.expired_overdue_prob = expired_overdue_prob
        self.all_overdue_prob = all_overdue_prob
        self.intra_rotation_sigma_hours = intra_rotation_sigma_hours
        self.missed_load_offset_hours = missed_load_offset_hours
        self._advanced_risk = any(value is not None for value in (
            initial_epsilon, expired_overdue_prob, all_overdue_prob,
            intra_rotation_sigma_hours))
        self._watch_overdue = mode == "survival_risk" or expired_overdue_prob is not None or all_overdue_prob is not None
        self._timing = (("raw_strict" if timing_key == "raw" else timing_key)
                        if mode in ("expected_blocking", "survival_risk") else mode)
        self.schedule_path = schedule_path
        self.current_time = 0.0
        self._meta = {}
        self._rotations = {}
        self._ends = {}
        self._ports = {}
        self._global_ports = ()
        self._stacks = []
        self._discharge_windows = {}

    def initialize(self, yard_layout, initial_state):
        self.current_time = 0.0
        self._meta.clear()
        self._rotations.clear()
        self._ends.clear()
        self._ports.clear()
        self._global_ports = ()
        self._discharge_windows.clear()
        self._stacks = [
            (block, bay, row, info["tiers"])
            for block, info in sorted(yard_layout["blocks"].items())
            for bay in range(1, info["bays"] + 1)
            for row in range(1, info["rows"] + 1)
        ]
        if (self._timing.startswith("s") or self.expired_overdue_prob is not None
                or self.intra_rotation_sigma_hours is not None):
            self._load_schedule()
        self._reset_caches(yard_layout, initial_state)
        for entry in initial_state.get("containers", []):
            self._register(entry["container_id"], self._make_meta(
                entry.get("departure_time", ""), entry.get("vessel_id", ""),
                entry.get("port_of_discharge", ""), entry.get("weight_class", "MEDIUM")))

    def _load_schedule(self):
        candidates = [Path(self.schedule_path)] if self.schedule_path is not None else [
            Path(__file__).resolve().parents[1] / "data/vessel_schedule.json",
            Path.cwd() / "data/vessel_schedule.json",
        ]
        schedule = None
        for path in candidates:
            try:
                schedule = json.loads(path.read_text())
                break
            except FileNotFoundError:
                continue
        if schedule is None:
            return
        all_ports = set()
        for vessel in schedule.get("vessels", []):
            vid = vessel["vessel_id"]
            ports = tuple(sorted(set(vessel.get("ports", []))))
            self._ports[vid] = ports
            self._discharge_windows[vid] = tuple(
                (parse_time(r.get("discharge_start","")),parse_time(r.get("discharge_end","")))
                for r in vessel.get("rotations", []))
            all_ports.update(ports)
            rotations = tuple(sorted((Rotation(parse_time(r["etd"]),
                                               parse_time(r["load_start"]),
                                               parse_time(r["load_end"]))
                                      for r in vessel.get("rotations", [])), key=lambda r: r.end))
            self._rotations[vid] = rotations
            self._ends[vid] = tuple(r.end for r in rotations)
        self._global_ports = tuple(sorted(all_ports))

    def _make_meta(self, departure_time, vessel, port, weight):
        departure = parse_time(departure_time)
        weight_rank = WEIGHT_RANK.get(weight, 1)
        ports = self._ports.get(vessel, ())
        if port not in ports:
            ports = self._global_ports
        fraction = ((3 * ports.index(port) + weight_rank + 0.5) / (3 * len(ports))
                    if port in ports else 0.5)
        rotations = self._rotations.get(vessel, ())
        matched = min(rotations, key=lambda r: abs(r.etd - departure)) if rotations else None
        if matched is not None and abs(matched.etd - departure) > 24 * 3600:
            matched = None
        s1_time = (matched.start + fraction * (matched.end - matched.start)
                   if matched is not None else departure)
        return ContainerMeta(departure, vessel, port, weight_rank, fraction, matched, s1_time)

    def _reset_caches(self, layout, initial):
        self._initial_ids = {c["container_id"] for c in initial.get("containers", [])}
        self.seen_in_yard_or_arrived = set(self._initial_ids)
        self.observed_missed_loads = set()
        self._missed_dwell = {}
        self._overdue_heap = []
        self._origins = {cid:"initial" for cid in self._initial_ids}
        self._arrival_times = {}
        self._cid_index = {}
        self._eff = {}
        self._members = defaultdict(set)
        self._next_by_vessel = self._next_rotations()
        self._transitions = sorted((r.end, vessel) for vessel, rs in self._rotations.items() for r in rs)
        self._transition_index = 0
        self._generation = 0
        self._indices = {(b, bay, row): i for i, (b, bay, row, _) in enumerate(self._stacks)}
        self._block_indices = defaultdict(list)
        for i, (block, _, _, _) in enumerate(self._stacks):
            self._block_indices[block].append(i)
        self._capacity = {b: v["bays"] * v["rows"] * v["tiers"] for b, v in layout["blocks"].items()}
        self._max_tiers = {b: v["tiers"] for b, v in layout["blocks"].items()}
        self._block_counts = defaultdict(int)
        for c in initial.get("containers", []):
            self._block_counts[c["position"]["block"]] += 1
        self._contents = [[] for _ in self._stacks]
        self._key_lists = [() for _ in self._stacks]
        self._token_lists = [() for _ in self._stacks]
        self._survival_lists = [() for _ in self._stacks]
        self._floors = [INFINITY for _ in self._stacks]
        self._key_generation = [-1 for _ in self._stacks]
        self._versions = [0 for _ in self._stacks]
        self._keys_dirty = set()
        self._prepared_generation = -1
        self._cached_count = 0
        self._ready = False
        self._dirty = set()
        self._dirty_blocks = set()
        self._risk_tables = OrderedDict()
        self.cache_guard_fallbacks = 0
        self.conservative_global_invalidations = 0
        self.priority_transition_updates = 0

    def _register(self, cid, meta):
        previous = self._meta.get(cid)
        if previous is not None:
            self._members[previous.vessel].discard(cid)
        self._meta[cid] = meta
        self._members[meta.vessel].add(cid)
        self._eff[cid] = self._resident_priority(cid)
        if self._watch_overdue and meta.departure >= self.current_time:
            heapq.heappush(self._overdue_heap, (meta.departure, cid))

    def _resident_priority(self, cid):
        if cid in self._missed_dwell:
            return self._missed_dwell[cid]
        return self._priority(self._meta[cid], self._next_by_vessel)

    def _remember(self, container):
        meta = self._make_meta(container.departure_time, container.vessel_id,
                               container.port_of_discharge, container.weight_class)
        self._register(container.container_id, meta)
        return meta

    def _advance_time(self, timestamp):
        now = parse_time(timestamp)
        changed = set()
        backwards = now < self.current_time
        if now < self.current_time:
            # Useful for isolated what-if calls; the real event stream is ordered.
            self._transition_index = bisect_left([x[0] for x in self._transitions], now)
            changed.update(self._rotations)
        else:
            while (self._transition_index < len(self._transitions)
                   and self._transitions[self._transition_index][0] < now):
                changed.add(self._transitions[self._transition_index][1])
                self._transition_index += 1
        self.current_time = now
        if changed:
            self._next_by_vessel = self._next_rotations()
            if self._timing in ("s2_rollover", "s3_next_feasible"):
                for vessel in changed:
                    for cid in self._members[vessel]:
                        self._eff[cid] = self._resident_priority(cid)
                        self.priority_transition_updates += 1
            if self._timing.startswith("s") or self._advanced_risk:
                self._generation += 1
        if self._watch_overdue:
            if backwards:
                self._overdue_heap = [(m.departure,cid) for cid,m in self._meta.items()
                                      if m.departure >= now]
                heapq.heapify(self._overdue_heap)
                self._generation += 1
            else:
                while self._overdue_heap and self._overdue_heap[0][0] < now:
                    due,cid = heapq.heappop(self._overdue_heap)
                    meta = self._meta.get(cid)
                    if meta is not None and meta.departure == due:
                        index = self._cid_index.get(cid)
                        if index is not None:
                            self._keys_dirty.add(index)
                            self._versions[index] += 1

    def _observe_arrival(self, event):
        cid = event.container_id
        self._origins.setdefault(cid,event.type)
        self._arrival_times.setdefault(cid,self.current_time)
        if (self.missed_load_offset_hours is not None and cid in self.observed_missed_loads
                and cid not in self._missed_dwell):
            self._missed_dwell[cid] = self.current_time + self.missed_load_offset_hours * 3600
            if cid in self._meta:
                self._eff[cid] = self._missed_dwell[cid]
        self.seen_in_yard_or_arrived.add(cid)

    def on_event(self, event):
        self._advance_time(event.timestamp)
        if event.type == "LOAD" and event.container_id not in self.seen_in_yard_or_arrived:
            self.observed_missed_loads.add(event.container_id)
        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            self._observe_arrival(event)
            self._remember(event)

    def on_container_retrieved(self, container_id, position, reshuffles):
        meta = self._meta.pop(container_id, None)
        self._eff.pop(container_id, None)
        self._missed_dwell.pop(container_id, None)
        if meta is not None:
            self._members[meta.vessel].discard(container_id)
        block = position.block
        if reshuffles:
            # In the provided simulator, removing k blockers + target creates
            # k+1 local slots, so normal retrieval cannot overflow this block.
            # Still implement the requested conservative other-stack-free guard.
            pre_height = position.tier + reshuffles
            other_free = (self._capacity[block] - self._block_counts[block]
                          - (self._max_tiers[block] - pre_height))
            if other_free < reshuffles:
                self._dirty_blocks.update(self._block_indices)
                self.conservative_global_invalidations += 1
            else:
                self._dirty_blocks.add(block)
        else:
            self._dirty.add(self._indices[(block, position.bay, position.row)])
        self._block_counts[block] -= 1

    def _refresh_indices(self, yard, indices):
        touched = set()
        for i in indices:
            block, bay, row, _ = self._stacks[i]
            height = yard.get_stack_height(block, bay, row)
            cids = [yard.get_container_at(block, bay, row, tier) for tier in range(1, height + 1)]
            self._cached_count += len(cids) - len(self._contents[i])
            if cids != self._contents[i]:
                for cid in self._contents[i]:
                    if self._cid_index.get(cid) == i:
                        self._cid_index.pop(cid, None)
                self._contents[i] = cids
                self._key_generation[i] = -1
                self._keys_dirty.add(i)
                self._versions[i] += 1
            for cid in cids:
                self._cid_index[cid] = i
                if cid not in self._meta:
                    self._remember(yard.get_container_info(cid))
            touched.add(block)
        for block in touched:
            self._block_counts[block] = sum(len(self._contents[i]) for i in self._block_indices[block])

    def _refresh(self, yard):
        if not self.use_cache or not self._ready:
            self._refresh_indices(yard, range(len(self._stacks)))
            self._ready = True
        else:
            indices = set(self._dirty)
            for block in self._dirty_blocks:
                indices.update(self._block_indices[block])
            self._refresh_indices(yard, sorted(indices))
        self._dirty.clear()
        self._dirty_blocks.clear()
        if self._cached_count != yard.total_containers():
            self.cache_guard_fallbacks += 1
            self._refresh_indices(yard, range(len(self._stacks)))
            if self._cached_count != yard.total_containers():
                raise RuntimeError("Live yard count disagrees with public stack contents")
        if not self.use_cache:
            # Independent control: bypass transition caching as well as dirty
            # stack caching, recomputing priorities from the physical clock.
            self._next_by_vessel = self._next_rotations()
            for cids in self._contents:
                for cid in cids:
                    self._eff[cid] = self._resident_priority(cid)
            self._generation += 1
        indices = (range(len(self._stacks)) if self._prepared_generation != self._generation
                   else self._keys_dirty)
        for i in indices:
            keys = tuple(self._eff[cid] for cid in self._contents[i])
            self._key_lists[i] = keys
            self._floors[i] = min(keys, default=INFINITY)
            self._key_generation[i] = self._generation
            if self._advanced_risk:
                self._token_lists[i] = tuple(self._risk_token(cid) for cid in self._contents[i])
            if self.mode == "survival_risk":
                self._survival_lists[i] = tuple(self._survival_key(cid) for cid in self._contents[i])
        self._keys_dirty.clear()
        self._prepared_generation = self._generation

    def _stack_keys(self, index):
        if self._key_generation[index] != self._generation:
            keys = tuple(self._eff[cid] for cid in self._contents[index])
            self._key_lists[index] = keys
            self._floors[index] = min(keys, default=INFINITY)
            self._key_generation[index] = self._generation
        return self._key_lists[index]

    def _next_rotations(self):
        next_rotations = {}
        for vessel, ends in self._ends.items():
            index = bisect_left(ends, self.current_time)
            if index < len(ends):
                next_rotations[vessel] = self._rotations[vessel][index]
        return next_rotations

    def _priority(self, meta, next_rotations, resident=True):
        if self._timing.startswith("raw"):
            return meta.departure
        if self._timing == "s1_stored_order":
            return meta.s1_time
        if self._timing == "s2_rollover":
            # S2 rolls over containers already resident in the yard. An incoming
            # container retains its S1 estimate for this placement decision.
            if not resident:
                return meta.s1_time
            rotation = meta.matched_rotation
            if rotation is None or self.current_time <= rotation.end:
                return meta.s1_time
            rotation = next_rotations.get(meta.vessel)
            if rotation is None:
                return meta.s1_time
        else:
            rotation = next_rotations.get(meta.vessel)
            if rotation is None:
                return meta.departure
        return rotation.start + meta.fraction * (rotation.end - rotation.start)

    def _incoming_priority(self, event, meta):
        if event.container_id in self._missed_dwell:
            return self._missed_dwell[event.container_id]
        return self._priority(meta, self._next_by_vessel, resident=False)

    def pair_probability(self, resident_key, incoming_key):
        if resident_key == incoming_key:
            return 0.5  # includes two +infinity long-dwell priorities
        z = (resident_key - incoming_key) / (3600.0 * self.sigma_hours)
        z = max(-30.0, min(30.0, z))
        return self.epsilon + (1.0 - 2.0 * self.epsilon) / (1.0 + exp(z))

    def _assigned_rotation(self, cid, resident=True):
        if cid in self._missed_dwell:
            return None
        meta = self._meta[cid]
        if self._timing == "s3_next_feasible":
            rotation = self._next_by_vessel.get(meta.vessel)
        elif self._timing in ("s1_stored_order", "s2_rollover"):
            rotation = meta.matched_rotation
            if (self._timing == "s2_rollover" and resident and rotation is not None
                    and self.current_time > rotation.end):
                rotation = self._next_by_vessel.get(meta.vessel, rotation)
        else:
            rotation = None
        return (meta.vessel,rotation) if rotation is not None else None

    def _risk_token(self, cid):
        meta = self._meta[cid]
        floor = (self.initial_epsilon if self.initial_epsilon is not None
                 and cid in self._initial_ids else self.epsilon)
        return (self._eff[cid], floor, meta.departure < self.current_time,
                meta.vessel not in self._next_by_vessel,
                self._assigned_rotation(cid) if self.intra_rotation_sigma_hours is not None else None)

    def _token_probability(self, token, incoming_context):
        key,floor,overdue,no_future,rotation = token
        incoming_key,incoming_rotation,incoming_overdue = incoming_context
        if self.expired_overdue_prob is not None and overdue and no_future:
            return self.expired_overdue_prob
        if self.all_overdue_prob is not None and overdue and not incoming_overdue:
            return self.all_overdue_prob
        if key == incoming_key:
            return 0.5
        scale = (self.intra_rotation_sigma_hours if rotation is not None
                 and rotation == incoming_rotation else self.sigma_hours)
        z = max(-30.0,min(30.0,(key-incoming_key)/(3600.0*scale)))
        return floor + (1.0-2.0*floor)/(1.0+exp(z))

    def _survival_key(self, cid):
        meta = self._meta[cid]
        rotation = self._next_by_vessel.get(meta.vessel)
        overdue = meta.departure < self.current_time
        if cid in self.observed_missed_loads:
            return ("missed_load",() if self.use_multi_rotation else INFINITY)
        if rotation is None:
            return ("expired_no_rot" if overdue else "unscheduled_active",() if self.use_multi_rotation else INFINITY)
        if overdue:
            group = "sched_overdue"
        elif cid in self._initial_ids:
            group = "sched_init_active"
        elif self._origins.get(cid) == "TRUCK_RECV":
            group = "sched_truck_active"
        else:
            group = "sched_disch_active"
        if self.use_expanded_survival and group in ("sched_disch_active","sched_truck_active"):
            if EXPANSION_FEATURE == "lead_bucket":
                lead = (meta.departure-self._arrival_times[cid])/3600
                level = "<24h" if lead < 24 else "24-72h" if lead <= 72 else ">72h"
            elif EXPANSION_FEATURE == "etd_vs_s3":
                diff = (meta.departure-rotation.etd)/3600
                level = "match" if abs(diff)<=24 else "later" if diff>24 else "earlier_or_other"
            else:
                level = str(any(a<=self._arrival_times[cid]<=b for a,b in self._discharge_windows.get(meta.vessel,())))
            if group+"|"+level in EXPANDED_PARAMETERS:
                group += "|"+level
        if self.use_multi_rotation:
            index=bisect_left(self._ends[meta.vessel],self.current_time)
            times=tuple(r.start+meta.fraction*(r.end-r.start) for r in self._rotations[meta.vessel][index:])
            return group,times
        return group,self._eff[cid]

    def _survival_values(self, descriptor):
        group,when = descriptor
        if "|" in group:
            alpha,mu = EXPANDED_PARAMETERS[group]
        else:
            alpha,mu = self.survival_parameters[group]
        if self.use_multi_rotation:
            overdue_alpha=self.survival_parameters["sched_overdue"][0]
            jumps=tuple((max(0.0,(t-self.current_time)/3600),alpha if i==0 else overdue_alpha)
                        for i,t in enumerate(when))
            return alpha,mu,jumps
        return alpha,mu,max(0.0,(when-self.current_time)/3600)

    def _place_survival(self, event):
        ac,mc,tc = self._survival_values(self._survival_key(event.container_id))
        probabilities = {}  # Per call only: absolute time matters to this model.
        best = None
        best_index = None
        for i,(_,_,_,maximum) in enumerate(self._stacks):
            height = len(self._contents[i])
            if height >= maximum: continue
            score = 0.0
            for descriptor in self._survival_lists[i]:
                probability = probabilities.get(descriptor)
                if probability is None:
                    ax,mx,tx = self._survival_values(descriptor)
                    if self.use_multi_rotation:
                        probability=multi_survival_block_probability(mx,tx,mc,tc,
                            self.survival_horizon_hours,self._discount_rate)
                    else:
                        probability = survival_block_probability(ax,mx,tx,ac,mc,tc,
                            self.survival_horizon_hours,self.survival_rotation_sigma_hours,self._discount_rate)
                    probabilities[descriptor] = probability
                score += probability
            rank = score,height
            if best is None or rank < best:
                best,best_index = rank,i
        if best_index is None: raise RuntimeError("No non-full stack is available")
        return self._finish_placement(best_index)

    def _finish_placement(self, index):
        block,bay,row,_ = self._stacks[index]
        position = Position(block,bay,row,len(self._contents[index])+1)
        self._dirty.add(index)
        self._block_counts[block] += 1
        return position

    def place_container(self, yard_state, event):
        self._advance_time(event.timestamp)
        self._observe_arrival(event)
        incoming = self._meta.get(event.container_id)
        if incoming is None:
            incoming = self._remember(event)
        self._refresh(yard_state)
        if self.mode == "survival_risk":
            return self._place_survival(event)
        incoming_key = self._incoming_priority(event, incoming)
        expected = self.mode == "expected_blocking"
        top_only = self.mode == "raw_top_only"
        nonstrict = self.mode == "raw_nonstrict"
        risks = scores = None
        incoming_context = None
        table_key = incoming_key
        if self._advanced_risk:
            incoming_context = (incoming_key,
                                self._assigned_rotation(event.container_id,resident=False)
                                if self.intra_rotation_sigma_hours is not None else None,
                                incoming.departure < self.current_time)
            table_key = incoming_context
        if expected:
            table = self._risk_tables.get(table_key)
            if table is None:
                risks = {}
                scores = [None] * len(self._stacks)
                self._risk_tables[table_key] = (risks, scores)
                if len(self._risk_tables) > 128:
                    self._risk_tables.popitem(last=False)
            else:
                risks, scores = table
                self._risk_tables.move_to_end(table_key)
        key_lists = self._key_lists
        floors = self._floors
        versions = self._versions
        generation = self._generation
        best_rank = None
        best_index = None
        for i, (block, bay, row, max_tiers) in enumerate(self._stacks):
            height = len(self._contents[i])
            if height >= max_tiers:
                continue
            if expected:
                if height == 0:
                    score = self.lambda_empty
                elif self.epsilon == 0.5 and not self._advanced_risk:
                    score = 0.5 * height
                else:
                    previous = scores[i] if self.use_cache else None
                    if previous is not None and previous[0] == versions[i] and previous[1] == generation:
                        score = previous[2]
                    else:
                        score = 0.0
                        for key in (self._token_lists[i] if self._advanced_risk else key_lists[i]):
                            probability = risks.get(key)
                            if probability is None:
                                probability = (self._token_probability(key,incoming_context)
                                               if self._advanced_risk else self.pair_probability(key, incoming_key))
                                risks[key] = probability
                            score += probability
                        scores[i] = (versions[i], generation, score)
                if self.lambda_gap and height:
                    gap = floors[i] - incoming_key
                    if gap > 0:
                        score += self.lambda_gap * min(gap / (240.0 * 3600), 1.0)
                rank = (score, height)
            elif height == 0:
                rank = (1,)
            else:
                keys = key_lists[i]
                if top_only:
                    keys = keys[-1:]
                floor = keys[-1] if top_only else floors[i]
                inversions = sum(key < incoming_key for key in keys)
                ties = sum(key == incoming_key for key in keys)
                if incoming_key < floor or (nonstrict and incoming_key == floor):
                    rank = (0, floor - incoming_key, height)
                elif incoming_key == floor:
                    rank = (2, ties, height)
                else:
                    rank = (3, inversions, ties, -floor, height)
            # Sorted coordinates supply deterministic final tie-breaking.
            if best_rank is None or rank < best_rank:
                best_rank, best_index = rank, i
        if best_index is None:
            raise RuntimeError("No non-full stack is available")
        return self._finish_placement(best_index)
