"""Frozen discounted competing-risks placement; all observations are online.

The only file read by this module is the supplied public vessel schedule.
See docs/design.md for calibration, research, and complexity.
"""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import heapq
import json
from math import exp, isfinite
from pathlib import Path

from src.models import Position
from src.placement_interface import PlacementStrategy
from .survival_params import SURVIVAL_PARAMETERS

WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
UNKNOWN_TIME = 253402300799.0
INFINITY = float("inf")
DISCOUNT_RATE = 1.0 / 240.0


def survival_block_probability(alpha_x, mu_x, tau_x, alpha_c, mu_c, tau_c,
                               discount_rate=DISCOUNT_RATE):
    """Discounted P(resident exits before incoming), with hours as time units.

    Each exit time combines an exponential background and one Bernoulli jump.
    Equal jump times use a fair ordering coin. The frozen rate is 1/240 per hour;
    the optional rate argument supports mathematical limit tests only.
    """
    rate = mu_x + mu_c + discount_rate
    jump_x = alpha_x > 0 and isfinite(tau_x)
    jump_c = alpha_c > 0 and isfinite(tau_c)
    ex = exp(-rate * tau_x) if jump_x and rate else 1.0
    ec = exp(-rate * tau_c) if jump_c and rate else 1.0
    background = 0.0
    if mu_x > 0:
        integral = 1.0
        if jump_x:
            integral -= alpha_x * ex
        if jump_c:
            integral -= alpha_c * ec
        if jump_x and jump_c:
            integral += alpha_x * alpha_c * min(ex, ec)
        background = mu_x / rate * integral
    atom = 0.0
    if jump_x:
        precedes = 0.0
        if jump_c:
            delta = tau_x - tau_c
            precedes = 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
        atom = alpha_x * ex * (1.0 - alpha_c * precedes)
    return min(1.0, max(0.0, background + atom))


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
    start: float
    end: float


@dataclass(frozen=True)
class ContainerMeta:
    departure: float
    vessel: str
    fraction: float


class MyStrategy(PlacementStrategy):
    """Sum discounted pairwise blocking risks over each non-full live stack.

    Only stack contents/descriptors are cached; probabilities are memoized anew
    per placement. use_cache=False independently rereads every public stack.
    """
    def __init__(self, schedule_path=None, use_cache=True):
        self.schedule_path = schedule_path
        self.use_cache = use_cache
        self.current_time = 0.0
        self._meta = {}
        self._rotations = {}
        self._ends = {}
        self._ports = {}
        self._global_ports = ()
        self._stacks = []

    def initialize(self, yard_layout, initial_state):
        self.current_time = 0.0
        self._meta.clear()
        self._rotations.clear()
        self._ends.clear()
        self._ports.clear()
        self._global_ports = ()
        self._stacks = [
            (block, bay, row, info["tiers"])
            for block, info in sorted(yard_layout["blocks"].items())
            for bay in range(1, info["bays"] + 1)
            for row in range(1, info["rows"] + 1)
        ]
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
            all_ports.update(ports)
            rotations = tuple(sorted((Rotation(parse_time(r["load_start"]),
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
        return ContainerMeta(departure, vessel, fraction)

    def _reset_caches(self, layout, initial):
        self._initial_ids = {c["container_id"] for c in initial.get("containers", [])}
        self.seen_in_yard_or_arrived = set(self._initial_ids)
        self.observed_missed_loads = set()
        self._overdue_heap = []
        self._origins = {cid:"initial" for cid in self._initial_ids}
        self._arrival_times = {}
        self._cid_index = {}
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
        self._block_slots = defaultdict(int)
        initial_heights = defaultdict(int)
        for c in initial.get("containers", []):
            p = c["position"]
            key = p["block"], p["bay"], p["row"]
            initial_heights[key] = max(initial_heights[key], p["tier"])
        for (block, _, _), height in initial_heights.items():
            self._block_slots[block] += height
        self._heights = [0 for _ in self._stacks]
        self._contents = [[] for _ in self._stacks]
        self._survival_lists = [() for _ in self._stacks]
        self._keys_dirty = set()
        self._prepared_generation = -1
        self._cached_count = 0
        self._ready = False
        self._dirty = set()
        self._dirty_blocks = set()
        self.cache_guard_fallbacks = 0
        self.conservative_global_invalidations = 0

    def _register(self, cid, meta):
        self._meta[cid] = meta
        if meta.departure >= self.current_time:
            heapq.heappush(self._overdue_heap, (meta.departure, cid))

    def _remember(self, container):
        meta = self._make_meta(container.departure_time, container.vessel_id,
                               container.port_of_discharge, container.weight_class)
        self._register(container.container_id, meta)
        return meta

    def _advance_time(self, timestamp):
        now = parse_time(timestamp)
        backwards = now < self.current_time
        changed = backwards
        if backwards:
            self._transition_index = bisect_left([x[0] for x in self._transitions], now)
        else:
            while (self._transition_index < len(self._transitions)
                   and self._transitions[self._transition_index][0] < now):
                changed = True
                self._transition_index += 1
        self.current_time = now
        if changed:
            self._next_by_vessel = self._next_rotations()
            self._generation += 1
        if backwards:
            self._overdue_heap = [(m.departure, cid) for cid, m in self._meta.items()
                                  if m.departure >= now]
            heapq.heapify(self._overdue_heap)
        else:
            while self._overdue_heap and self._overdue_heap[0][0] < now:
                due, cid = heapq.heappop(self._overdue_heap)
                meta = self._meta.get(cid)
                if meta is not None and meta.departure == due:
                    index = self._cid_index.get(cid)
                    if index is not None:
                        self._keys_dirty.add(index)

    def _observe_arrival(self, event):
        cid = event.container_id
        self._origins.setdefault(cid, event.type)
        self._arrival_times.setdefault(cid, self.current_time)
        self.seen_in_yard_or_arrived.add(cid)

    def on_event(self, event):
        self._advance_time(event.timestamp)
        if event.type == "LOAD" and event.container_id not in self.seen_in_yard_or_arrived:
            self.observed_missed_loads.add(event.container_id)
        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            self._observe_arrival(event)
            self._remember(event)

    def on_container_retrieved(self, container_id, position, reshuffles):
        # Arrival identity/history remains retained, even after successful exit.
        self._meta.pop(container_id, None)
        block = position.block
        if reshuffles:
            # Physical slots include padded gaps. A gap can prevent removal of
            # the target, so retain conservative slot counts until live refresh.
            pre_height = position.tier + reshuffles
            other_free = (self._capacity[block] - self._block_slots[block]
                          - (self._max_tiers[block] - pre_height))
            if other_free < reshuffles:
                self._dirty_blocks.update(self._block_indices)
                self.conservative_global_invalidations += 1
            else:
                self._dirty_blocks.add(block)
        else:
            self._dirty.add(self._indices[(block, position.bay, position.row)])
            self._block_slots[block] = max(0, self._block_slots[block] - 1)

    def _refresh_indices(self, yard, indices):
        touched = set()
        for i in indices:
            block, bay, row, _ = self._stacks[i]
            height = yard.get_stack_height(block, bay, row)
            self._heights[i] = height
            cids = [cid for tier in range(1, height + 1)
                    if (cid := yard.get_container_at(block, bay, row, tier))]
            self._cached_count += len(cids) - len(self._contents[i])
            if cids != self._contents[i]:
                for cid in self._contents[i]:
                    if self._cid_index.get(cid) == i:
                        self._cid_index.pop(cid, None)
                self._contents[i] = cids
                self._keys_dirty.add(i)
            for cid in cids:
                self._cid_index[cid] = i
                if cid not in self._meta:
                    info = yard.get_container_info(cid)
                    if info is not None:
                        self._remember(info)
                        self._keys_dirty.add(i)
            touched.add(block)
        for block in touched:
            self._block_slots[block] = sum(self._heights[i] for i in self._block_indices[block])

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
            self._next_by_vessel = self._next_rotations()
            self._generation += 1
        indices = (range(len(self._stacks)) if self._prepared_generation != self._generation
                   else self._keys_dirty)
        for i in indices:
            self._survival_lists[i] = tuple(self._survival_key(cid) for cid in self._contents[i])
        self._keys_dirty.clear()
        self._prepared_generation = self._generation

    def _next_rotations(self):
        next_rotations = {}
        for vessel, ends in self._ends.items():
            index = bisect_left(ends, self.current_time)
            if index < len(ends):
                next_rotations[vessel] = self._rotations[vessel][index]
        return next_rotations

    def _survival_key(self, cid):
        meta = self._meta[cid]
        rotation = self._next_by_vessel.get(meta.vessel)
        overdue = meta.departure < self.current_time
        if cid in self.observed_missed_loads:
            return ("missed_load",INFINITY)
        if rotation is None:
            return ("expired_no_rot" if overdue else "unscheduled_active",INFINITY)
        if overdue:
            group = "sched_overdue"
        elif cid in self._initial_ids:
            group = "sched_init_active"
        elif self._origins.get(cid) == "TRUCK_RECV":
            group = "sched_truck_active"
        else:
            group = "sched_disch_active"
        if group in ("sched_disch_active", "sched_truck_active"):
            lead = (meta.departure - self._arrival_times.get(cid, self.current_time)) / 3600
            level = "<24h" if lead < 24 else "24-72h" if lead <= 72 else ">72h"
            group += "|" + level
        when = rotation.start + meta.fraction * (rotation.end - rotation.start)
        return group, when

    def _survival_values(self, descriptor):
        group, when = descriptor
        alpha, mu = SURVIVAL_PARAMETERS[group]
        return alpha, mu, max(0.0, (when - self.current_time) / 3600)

    def _place_survival(self, event):
        ac,mc,tc = self._survival_values(self._survival_key(event.container_id))
        probabilities = {}  # Per call only: absolute time matters to this model.
        best = None
        best_index = None
        for i,(_,_,_,maximum) in enumerate(self._stacks):
            height = self._heights[i]
            if height >= maximum: continue
            score = 0.0
            for descriptor in self._survival_lists[i]:
                probability = probabilities.get(descriptor)
                if probability is None:
                    ax,mx,tx = self._survival_values(descriptor)
                    probability = survival_block_probability(ax, mx, tx, ac, mc, tc)
                    probabilities[descriptor] = probability
                score += probability
            rank = score,height
            if best is None or rank < best:
                best,best_index = rank,i
        if best_index is None: raise RuntimeError("No non-full stack is available")
        return self._finish_placement(best_index)

    def _finish_placement(self, index):
        block,bay,row,_ = self._stacks[index]
        position = Position(block,bay,row,self._heights[index]+1)
        self._dirty.add(index)
        self._block_slots[block] += 1
        return position

    def place_container(self, yard_state, event):
        self._advance_time(event.timestamp)
        self._observe_arrival(event)
        if event.container_id not in self._meta:
            self._remember(event)
        self._refresh(yard_state)
        return self._place_survival(event)
