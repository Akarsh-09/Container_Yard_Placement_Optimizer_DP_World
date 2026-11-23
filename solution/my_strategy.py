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
import json
from math import exp
from pathlib import Path
from typing import Optional

from src.models import Position
from src.placement_interface import PlacementStrategy


MODES = ("raw_strict", "raw_nonstrict", "raw_top_only", "s1_stored_order",
         "s2_rollover", "s3_next_feasible", "expected_blocking")
WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
UNKNOWN_TIME = 253402300799.0
INFINITY = float("inf")


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
                 lambda_empty=0.0, lambda_gap=0.0, use_cache=True):
        if mode not in MODES:
            raise ValueError(f"Unknown strategy mode: {mode}")
        if timing_key not in ("raw", "s2_rollover", "s3_next_feasible"):
            raise ValueError("Unknown timing key")
        if not 0 <= epsilon <= 0.5 or not sigma_hours > 0:
            raise ValueError("Require 0 <= epsilon <= 0.5 and sigma_hours > 0")
        if lambda_empty < 0 or lambda_gap < 0:
            raise ValueError("Opportunity costs must be nonnegative")
        self.mode = mode
        self.timing_key = timing_key
        self.epsilon = epsilon
        self.sigma_hours = sigma_hours
        self.lambda_empty = lambda_empty
        self.lambda_gap = lambda_gap
        self.use_cache = use_cache
        self._timing = (("raw_strict" if timing_key == "raw" else timing_key)
                        if mode == "expected_blocking" else mode)
        self.schedule_path = schedule_path
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
        if self._timing.startswith("s"):
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

    def _resident_priority(self, cid):
        return self._priority(self._meta[cid], self._next_by_vessel)

    def _remember(self, container):
        meta = self._make_meta(container.departure_time, container.vessel_id,
                               container.port_of_discharge, container.weight_class)
        self._register(container.container_id, meta)
        return meta

    def _advance_time(self, timestamp):
        now = parse_time(timestamp)
        changed = set()
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
        if changed and self._timing in ("s2_rollover", "s3_next_feasible"):
            self._next_by_vessel = self._next_rotations()
            for vessel in changed:
                for cid in self._members[vessel]:
                    self._eff[cid] = self._resident_priority(cid)
                    self.priority_transition_updates += 1
            self._generation += 1

    def on_event(self, event):
        self._advance_time(event.timestamp)
        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            self._remember(event)

    def on_container_retrieved(self, container_id, position, reshuffles):
        meta = self._meta.pop(container_id, None)
        self._eff.pop(container_id, None)
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
                self._contents[i] = cids
                self._key_generation[i] = -1
                self._keys_dirty.add(i)
                self._versions[i] += 1
            for cid in cids:
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
        return self._priority(meta, self._next_by_vessel, resident=False)

    def pair_probability(self, resident_key, incoming_key):
        z = (resident_key - incoming_key) / (3600.0 * self.sigma_hours)
        z = max(-30.0, min(30.0, z))
        return self.epsilon + (1.0 - 2.0 * self.epsilon) / (1.0 + exp(z))

    def place_container(self, yard_state, event):
        self._advance_time(event.timestamp)
        incoming = self._meta.get(event.container_id)
        if incoming is None:
            incoming = self._remember(event)
        self._refresh(yard_state)
        incoming_key = self._incoming_priority(event, incoming)
        expected = self.mode == "expected_blocking"
        top_only = self.mode == "raw_top_only"
        nonstrict = self.mode == "raw_nonstrict"
        risks = scores = None
        if expected:
            table = self._risk_tables.get(incoming_key)
            if table is None:
                risks = {}
                scores = [None] * len(self._stacks)
                self._risk_tables[incoming_key] = (risks, scores)
                if len(self._risk_tables) > 128:
                    self._risk_tables.popitem(last=False)
            else:
                risks, scores = table
                self._risk_tables.move_to_end(incoming_key)
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
                elif self.epsilon == 0.5:
                    score = 0.5 * height
                else:
                    previous = scores[i] if self.use_cache else None
                    if previous is not None and previous[0] == versions[i] and previous[1] == generation:
                        score = previous[2]
                    else:
                        score = 0.0
                        for key in key_lists[i]:
                            probability = risks.get(key)
                            if probability is None:
                                probability = self.pair_probability(key, incoming_key)
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
        block, bay, row, _ = self._stacks[best_index]
        position = Position(block, bay, row, len(self._contents[best_index]) + 1)
        self._dirty.add(best_index)
        self._block_counts[block] += 1  # anticipated successful placement
        return position
