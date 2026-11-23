"""Online departure-aware placement with explicit Phase 2 ablation modes.

Inspired by Borgman, van Asperen & Dekker (2010), Online rules for container
stacking, https://doi.org/10.1007/s00291-010-0205-4. Full-stack compatibility,
strict tie handling and schedule variants are adaptations for this simulator.
Only the supplied schedule, current event, initial state and live yard are read.
"""
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

from src.models import Position
from src.placement_interface import PlacementStrategy


MODES = ("raw_strict", "raw_nonstrict", "raw_top_only", "s1_stored_order",
         "s2_rollover", "s3_next_feasible")
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
    """Four-tier LDT policy. Temporary default is the Phase 2 overall leader S3.

    A placement scans live non-full stacks; only immutable metadata is cached.
    For S stacks, maximum height H, and R rotations per vessel, decisions take
    O(S*H + V*log R) time and O(V) temporary space after metadata warm-up.
    Persistent metadata uses O(N + S + schedule size) space for N containers.
    """
    def __init__(self, mode="s3_next_feasible", schedule_path=None):
        if mode not in MODES:
            raise ValueError(f"Unknown strategy mode: {mode}")
        self.mode = mode
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
        if self.mode.startswith("s"):
            self._load_schedule()
        for entry in initial_state.get("containers", []):
            self._meta[entry["container_id"]] = self._make_meta(
                entry.get("departure_time", ""), entry.get("vessel_id", ""),
                entry.get("port_of_discharge", ""), entry.get("weight_class", "MEDIUM"))

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

    def _remember(self, container):
        meta = self._make_meta(container.departure_time, container.vessel_id,
                               container.port_of_discharge, container.weight_class)
        self._meta[container.container_id] = meta
        return meta

    def on_event(self, event):
        self.current_time = parse_time(event.timestamp)
        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            self._remember(event)

    def on_container_retrieved(self, container_id, position, reshuffles):
        self._meta.pop(container_id, None)

    def _next_rotations(self):
        next_rotations = {}
        for vessel, ends in self._ends.items():
            index = bisect_left(ends, self.current_time)
            if index < len(ends):
                next_rotations[vessel] = self._rotations[vessel][index]
        return next_rotations

    def _priority(self, meta, next_rotations, resident=True):
        if self.mode.startswith("raw"):
            return meta.departure
        if self.mode == "s1_stored_order":
            return meta.s1_time
        if self.mode == "s2_rollover":
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

    def place_container(self, yard_state, event):
        self.current_time = parse_time(event.timestamp)
        incoming = self._meta.get(event.container_id)
        if incoming is None:
            incoming = self._remember(event)
        dynamic = self.mode in ("s2_rollover", "s3_next_feasible")
        next_rotations = self._next_rotations() if dynamic else {}
        incoming_key = self._priority(incoming, next_rotations, resident=False)
        top_only = self.mode == "raw_top_only"
        nonstrict = self.mode == "raw_nonstrict"
        get_height = yard_state.get_stack_height
        get_at = yard_state.get_container_at
        get_info = yard_state.get_container_info
        metadata = self._meta
        priority = self._priority
        best_rank = None
        best_position = None
        for block, bay, row, max_tiers in self._stacks:
            height = get_height(block, bay, row)
            if height >= max_tiers:
                continue
            if height == 0:
                rank = (1,)
            else:
                floor = INFINITY
                inversions = ties = 0
                for tier in range(height if top_only else 1, height + 1):
                    cid = get_at(block, bay, row, tier)
                    meta = metadata.get(cid)
                    if meta is None:
                        meta = self._remember(get_info(cid))
                    key = priority(meta, next_rotations)
                    if key < floor:
                        floor = key
                    inversions += key < incoming_key
                    ties += key == incoming_key
                if incoming_key < floor or (nonstrict and incoming_key == floor):
                    rank = (0, floor - incoming_key, height)
                elif incoming_key == floor:
                    rank = (2, ties, height)
                else:
                    rank = (3, inversions, ties, -floor, height)
            # Stacks are sorted lexically; retaining the first equal rank supplies
            # the specified coordinate tie-break without allocating extra tuples.
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_position = (block, bay, row, height + 1)
        if best_position is None:
            raise RuntimeError("No non-full stack is available")
        return Position(*best_position)
