"""Behavioral tests for online placement and schedule ablations."""
import json
import random

import pytest

from solution.my_strategy import MODES, MyStrategy, parse_time
from src.models import Container, Event, Position
from src.simulator import Simulator
from src.yard_state import YardState


def when(day, hour=0):
    return f"2025-01-{day:02d}T{hour:02d}:00:00"


def incoming(day=5, cid="incoming", vessel="V1", port="PORT_01", weight="MEDIUM", now=None):
    return Event(1, now or when(1), "DISCHARGE", cid,
                 vessel_id=vessel, port_of_discharge=port,
                 weight_class=weight, departure_time=when(day))


def setup(stacks, mode="raw_strict", schedule_path=None, **kwargs):
    layout = {"blocks": {"A": {"bays": len(stacks), "rows": 1, "tiers": 5}}}
    yard = YardState(layout)
    for bay, days in enumerate(stacks, 1):
        for tier, day in enumerate(days, 1):
            yard.place_container(Container(f"c{bay}_{tier}", departure_time=when(day),
                                           vessel_id="V1", port_of_discharge="PORT_02",
                                           weight_class="LIGHT"), Position("A",bay,1,tier))
    strategy = MyStrategy(mode, schedule_path, **kwargs)
    strategy.initialize(layout,yard.export_state())
    return strategy,yard


@pytest.fixture
def schedule_path(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"vessels": [
        {"vessel_id":"V1", "ports":["PORT_02","PORT_01"], "rotations":[
            {"etd":when(3), "load_start":when(2), "load_end":when(2,12)},
            {"etd":when(11), "load_start":when(10), "load_end":when(10,12)}]},
        {"vessel_id":"V2", "ports":["PORT_03"], "rotations":[]},
        {"vessel_id":"VSL019", "ports":["PORT_01"], "rotations":[]},
    ]}))
    return path


def priority(strategy, event):
    strategy.on_event(event)
    return strategy._priority(strategy._meta[event.container_id],strategy._next_rotations())


def test_zero_arg_default():
    strategy=MyStrategy()
    assert strategy.mode == "expected_blocking"
    assert (strategy.timing_key,strategy.epsilon,strategy.sigma_hours)==("s3_next_feasible",0.05,12)
    assert strategy.lambda_empty == strategy.lambda_gap == 0


def test_invalid_mode():
    with pytest.raises(ValueError):
        MyStrategy("typo")


@pytest.mark.parametrize("mode", MODES)
def test_legal_deterministic_and_no_yard_mutation(mode, tmp_path):
    strategy,yard=setup([[8],[2,10],[],[9]*5],mode,tmp_path/"missing.json")
    before=yard.snapshot()
    a=strategy.place_container(yard,incoming())
    b=strategy.place_container(yard,incoming())
    assert a == b
    assert yard.is_position_valid(a)
    assert before == yard.snapshot()
    assert a.bay != 4


def test_nondiscriminating_original_fixture():
    for mode in ("raw_strict","raw_top_only"):
        strategy,yard=setup([[2,10],[8]],mode)
        assert strategy.place_container(yard,incoming()).bay == 2


def test_full_stack_avoids_hidden_early_container():
    strict,yard=setup([[2,10],[12]])
    top,_=setup([[2,10],[12]],"raw_top_only")
    assert strict.place_container(yard,incoming()).bay == 2
    assert top.place_container(yard,incoming()).bay == 1


def test_strict_and_nonstrict_equal_priority_difference():
    strict,yard=setup([[5],[8]])
    nonstrict,_=setup([[5],[8]],"raw_nonstrict")
    assert strict.place_container(yard,incoming()).bay == 2
    assert nonstrict.place_container(yard,incoming()).bay == 1


def test_fewer_tied_containers_precedes_height():
    strategy,yard=setup([[5,5],[5,8,9]])
    assert strategy.place_container(yard,incoming()).bay == 2


@pytest.mark.parametrize("stacks,expected", [
    ([[8],[]],1),                # strict compatibility before empty
    ([[5],[]],2),                # empty before ties
    ([[2],[5]],2),               # ties before forced inversions
    ([[8,10],[8]],2),            # equal safe gap: shorter stack
    ([[1,2],[1,8]],2),           # fewer forced inversions
    ([[1,5],[2,9]],2),           # fewer ties among equal inversion counts
    ([[2],[4]],2),               # higher floor among equal inversion/tie counts
    ([[2,9],[2]],2),             # height is last numerical tie-break
    ([[8],[8]],1),               # coordinate tie-break
])
def test_tier_order_and_tie_breaks(stacks,expected):
    strategy,yard=setup(stacks)
    assert strategy.place_container(yard,incoming()).bay == expected


def test_no_available_stack_raises():
    strategy,yard=setup([[8]*5])
    with pytest.raises(RuntimeError,match="No non-full"):
        strategy.place_container(yard,incoming())


def test_reads_actual_simulator_relocations():
    strategy,yard=setup([[20,2,10,12],[]])
    Simulator(yard,strategy).run([Event(1,when(2),"LOAD","c1_2")])
    assert yard.get_container_position("c1_3") == Position("A",2,1,1)
    assert strategy.place_container(yard,incoming(now=when(2))) == Position("A",2,1,2)


def test_intra_window_port_and_weight_order(schedule_path):
    strategy,_=setup([[]],"s1_stored_order",schedule_path)
    events=[incoming(3,cid=str(i),port=p,weight=w) for i,(p,w) in enumerate([
        ("PORT_01","HEAVY"),("PORT_01","LIGHT"),("PORT_02","HEAVY")])]
    keys=[priority(strategy,e) for e in events]
    assert parse_time(when(2)) < keys[0] < keys[1] < keys[2] < parse_time(when(2,12))


def test_schedule_earlier_category_chooses_safe_stack(schedule_path):
    strategy,yard=setup([[3]],"s1_stored_order",schedule_path)
    # Existing PORT_02 LIGHT is later than incoming PORT_01 HEAVY.
    assert strategy.place_container(yard,incoming(3,weight="HEAVY")) == Position("A",1,1,2)


def test_s2_rollover_and_inclusive_end_boundary(schedule_path):
    strategy,_=setup([[]],"s2_rollover",schedule_path)
    event=incoming(3,now=when(2,12))
    first=priority(strategy,event)
    assert first < parse_time(when(3))
    strategy.on_event(Event(2,when(2,13),"LOAD","other"))
    rolled=strategy._priority(strategy._meta[event.container_id],strategy._next_rotations())
    assert parse_time(when(10)) < rolled < parse_time(when(10,12))
    assert strategy._priority(strategy._meta[event.container_id],strategy._next_rotations(),resident=False) == first
    strategy.on_event(Event(3,when(12),"LOAD","other"))
    assert strategy._priority(strategy._meta[event.container_id],strategy._next_rotations()) == first


def test_s3_ignores_stored_rotation(schedule_path):
    strategy,_=setup([[]],"s3_next_feasible",schedule_path)
    assert priority(strategy,incoming(11,now=when(1))) < parse_time(when(3))


@pytest.mark.parametrize("mode", ["s1_stored_order","s2_rollover","s3_next_feasible"])
def test_unscheduled_falls_back_to_raw(mode,schedule_path):
    strategy,_=setup([[]],mode,schedule_path)
    assert priority(strategy,incoming(3,vessel="VSL019")) == parse_time(when(3))


@pytest.mark.parametrize("mode", ["s1_stored_order","s2_rollover","s3_next_feasible"])
def test_missing_schedule_falls_back_to_raw(mode,tmp_path):
    strategy,_=setup([[]],mode,tmp_path/"missing.json")
    assert priority(strategy,incoming(3)) == parse_time(when(3))


def test_s1_match_tolerance_and_unknown_port(schedule_path):
    strategy,_=setup([[]],"s1_stored_order",schedule_path)
    assert priority(strategy,incoming(4)) < parse_time(when(3))  # exactly 24h is accepted
    assert priority(strategy,incoming(5)) == parse_time(when(5))
    meta=strategy._remember(incoming(3,port="PORT_03"))
    assert 0 < meta.fraction < 1
    assert strategy._remember(incoming(3,port="UNKNOWN")).fraction == 0.5


def test_s1_priority_is_static_after_window(schedule_path):
    strategy,_=setup([[]],"s1_stored_order",schedule_path)
    assert priority(strategy,incoming(3,now=when(1))) == priority(strategy,incoming(3,now=when(12)))


def test_package_schedule_path_independent_of_cwd(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    strategy=MyStrategy("s1_stored_order")
    strategy.initialize({"blocks":{}},{"containers":[]})
    assert strategy._rotations["VSL001"]


def test_empty_metadata_has_finite_deterministic_fallback():
    strategy,yard=setup([[]])
    event=incoming()
    event.departure_time=""
    assert yard.is_position_valid(strategy.place_container(yard,event))
    assert parse_time("") == parse_time("invalid")


def test_arbitrary_small_live_yards(tmp_path):
    rng=random.Random(7)
    for _ in range(30):
        stacks=[[rng.randint(2,20) for _ in range(rng.randrange(6))] for _ in range(4)]
        stacks[0]=stacks[0][:4]
        strategy,yard=setup(stacks,schedule_path=tmp_path/"missing")
        before=yard.snapshot()
        assert yard.is_position_valid(strategy.place_container(yard,incoming(rng.randint(2,20))))
        assert yard.snapshot() == before


def test_probability_symmetry_and_greedy_endpoint():
    strategy=MyStrategy("expected_blocking",epsilon=0.15,sigma_hours=12)
    a,b=parse_time(when(3)),parse_time(when(8))
    assert strategy.pair_probability(a,a) == 0.5
    assert strategy.pair_probability(a,b)+strategy.pair_probability(b,a) == pytest.approx(1)
    strategy,yard=setup([[8,8],[2],[]],"expected_blocking",epsilon=0.5)
    from src.baseline_greedy import GreedyStrategy
    assert strategy.place_container(yard,incoming()) == GreedyStrategy().place_container(yard,incoming())


@pytest.mark.parametrize("stacks,epsilon,expected", [
    ([[9]*4,[9]],0.15,2),
    ([[2],[8,8]],0.15,2),
    ([[2],[8]*4],0.30,1),
])
def test_continuous_risk_height_tradeoff(stacks,epsilon,expected):
    strategy,yard=setup(stacks,"expected_blocking",epsilon=epsilon,sigma_hours=12)
    assert strategy.place_container(yard,incoming()).bay == expected


def test_empty_stack_opportunity_cost():
    ordinary,yard=setup([[9],[]],"expected_blocking",epsilon=0.15,sigma_hours=12)
    preserve,_=setup([[9],[]],"expected_blocking",epsilon=0.15,sigma_hours=12,lambda_empty=0.20)
    assert ordinary.place_container(yard,incoming()).bay == 2
    assert preserve.place_container(yard,incoming()).bay == 1


def test_gap_waste_preserves_far_future_base():
    ordinary,yard=setup([[20],[8]],"expected_blocking",epsilon=0.15,sigma_hours=12)
    waste,_=setup([[20],[8]],"expected_blocking",epsilon=0.15,sigma_hours=12,lambda_gap=0.15)
    assert ordinary.place_container(yard,incoming(2)).bay == 1
    assert waste.place_container(yard,incoming(2)).bay == 2


@pytest.mark.parametrize("timing", ["raw","s2_rollover","s3_next_feasible"])
def test_cached_vs_live_multistep(timing,schedule_path):
    from dataclasses import asdict
    trace=[]
    states=[]
    for use_cache in (True,False):
        strategy,yard=setup([[20,2,10,12],[]],"expected_blocking",schedule_path,
                            timing_key=timing,use_cache=use_cache)
        positions=[]
        for event in [Event(1,when(2),"LOAD","c1_2"),incoming(3,cid="x",now=when(2,12)),
                      Event(3,when(2,13),"LOAD","c1_3"),incoming(11,cid="y",now=when(3)),
                      Event(5,when(10,13),"LOAD","x"),incoming(12,cid="z",now=when(12))]:
            if event.type in ("DISCHARGE","TRUCK_RECV"):
                strategy.on_event(event)
                pos=strategy.place_container(yard,event)
                positions.append(asdict(pos))
                assert yard.place_container(event.to_container(),pos)
            else:
                Simulator(yard,strategy).run([event])
        trace.append(positions)
        states.append(yard.snapshot())
    assert trace[0] == trace[1]
    assert states[0] == states[1]


def test_full_block_retrieval_does_not_overflow():
    # Removing blockers and target always creates enough room in this block.
    layout={"blocks":{"A":{"bays":1,"rows":1,"tiers":5},"B":{"bays":1,"rows":1,"tiers":5}}}
    traces=[]
    for cached in (True,False):
        yard=YardState(layout)
        for tier in range(1,6):
            yard.place_container(Container(f"a{tier}",departure_time=when(tier+1)),Position("A",1,1,tier))
        strategy=MyStrategy("expected_blocking",use_cache=cached)
        strategy.initialize(layout,yard.export_state())
        # Populate cache without applying this proposed placement.
        strategy.place_container(yard,incoming(cid="unused"))
        Simulator(yard,strategy).run([Event(1,when(2),"LOAD","a1")])
        assert all(yard.get_container_position(f"a{i}").block=="A" for i in range(2,6))
        assert strategy.conservative_global_invalidations == 1
        traces.append(strategy.place_container(yard,incoming(cid="next")))
    assert traces[0] == traces[1]


def test_count_guard_recovers_unreported_cross_block_move():
    # Contrived external move exercises the global guard; actual valid simulator
    # retrievals cannot overflow. The low-level fallback helper is tested below.
    strategy,yard=setup([[8],[]],"expected_blocking")
    strategy.place_container(yard,incoming())
    c=yard.get_container_info("c1_1")
    yard.remove_container("c1_1")
    yard.place_container(c,Position("A",2,1,1))
    strategy._dirty.clear()
    # Same-block count-preserving arbitrary external edits cannot be detected
    # by a count guard; flag its source, as a retrieval callback would.
    strategy._dirty.add(strategy._indices[("A",1,1)])
    strategy.place_container(yard,incoming())
    assert strategy.cache_guard_fallbacks == 1
    assert strategy._contents[strategy._indices[("A",2,1)]] == ["c1_1"]


def test_low_level_full_block_fallback():
    layout={"blocks":{"A":{"bays":1,"rows":1,"tiers":1},"B":{"bays":1,"rows":1,"tiers":1}}}
    yard=YardState(layout)
    c=Container("a")
    yard.place_container(c,Position("A",1,1,1))
    sim=Simulator(yard,MyStrategy())
    assert sim._find_reshuffle_position("A",c) == Position("B",1,1,1)


def test_contrived_cross_block_relocation_equivalence():
    layout={"blocks":{"A":{"bays":2,"rows":1,"tiers":3},"B":{"bays":1,"rows":1,"tiers":3}}}
    traces=[]
    for cached in (True,False):
        yard=YardState(layout)
        for cid,day,tier in [("target",2,1),("blocker",8,2)]:
            yard.place_container(Container(cid,departure_time=when(day)),Position("A",1,1,tier))
        strategy=MyStrategy("expected_blocking",use_cache=cached)
        strategy.initialize(layout,yard.export_state())
        strategy.place_container(yard,incoming(cid="unused"))
        blocker=yard.get_container_info("blocker")
        yard.remove_container("blocker")
        yard.remove_container("target")
        yard.place_container(blocker,Position("B",1,1,1))
        strategy.on_container_retrieved("target",Position("A",1,1,1),1)
        traces.append(strategy.place_container(yard,incoming(cid="next")))
        if cached:
            assert strategy.cache_guard_fallbacks == 1
    assert traces[0] == traces[1]
