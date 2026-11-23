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
    assert not strategy._advanced_risk and strategy.missed_load_offset_hours is None


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


def test_initial_floor_distinguishes_identical_priorities():
    strategy,yard=setup([[9],[]],"expected_blocking",initial_epsilon=0.25)
    yard.place_container(Container("new",departure_time=when(9),vessel_id="V1",
                                   port_of_discharge="PORT_02",weight_class="LIGHT"),Position("A",2,1,1))
    assert strategy.place_container(yard,incoming()).bay == 2
    assert strategy._risk_token("c1_1")[1] == 0.25
    assert strategy._risk_token("new")[1] == 0.05


@pytest.mark.parametrize("flag", ["expired_overdue_prob","all_overdue_prob"])
def test_overdue_change_invalidates_scores_without_schedule_boundary(flag):
    strategy,yard=setup([[3],[8]*4],"expected_blocking",timing_key="raw",**{flag:0.10})
    assert strategy.place_container(yard,incoming(now=when(2))).bay == 2
    assert strategy.place_container(yard,incoming(now=when(4))).bay == 1


def test_overdue_rules_have_distinct_scope():
    expired=MyStrategy(expired_overdue_prob=0.2)
    overdue=MyStrategy(all_overdue_prob=0.3)
    a,b=parse_time(when(3)),parse_time(when(5))
    scheduled=(a,0.05,True,False,None)
    no_future=(a,0.05,True,True,None)
    incoming_context=(b,None,False)
    assert expired._token_probability(no_future,incoming_context) == 0.2
    assert expired._token_probability(scheduled,incoming_context) > 0.9
    assert overdue._token_probability(scheduled,incoming_context) == 0.3
    assert overdue._token_probability(scheduled,(b,None,True)) > 0.9


def test_intra_rotation_sharpening_excludes_other_vessels_and_fallbacks(schedule_path):
    strategy,_=setup([[]],"expected_blocking",schedule_path,intra_rotation_sigma_hours=0.5)
    meta=strategy._remember(incoming(3,cid="resident",port="PORT_02",weight="LIGHT"))
    strategy.current_time=parse_time(when(1))
    rotation=strategy._assigned_rotation("resident")
    assert rotation is not None
    token=(parse_time(when(2,8)),0.05,False,False,rotation)
    key=parse_time(when(2,6))
    sharp=strategy._token_probability(token,(key,rotation,False))
    cross=strategy._token_probability(token,(key,("OTHER",rotation[1]),False))
    fallback=strategy._token_probability(token,(key,None,False))
    assert sharp < 0.1 and cross > 0.4
    assert fallback == cross


@pytest.mark.parametrize("offset", [336,float("inf")])
def test_missed_load_uses_only_observed_history_and_fixed_priority(offset,schedule_path):
    strategy,yard=setup([[],[]],"expected_blocking",schedule_path,missed_load_offset_hours=offset)
    strategy.on_event(Event(0,when(1),"LOAD","late"))
    e=incoming(3,cid="late",now=when(1,1))
    strategy.on_event(e)
    pos=strategy.place_container(yard,e)
    assert yard.place_container(e.to_container(),pos)
    expected=parse_time(e.timestamp)+offset*3600
    assert strategy._eff["late"] == expected
    strategy.on_event(Event(2,when(4),"LOAD","other_missing"))
    assert strategy._eff["late"] == expected
    assert strategy.pair_probability(float("inf"),float("inf")) == 0.5
    strategy.initialize({"blocks":{"A":{"bays":2,"rows":1,"tiers":5}}},{"containers":[]})
    assert not strategy.observed_missed_loads and not strategy._missed_dwell


@pytest.mark.parametrize("flag", [
    {"initial_epsilon":0.20},
    {"expired_overdue_prob":0.20},
    {"all_overdue_prob":0.20},
    {"intra_rotation_sigma_hours":0.5,"sigma_hours":48},
    {"missed_load_offset_hours":336},
    {"missed_load_offset_hours":float("inf")},
])
def test_isolated_flags_cached_vs_uncached_across_history_and_time(flag,schedule_path):
    traces=[]
    states=[]
    histories=[]
    for cached in (True,False):
        strategy,yard=setup([[20,2,10,12],[]],"expected_blocking",schedule_path,
                            use_cache=cached,**flag)
        trace=[]
        events=[Event(0,when(1),"LOAD","late"),incoming(3,cid="late",now=when(1,1)),
                Event(2,when(2),"LOAD","c1_2"),incoming(3,cid="x",now=when(2,12)),
                Event(4,when(2,13),"LOAD","x"),incoming(11,cid="y",now=when(4)),
                Event(6,when(10,13),"LOAD","c1_3"),incoming(12,cid="z",now=when(12))]
        for e in events:
            if e.type in ("DISCHARGE","TRUCK_RECV"):
                strategy.on_event(e)
                position=strategy.place_container(yard,e)
                assert yard.is_position_valid(position)
                trace.append(position)
                assert yard.place_container(e.to_container(),position)
            else:
                Simulator(yard,strategy).run([e])
        traces.append(trace)
        states.append(yard.snapshot())
        histories.append(strategy.observed_missed_loads)
    assert traces[0] == traces[1]
    assert states[0] == states[1]
    assert histories[0] == histories[1]


def test_previously_retrieved_container_is_not_a_missed_load():
    strategy,yard=setup([[3]],"expected_blocking",missed_load_offset_hours=336)
    Simulator(yard,strategy).run([Event(1,when(2),"LOAD","c1_1"),Event(2,when(2,1),"LOAD","c1_1")])
    assert "c1_1" not in strategy.observed_missed_loads


@pytest.mark.parametrize("tx,tc,expected", [(1,2,1),(2,1,0),(1,1,0.5)])
def test_survival_deterministic_jump_limit(tx,tc,expected):
    from solution.my_strategy import survival_block_probability as probability
    assert probability(1,0,tx,1,0,tc) == expected


def test_survival_greedy_and_infinite_survivor_limits():
    from solution.my_strategy import survival_block_probability as probability
    assert probability(0,.001,float("inf"),0,.001,float("inf")) == 0.5
    assert probability(0,0,float("inf"),0,0,float("inf")) == 0
    assert probability(1,0,5,0,0,float("inf"),4) == 0
    assert probability(1,0,5,0,0,float("inf"),5) == 1
    params={g:(0,.001) for g in MyStrategy().survival_parameters}
    strategy,yard=setup([[9,9],[2],[]],"survival_risk",survival_horizon_hours=float("inf"),survival_parameters=params)
    from src.baseline_greedy import GreedyStrategy
    assert strategy.place_container(yard,incoming()) == GreedyStrategy().place_container(yard,incoming())


@pytest.mark.parametrize("tx,tc,sigma",[(7.3,12.7,0),(12.7,7.3,0),(9,9,0),(12.7,7.3,2)])
def test_survival_closed_form_matches_numerical_integral(tx,tc,sigma):
    from math import exp
    from solution.my_strategy import survival_block_probability as probability
    ax,ac,mx,mc,horizon=.6,.4,.03,.02,20
    n=20000
    step=horizon/n
    numerical=0
    for i in range(n):
        t=(i+.5)*step
        sx=exp(-mx*t)*(1-ax if t>=tx else 1)
        sc=exp(-mc*t)*(1-ac if t>=tc else 1)
        numerical+=mx*sx*sc*step
    phi=(1/(1+exp(-(tx-tc)/sigma)) if sigma else 1 if tx>tc else .5 if tx==tc else 0)
    numerical+=ax*exp(-(mx+mc)*tx)*(1-ac*phi)
    assert probability(ax,mx,tx,ac,mc,tc,horizon,sigma) == pytest.approx(numerical,abs=1e-7)


def test_exact_survival_pair_mass_and_tie_symmetry():
    from math import exp
    from solution.my_strategy import survival_block_probability as p
    ax,ac,mx,mc,tx,tc,h=.6,.4,.01,.02,5,5,20
    assert p(ax,mx,tx,ac,mc,tc,h)+p(ac,mc,tc,ax,mx,tx,h) == pytest.approx(1-exp(-(mx+mc)*h)*(1-ax)*(1-ac))
    assert p(.6,0,5,.6,0,5) == pytest.approx(.42)


def test_survival_missed_incoming_chooses_expired_base(schedule_path):
    layout={"blocks":{"A":{"bays":2,"rows":1,"tiers":5}}}
    yard=YardState(layout)
    yard.place_container(Container("active",vessel_id="V1",departure_time=when(11),port_of_discharge="PORT_01"),Position("A",1,1,1))
    yard.place_container(Container("expired",vessel_id="VSL019",departure_time=when(2)),Position("A",2,1,1))
    params={g:(.9,.001) for g in MyStrategy().survival_parameters}
    params["missed_load"]=(0,0)
    params["expired_no_rot"]=(0,.0001)
    strategy=MyStrategy("survival_risk",schedule_path,survival_parameters=params)
    strategy.initialize(layout,yard.export_state())
    strategy.on_event(Event(0,when(3),"LOAD","late"))
    event=incoming(1,cid="late",now=when(4))
    strategy.on_event(event)
    before=yard.snapshot()
    assert strategy.place_container(yard,event).bay == 2
    assert yard.snapshot() == before


@pytest.mark.parametrize("expanded",[False,True])
def test_survival_cached_live_history_and_cohort_transitions(expanded,schedule_path):
    traces=[]
    states=[]
    for cached in (True,False):
        strategy,yard=setup([[20,2,10,12],[]],"survival_risk",schedule_path,
                            use_cache=cached,use_expanded_survival=expanded)
        trace=[]
        for e in [Event(0,when(1),"LOAD","late"),incoming(3,cid="late",now=when(1,1)),
                  Event(2,when(2),"LOAD","c1_2"),incoming(3,cid="x",now=when(3)),
                  incoming(11,cid="y",now=when(3,1)),Event(5,when(4),"LOAD","c1_3"),
                  incoming(12,cid="z",now=when(12))]:
            if e.type in ("DISCHARGE","TRUCK_RECV"):
                strategy.on_event(e)
                pos=strategy.place_container(yard,e)
                trace.append(pos)
                assert yard.place_container(e.to_container(),pos)
            else: Simulator(yard,strategy).run([e])
        traces.append(trace)
        states.append(yard.snapshot())
    assert traces[0] == traces[1]
    assert states[0] == states[1]
