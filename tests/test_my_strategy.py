"""Production policy math, online information, public-state and cache checks."""
import json
from math import exp, inf
import random

import pytest

from solution.my_strategy import (DISCOUNT_RATE, MyStrategy, parse_time,
                                   survival_block_probability as probability)
from src.baseline_greedy import GreedyStrategy
from src.models import Container, Event, Position
from src.simulator import Simulator
from src.yard_state import YardState


def when(day, hour=0):
    return f"2025-01-{day:02d}T{hour:02d}:00:00"


def incoming(day=5, cid="incoming", vessel="V1", port="PORT_01", weight="MEDIUM", now=None, kind="DISCHARGE"):
    return Event(1, now or when(1), kind, cid, vessel_id=vessel,
                 port_of_discharge=port, weight_class=weight, departure_time=when(day))


def setup(stacks, schedule_path=None, use_cache=True):
    layout = {"blocks": {"A": {"bays": len(stacks), "rows": 1, "tiers": 5}}}
    yard = YardState(layout)
    for bay, days in enumerate(stacks, 1):
        for tier, day in enumerate(days, 1):
            yard.place_container(Container(f"c{bay}_{tier}", departure_time=when(day),
                                           vessel_id="V1", port_of_discharge="PORT_02",
                                           weight_class="LIGHT"), Position("A",bay,1,tier))
    strategy = MyStrategy(schedule_path=schedule_path, use_cache=use_cache)
    strategy.initialize(layout,yard.export_state())
    return strategy,yard


@pytest.fixture
def schedule_path(tmp_path):
    path=tmp_path/'schedule.json'
    path.write_text(json.dumps({'vessels':[
        {'vessel_id':'V1','ports':['PORT_02','PORT_01'],'rotations':[
            {'load_start':when(2),'load_end':when(2,12)},
            {'load_start':when(10),'load_end':when(10,12)}]},
        {'vessel_id':'V2','ports':['PORT_03'],'rotations':[]},
        {'vessel_id':'VSL019','ports':['PORT_01'],'rotations':[]},
        {'vessel_id':'VSL020','ports':[],'rotations':[]}]}))
    return path


def descriptor(strategy,event):
    strategy.on_event(event)
    return strategy._survival_key(event.container_id)


def test_zero_arg_and_only_frozen_constructor_options():
    import inspect
    assert list(inspect.signature(MyStrategy).parameters)==['schedule_path','use_cache']
    assert MyStrategy().use_cache and DISCOUNT_RATE==1/240
    from solution.survival_params import SURVIVAL_PARAMETERS
    assert len(SURVIVAL_PARAMETERS)==11


@pytest.mark.parametrize('cached',[True,False])
def test_legal_full_avoidance_determinism_and_no_mutation(cached,tmp_path):
    strategy,yard=setup([[8],[2,10],[],[9]*5],tmp_path/'missing',cached)
    before=yard.snapshot()
    a=strategy.place_container(yard,incoming())
    b=strategy.place_container(yard,incoming())
    assert a==b and yard.is_position_valid(a) and a.bay!=4
    assert before==yard.snapshot()


def test_no_available_stack_raises():
    strategy,yard=setup([[8]*5])
    with pytest.raises(RuntimeError,match='No non-full'):strategy.place_container(yard,incoming())


def test_package_schedule_independent_of_cwd(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    strategy=MyStrategy()
    strategy.initialize({'blocks':{}},{'containers':[]})
    assert strategy._rotations['VSL001']


def test_cwd_schedule_fallback(tmp_path,monkeypatch,schedule_path):
    # Simulate an installed module whose package-relative data file is absent.
    import solution.my_strategy as module
    monkeypatch.setattr(module,'__file__',str(tmp_path/'installed'/'solution'/'my_strategy.py'))
    (tmp_path/'data').mkdir()
    (tmp_path/'data'/'vessel_schedule.json').write_text(schedule_path.read_text())
    monkeypatch.chdir(tmp_path)
    strategy=MyStrategy();strategy.initialize({'blocks':{}},{'containers':[]})
    assert strategy._rotations['V1']


def test_missing_schedule_graceful_cohorts(tmp_path):
    strategy,yard=setup([[]],tmp_path/'missing')
    assert descriptor(strategy,incoming(3))[0]=='unscheduled_active'
    assert descriptor(strategy,incoming(2,cid='old',now=when(4)))[0]=='expired_no_rot'
    assert yard.is_position_valid(strategy.place_container(yard,incoming(8,now=when(4))))


@pytest.mark.parametrize('vessel',['VSL019','VSL020','UNKNOWN'])
def test_unscheduled_vessels_have_no_jump(vessel,schedule_path):
    strategy,_=setup([[]],schedule_path)
    assert descriptor(strategy,incoming(3,vessel=vessel))==('unscheduled_active',inf)
    assert descriptor(strategy,incoming(2,cid='old',vessel=vessel,now=when(4)))==('expired_no_rot',inf)


def test_port_weight_order_and_unknown_port_fallback(schedule_path):
    strategy,_=setup([[]],schedule_path)
    keys=[descriptor(strategy,incoming(3,cid=str(i),port=p,weight=w))[1]
          for i,(p,w) in enumerate([('PORT_01','HEAVY'),('PORT_01','LIGHT'),('PORT_02','HEAVY')])]
    assert parse_time(when(2))<keys[0]<keys[1]<keys[2]<parse_time(when(2,12))
    assert strategy._remember(incoming(port='PORT_03')).fraction==pytest.approx(7.5/9.0)
    assert strategy._remember(incoming(port='UNKNOWN')).fraction==.5


def test_rollover_inclusive_end_and_stored_etd_ignored(schedule_path):
    strategy,_=setup([[]],schedule_path)
    first=descriptor(strategy,incoming(11,now=when(2,12)))[1]
    assert first<parse_time(when(3))
    strategy.on_event(Event(2,when(2,13),'LOAD','missing'))
    rolled=strategy._survival_key('incoming')[1]
    assert parse_time(when(10))<rolled<parse_time(when(10,12))
    strategy.on_event(Event(3,when(12),'LOAD','missing'))
    assert strategy._survival_key('incoming')==('expired_no_rot',inf)


@pytest.mark.parametrize('kind,prefix',[('DISCHARGE','sched_disch_active'),('TRUCK_RECV','sched_truck_active')])
@pytest.mark.parametrize('day,bucket',[(1,'<24h'),(2,'24-72h'),(4,'24-72h'),(5,'>72h')])
def test_expanded_lead_time_buckets(kind,prefix,day,bucket,schedule_path):
    strategy,_=setup([[]],schedule_path)
    assert descriptor(strategy,incoming(day,kind=kind))[0]==prefix+'|'+bucket


def test_initial_and_raw_etd_overdue_transition_invalidates_cache(schedule_path):
    strategy,yard=setup([[3],[]],schedule_path)
    strategy.place_container(yard,incoming(11,now=when(3)))
    assert strategy._survival_lists[0][0][0]=='sched_init_active'  # equality is active
    strategy.place_container(yard,incoming(11,now=when(3,1)))
    assert strategy._survival_lists[0][0][0]=='sched_overdue'


def test_missed_load_only_uses_past_observed_absence_and_resets(schedule_path):
    strategy,yard=setup([[3]],schedule_path)
    assert not strategy.observed_missed_loads
    strategy.on_event(Event(0,when(1),'LOAD','late'))
    assert descriptor(strategy,incoming(3,cid='late',now=when(1,1)))==('missed_load',inf)
    Simulator(yard,strategy).run([Event(1,when(2),'LOAD','c1_1'),Event(2,when(2,1),'LOAD','c1_1')])
    assert 'c1_1' not in strategy._meta
    assert 'c1_1' in strategy.seen_in_yard_or_arrived
    assert 'c1_1' not in strategy.observed_missed_loads
    strategy.initialize({'blocks':{}},{'containers':[]})
    assert not strategy.observed_missed_loads and not strategy.seen_in_yard_or_arrived


@pytest.mark.parametrize('empty',[False,True])
def test_missed_incoming_routes_away_from_active_resident(empty,schedule_path):
    strategy,yard=setup([[11],[]],schedule_path)
    if not empty:
        yard.place_container(Container('expired',vessel_id='VSL019',departure_time=when(1)),Position('A',2,1,1))
    strategy.on_event(Event(0,when(1),'LOAD','late'))
    e=incoming(3,cid='late',now=when(1,1))
    strategy.on_event(e)
    before=yard.snapshot()
    assert strategy.place_container(yard,e).bay==2
    assert yard.snapshot()==before


def numerical(ax,mx,tx,ac,mc,tc,gamma):
    # Independent midpoint integration, split exactly at each jump. The 1000h
    # tail is negligible for the positive rates used below; atoms are explicit.
    horizon=1000
    boundaries=sorted({0,horizon}|{t for t in (tx,tc) if 0<t<horizon})
    total=0
    for a,b in zip(boundaries,boundaries[1:]):
        n=max(1000,int((b-a)*100));step=(b-a)/n
        for i in range(n):
            t=a+(i+.5)*step
            sx=exp(-mx*t)*(1-ax if t>tx else 1)
            sc=exp(-mc*t)*(1-ac if t>tc else 1)
            total+=exp(-gamma*t)*mx*sx*sc*step
    if ax and tx<inf:
        phi=1 if tx>tc else .5 if tx==tc else 0
        total+=ax*exp(-(mx+mc+gamma)*tx)*(1-ac*phi)
    return total


@pytest.mark.parametrize('tx,tc',[(7,13),(13,7),(9,9),(inf,7),(7,inf)])
@pytest.mark.parametrize('gamma',[0,DISCOUNT_RATE,.1])
def test_discounted_math_matches_numerical_integration(tx,tc,gamma):
    ax,ac=(.6 if tx<inf else 0),(.4 if tc<inf else 0)
    assert probability(ax,.03,tx,ac,.02,tc,gamma)==pytest.approx(numerical(ax,.03,tx,ac,.02,tc,gamma),abs=1e-7)


@pytest.mark.parametrize('tx,tc,expected',[(1,2,1),(2,1,0),(1,1,.5)])
def test_deterministic_limit_and_ties(tx,tc,expected):
    assert probability(1,0,tx,1,0,tc,0)==expected
    assert probability(1,0,tx,1,0,tc)==pytest.approx(expected*exp(-DISCOUNT_RATE*tx))


def test_zero_jump_equal_hazard_greedy_limit(monkeypatch):
    import solution.my_strategy as module
    monkeypatch.setattr(module,'SURVIVAL_PARAMETERS',{g:(0,.001) for g in module.SURVIVAL_PARAMETERS})
    assert probability(0,.001,inf,0,.001,inf)==pytest.approx(.001/(.002+DISCOUNT_RATE))
    strategy,yard=setup([[9,9],[2],[]])
    assert strategy.place_container(yard,incoming())==GreedyStrategy().place_container(yard,incoming())
    assert probability(0,0,inf,0,0,inf)==0
    assert probability(.6,.03,7,.4,.02,13,1e-12)==pytest.approx(probability(.6,.03,7,.4,.02,13,0),abs=1e-10)


def replay(cached,schedule_path):
    strategy,yard=setup([[20,2,10,12],[]],schedule_path,cached)
    trace=[]
    for e in [Event(0,when(1),'LOAD','late'),incoming(3,cid='late',now=when(1,1)),
              Event(2,when(2),'LOAD','c1_2'),incoming(11,cid='x',now=when(2,12)),
              Event(4,when(2,13),'LOAD','c1_3'),incoming(11,cid='y',now=when(4)),
              incoming(12,cid='z',now=when(12))]:
        if e.type=='DISCHARGE':
            strategy.on_event(e);before=yard.snapshot();p=strategy.place_container(yard,e)
            assert before==yard.snapshot() and yard.is_position_valid(p)
            trace.append(p);assert yard.place_container(e.to_container(),p)
        else:Simulator(yard,strategy).run([e])
    return trace,yard.snapshot(),strategy.observed_missed_loads


def test_cached_live_exact_multistep_and_repeated_run_determinism(schedule_path):
    assert replay(True,schedule_path)==replay(False,schedule_path)==replay(True,schedule_path)


@pytest.mark.parametrize('target,reshuffles',[('c1_4',0),('c1_2',2)])
def test_actual_retrieval_dirty_stack_and_block(target,reshuffles,schedule_path):
    strategy,yard=setup([[20,2,10,12],[]],schedule_path)
    strategy.place_container(yard,incoming(cid='proposal'))
    stats=Simulator(yard,strategy).run([Event(1,when(2),'LOAD',target)])
    assert stats.total_reshuffles==reshuffles
    assert bool(strategy._dirty_blocks)==bool(reshuffles)
    strategy.place_container(yard,incoming(cid='next',now=when(3)))
    for i,(b,bay,row,_) in enumerate(strategy._stacks):
        assert strategy._contents[i]==[yard.get_container_at(b,bay,row,t) for t in range(1,yard.get_stack_height(b,bay,row)+1)]


def test_full_block_retrieval_and_conservative_guard():
    layout={'blocks':{'A':{'bays':1,'rows':1,'tiers':5},'B':{'bays':1,'rows':1,'tiers':5}}}
    traces=[]
    for cached in (True,False):
        yard=YardState(layout)
        for tier in range(1,6):yard.place_container(Container(f'a{tier}',departure_time=when(tier+1)),Position('A',1,1,tier))
        strategy=MyStrategy(use_cache=cached);strategy.initialize(layout,yard.export_state())
        strategy.place_container(yard,incoming(cid='unused'))  # unapplied proposal deliberately makes counts conservative
        Simulator(yard,strategy).run([Event(1,when(2),'LOAD','a1')])
        assert all(yard.get_container_position(f'a{i}').block=='A' for i in range(2,6))
        assert strategy.conservative_global_invalidations==1
        traces.append(strategy.place_container(yard,incoming(cid='next')))
    assert traces[0]==traces[1]


def test_count_guard_recovers_partially_reported_move():
    strategy,yard=setup([[8],[]])
    strategy.place_container(yard,incoming())
    c=yard.get_container_info('c1_1');yard.remove_container('c1_1');yard.place_container(c,Position('A',2,1,1))
    strategy._dirty.clear();strategy._dirty.add(strategy._indices[('A',1,1)])
    strategy.place_container(yard,incoming())
    assert strategy.cache_guard_fallbacks==1
    assert strategy._contents[1]==['c1_1']


def test_low_level_cross_block_fallback():
    layout={'blocks':{'A':{'bays':1,'rows':1,'tiers':1},'B':{'bays':1,'rows':1,'tiers':1}}}
    yard=YardState(layout);c=Container('a');yard.place_container(c,Position('A',1,1,1))
    assert Simulator(yard,MyStrategy())._find_reshuffle_position('A',c)==Position('B',1,1,1)


def test_contrived_cross_block_relocation_recovery():
    layout={'blocks':{'A':{'bays':2,'rows':1,'tiers':3},'B':{'bays':1,'rows':1,'tiers':3}}}
    traces=[]
    for cached in (True,False):
        yard=YardState(layout)
        for cid,day,tier in [('target',2,1),('blocker',8,2)]:yard.place_container(Container(cid,departure_time=when(day)),Position('A',1,1,tier))
        strategy=MyStrategy(use_cache=cached);strategy.initialize(layout,yard.export_state())
        strategy.place_container(yard,incoming(cid='unused'))
        blocker=yard.get_container_info('blocker');yard.remove_container('blocker');yard.remove_container('target')
        yard.place_container(blocker,Position('B',1,1,1))
        strategy.on_container_retrieved('target',Position('A',1,1,1),1)
        traces.append(strategy.place_container(yard,incoming(cid='next')))
        if cached:assert strategy.cache_guard_fallbacks==1
    assert traces[0]==traces[1]


def test_arbitrary_live_yards_and_invalid_timestamps(tmp_path):
    rng=random.Random(7)
    for _ in range(30):
        stacks=[[rng.randint(2,20) for _ in range(rng.randrange(6))] for _ in range(4)]
        stacks[0]=stacks[0][:4]
        strategy,yard=setup(stacks,tmp_path/'missing')
        before=yard.snapshot()
        assert yard.is_position_valid(strategy.place_container(yard,incoming(rng.randint(2,20))))
        assert yard.snapshot()==before
    assert parse_time('')==parse_time('invalid')
