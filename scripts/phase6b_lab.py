"""Isolated train-only Phase 6B laboratory; no submitted files are edited.

The engine snapshot is a byte-identical copy loaded with package='solution' so
its unchanged relative parameter import resolves normally. Only the diagnostic
oracle sees full-train future labels. Every deployable constant is fit before
Jan13. No test file is read, replayed, or used for selecting configurations.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from bisect import bisect_left
import hashlib
import importlib.util
import json
from math import exp, expm1, inf, isfinite
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.event_reader import read_events
from src.simulator import Simulator
from src.yard_state import YardState
from src.models import Event, Container, Position
from src.placement_interface import PlacementStrategy

OUT=ROOT/'results/phase6b'
START=1735689600.0
CUT=1736726400.0
END=1737417600.0
ARRIVALS={'DISCHARGE','TRUCK_RECV'}


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def load_engine():
    source=ROOT/'solution/my_strategy.py';snapshot=ROOT/'scripts/phase6b_engine_snapshot.py'
    snapshot.write_bytes(source.read_bytes())
    assert snapshot.read_bytes()==source.read_bytes()
    name='solution._phase6b_engine_snapshot'
    spec=importlib.util.spec_from_file_location(name,snapshot)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

ENGINE=load_engine()
Base=ENGINE.MyStrategy
parse_time=ENGINE.parse_time
probability=ENGINE.survival_block_probability
PARAMS=ENGINE.SURVIVAL_PARAMETERS
GAMMA=ENGINE.DISCOUNT_RATE
WEIGHTS=ENGINE.WEIGHT_RANK


def base_group(r,now):
    if r['missed']:return 'missed_load'
    if parse_time(r.get('departure_time',''))<now:return 'sched_overdue'
    if r['origin']=='initial':return 'sched_init_active'
    lead=(parse_time(r.get('departure_time',''))-r['entry'])/3600
    bucket='<24h' if lead<24 else '24-72h' if lead<=72 else '>72h'
    prefix='sched_disch_active' if r['origin']=='DISCHARGE' else 'sched_truck_active'
    return prefix+'|'+bucket


def ledger(initial,events,rotations,cutoff,with_probes=False):
    """Identity-only labels/snapshots, with outcomes censored at cutoff."""
    records={c['container_id']:dict(c,origin='initial',entry=START,end=cutoff,exit=None,missed=False) for c in initial['containers']}
    active=dict(records);seen=set(active);missed=set()
    eligible_ro=[r for r in rotations if START<=r.start and r.end<cutoff]
    probes=[]
    for r in eligible_ro:
        probes.append((r.start,0,r))
        if with_probes:
            for h in (6,12,24,48,96):
                t=r.start-h*3600
                if t>=START:probes.append((t,h,r))
    probes.sort(key=lambda p:(p[0],p[2].vessel,p[2].number,p[1]))
    index=0;snapshots={};drift_counts={}
    for e in events:
        now=parse_time(e.timestamp)
        if now>=cutoff:break
        while index<len(probes) and probes[index][0]<=now:
            t,h,r=probes[index]
            queue={cid for cid,x in active.items() if x.get('vessel_id','')==r.vessel}
            qelig=sum(cid not in missed for cid in queue)
            drift_counts[(r.vessel,r.end,h)]=qelig
            if h==0:snapshots[(r.vessel,r.end)]=queue
            index+=1
        if e.type in ARRIVALS:
            record=dict(vars(e),origin=e.type,entry=now,end=cutoff,exit=None,missed=e.container_id in missed)
            records[e.container_id]=active[e.container_id]=record;seen.add(e.container_id)
        else:
            r=active.pop(e.container_id,None)
            if r is not None:r.update(end=now,exit=e.type)
            elif e.type=='LOAD' and e.container_id not in seen:missed.add(e.container_id)
    # Probes after the final event still use the known resident state.
    while index<len(probes):
        t,h,r=probes[index]
        queue={cid for cid,x in active.items() if x.get('vessel_id','')==r.vessel}
        drift_counts[(r.vessel,r.end,h)]=sum(cid not in missed for cid in queue)
        if h==0:snapshots[(r.vessel,r.end)]=queue
        index+=1
    rows=[];opportunities=Counter()
    for ro in eligible_ro:
        queue=snapshots[(ro.vessel,ro.end)]
        loads=sum(records[cid]['exit']=='LOAD' and ro.start<=records[cid]['end']<=ro.end for cid in queue)
        for cid in queue:opportunities[base_group(records[cid],ro.start)]+=1
        rows.append(dict(vessel=ro.vessel,rotation_number=ro.number,end=ro.end,queue=len(queue),
                         eligible_queue=drift_counts[(ro.vessel,ro.end,0)],loads=loads,
                         hours=(ro.end-ro.start)/3600,alpha=loads/len(queue) if queue else 0))
    return records,rows,opportunities,drift_counts


@dataclass(frozen=True)
class Call:
    vessel:str
    number:int
    start:float
    end:float
    discharge_start:float
    discharge_end:float


def calibrate(initial,events,schedule):
    rotations=[]
    for v in schedule['vessels']:
        for r in v.get('rotations',[]):
            rotations.append(Call(v['vessel_id'],r['rotation_number'],parse_time(r['load_start']),parse_time(r['load_end']),
                                  parse_time(r['discharge_start']),parse_time(r['discharge_end'])))
    rotations.sort(key=lambda r:(r.end,r.vessel))
    fit_events=[e for e in events if parse_time(e.timestamp)<CUT]
    records,rows,opps,counts=ledger(initial,fit_events,rotations,CUT,True)
    assert len(rows)==13
    rate=statistics.median(r['loads']/r['hours'] for r in rows)
    active={g:n for g,n in opps.items() if g not in ('missed_load','sched_overdue')}
    active_alpha=sum(PARAMS[g][0]*n for g,n in active.items())/sum(active.values())
    overdue_ratio=PARAMS['sched_overdue'][0]/active_alpha
    age_rates={};age_rows=[]
    for origin in ('DISCHARGE','TRUCK_RECV'):
        rs=[r for r in records.values() if r['origin']==origin]
        cells=[]
        for label,lo,hi in (('<=72h',0,72),('>72h',72,inf)):
            hours=sum(max(0,min((r['end']-r['entry'])/3600,hi)-lo) for r in rs)
            exits=sum(r['exit']=='TRUCK_DLVR' and lo<=(r['end']-r['entry'])/3600<hi for r in rs)
            mu=exits/hours;cells.append(mu)
            age_rows.append(dict(origin=origin,age=label,container_hours=hours,truck_exits=exits,mu=mu))
        age_rates[origin]=cells
    drift=[]
    completed={ (r.vessel,r.end):r for r in rotations if START<=r.start and r.end<CUT }
    for h in (0,6,12,24,48,96):
        for phase in ('before_discharge_end','at_or_after_discharge_end'):
            ratios=[]
            for key,r in completed.items():
                t=r.start-h*3600
                if t<START or ((t<r.discharge_end)!=(phase=='before_discharge_end')):continue
                denom=counts[(r.vessel,r.end,0)]
                if denom:ratios.append(counts[(r.vessel,r.end,h)]/denom)
            drift.append(dict(hours_before_load=h,phase=phase,n=len(ratios),median_ratio=statistics.median(ratios) if ratios else None))
    # Predeclared calibration-only gate: require >=90% built queue at BOTH 6h
    # and 12h pre-load probes before discharge completion to use discharge_start.
    probe=[r for r in drift if r['phase']=='before_discharge_end' and r['hours_before_load'] in (6,12)]
    gate='discharge_start' if len(probe)==2 and all(r['n'] and r['median_ratio']>=.90 for r in probe) else 'discharge_end'
    _,oracle_rows,_,_=ledger(initial,events,rotations,END)
    oracle={(r['vessel'],r['end']):r['alpha'] for r in oracle_rows}
    config=dict(C_rate=rate,active_alpha=active_alpha,overdue_ratio=overdue_ratio,age_rates=age_rates,queue_gate=gate)
    save('calibration.json',dict(fit_cutoff_exclusive='2025-01-13T00:00:00',fit_rotations=rows,active_opportunities=dict(active),
                               config=config,age_hazards=age_rows,queue_drift=drift,
                               gate_rule='Use discharge_start only if before-discharge-end median queue ratios at both 6h and 12h are >=.90; otherwise discharge_end.',
                               notes=['C_rate is median(actual LOADs from start queue / window hours), fit only on 13 complete early windows.',
                                      'Active alpha is opportunity-weighted over the frozen active cohorts; overdue scaling is frozen overdue alpha / this mean.',
                                      'Age hazards use all origin arrivals at risk, including missed_load, with competing LOAD exits and Day12 censoring.',
                                      'Q_elig excludes historically observed missed_load IDs. Probes before the initial snapshot are excluded.',
                                      'Oracle uses all-resident start queues for 19 complete full-train windows; out-of-horizon calls retain frozen alpha. It is NON-DEPLOYABLE.']))
    save('oracle_inputs_NONDEPLOYABLE.json',oracle_rows)
    return rotations,config,oracle


def piecewise_probability(ax,mx,tx,sx,lx,ac,mc,tc,sc,lc):
    """Exact discounted pair integral with up to two age-hazard switches.

    sx/sc are hours until the hazard switch; lx/lc are the later hazards.
    Scheduled jumps retain the frozen clamped-time tie rule.
    """
    if not isfinite(sx) and not isfinite(sc):return probability(ax,mx,tx,ac,mc,tc)
    times=sorted(set([0.0]+[s for s in (sx,sc) if isfinite(s) and s>0]))
    segments=[];survival=1.0
    for i,a in enumerate(times):
        b=times[i+1] if i+1<len(times) else inf
        ux=lx if a>=sx else mx;uc=lc if a>=sc else mc
        rate=ux+uc+GAMMA
        area=survival*ux/rate*(-expm1(-rate*(b-a)) if isfinite(b) else 1.0)
        segments.append((a,b,ux,rate,survival,area))
        if isfinite(b):survival*=exp(-rate*(b-a))
    tail_after=[0.0]*len(segments)
    for i in range(len(segments)-2,-1,-1):tail_after[i]=tail_after[i+1]+segments[i+1][5]
    def tail(t):
        if not isfinite(t):return 0.0
        for i,(a,b,ux,rate,surv,area) in enumerate(segments):
            if t<b:
                at=surv*exp(-rate*(t-a))
                return at*ux/rate*(-expm1(-rate*(b-t)) if isfinite(b) else 1.0)+tail_after[i]
        return 0.0
    bg=sum(s[5] for s in segments)
    jx=ax>0 and isfinite(tx);jc=ac>0 and isfinite(tc)
    if jx:bg-=ax*tail(tx)
    if jc:bg-=ac*tail(tc)
    if jx and jc:bg+=ax*ac*tail(max(tx,tc))
    atom=0.0
    if jx:
        integral_x=mx*min(tx,sx)+(lx*max(0,tx-sx) if isfinite(sx) else 0)
        integral_c=mc*min(tx,sc)+(lc*max(0,tx-sc) if isfinite(sc) else 0)
        phi=(1.0 if tx>tc else .5 if tx==tc else 0.0) if jc else 0.0
        atom=ax*exp(-integral_x-integral_c-GAMMA*tx)*(1-ac*phi)
    return min(1.0,max(0.0,bg+atom))


class LabStrategy(Base):
    def __init__(self,mode,fit,rotations,value=None,oracle=None,use_cache=True,schedule_path=None):
        super().__init__(schedule_path=schedule_path,use_cache=use_cache)
        self.mode=mode;self.fit=fit;self.calls=rotations;self.value=value;self.oracle=oracle
        self.cutoff=mode.startswith('L1')
        self.queue_mode=mode.startswith('L2')
        self.age_mode=mode.startswith('L3')
    def initialize(self,layout,initial):
        self.progress={};self.queue=defaultdict(Counter);self.category={};self.age_heap=[];self.age_registered=set();self.pending_incoming=None;self._q_cache={}
        self.call_by_end={(r.vessel,r.end):r for r in self.calls}
        self.calls_by_vessel=defaultdict(list)
        for r in self.calls:self.calls_by_vessel[r.vessel].append(r)
        for rs in self.calls_by_vessel.values():rs.sort(key=lambda r:r.end)
        for c in initial.get('containers',[]):self.category[c['container_id']]=(c.get('port_of_discharge',''),WEIGHTS.get(c.get('weight_class','MEDIUM'),1))
        super().initialize(layout,initial)
        if self.queue_mode:
            for cid,m in self._meta.items():self.queue[m.vessel][self.category[cid]]+=1
        self.scan_order=sorted(range(len(self._stacks)),key=lambda i:(self._stacks[i][0],-self._stacks[i][1],-self._stacks[i][2])) if self.mode=='L5a_reverse_scan' else range(len(self._stacks))
    def _remember(self,container):
        cid=container.container_id;new=cid not in self._meta
        self.category[cid]=(container.port_of_discharge,WEIGHTS.get(container.weight_class,1))
        meta=super()._remember(container)
        if self.queue_mode and new and cid not in self.observed_missed_loads:self.queue[meta.vessel][self.category[cid]]+=1
        if self.age_mode and cid not in self._initial_ids and cid not in self.age_registered:
            import heapq
            self.age_registered.add(cid)
            deadline=self._arrival_times.get(cid,self.current_time)+72*3600
            if deadline>=self.current_time:heapq.heappush(self.age_heap,(deadline,cid))
        return meta
    def _advance_time(self,timestamp):
        super()._advance_time(timestamp)
        if self.age_mode:
            import heapq
            while self.age_heap and self.age_heap[0][0]<self.current_time:
                due,cid=heapq.heappop(self.age_heap)
                i=self._cid_index.get(cid)
                if i is not None:self._keys_dirty.add(i)
    def _progress_update(self,e):
        rotation=self._next_by_vessel.get(e.vessel_id)
        if rotation is None or not rotation.start<=self.current_time<=rotation.end:return
        key=e.vessel_id,rotation.end;cat=(e.port_of_discharge,WEIGHTS.get(e.weight_class,1))
        if cat>self.progress.get(key,('',-1)):
            self.progress[key]=cat
            if self.cutoff:self._generation+=1
    def on_event(self,e):
        is_new=e.type in ARRIVALS and e.container_id not in self._meta
        super().on_event(e);self.event=e
        self.pending_incoming=e.container_id if self.queue_mode and is_new and e.container_id not in self.observed_missed_loads else None
        self._q_cache={}
        if e.type=='LOAD' and (self.queue_mode or (self.cutoff and self.mode!='L1a_found_only')):self._progress_update(e)
    def on_container_retrieved(self,cid,pos,reshuffles):
        self._q_cache={}
        if self.queue_mode and cid in self._meta and cid not in self.observed_missed_loads:
            m=self._meta[cid];self.queue[m.vessel][self.category[cid]]-=1
        if self.cutoff and self.mode=='L1a_found_only' and self.event.type=='LOAD':self._progress_update(self.event)
        super().on_container_retrieved(cid,pos,reshuffles)
    def _survival_key(self,cid):
        g,when=super()._survival_key(cid)
        meta=self._meta[cid];ro=self._next_by_vessel.get(meta.vessel)
        if self.cutoff and ro is not None and g!='missed_load' and ro.start<=self.current_time<=ro.end:
            progress=self.progress.get((meta.vessel,ro.end))
            if progress is not None and self.category[cid]<progress:
                rs=self.calls_by_vessel[meta.vessel]
                ro=next((r for r in rs if r.end>ro.end),None)
                if ro is None:g,when='expired_no_rot',inf
                else:g,when='sched_overdue',ro.start+meta.fraction*(ro.end-ro.start)
        end=ro.end if ro is not None else inf
        key=(g,when,meta.vessel,end)
        if self.age_mode:
            origin=self._origins.get(cid,'initial')
            if origin in ARRIVALS:
                deadline=self._arrival_times.get(cid,self.current_time)+72*3600
                extra=(origin,deadline) if self.mode=='L3b_age72_exact_piecewise' else (origin,self.current_time>deadline)
            else:extra=('initial',inf)
            key+=extra
        return key
    def q_eligible(self,vessel,end):
        key=vessel,end
        if key in self._q_cache:return self._q_cache[key]
        call=self.call_by_end.get(key);progress=self.progress.get(key)
        counts=self.queue[vessel]
        filtering=bool(call and call.start<=self.current_time<=call.end and progress is not None)
        n=sum(count for cat,count in counts.items() if not filtering or cat>=progress)
        cid=self.pending_incoming
        if cid is not None and self._meta[cid].vessel==vessel and (not filtering or self.category[cid]>=progress):n-=1
        self._q_cache[key]=n
        return n
    def _finish_placement(self,index):
        pos=super()._finish_placement(index)
        self.pending_incoming=None;self._q_cache={}
        return pos
    def _state(self,key):
        g,when,vessel,end=key[:4];alpha,mu=PARAMS[g]
        if self.mode=='D_oracle_alpha' and g.startswith('sched_'):
            alpha=self.oracle.get((vessel,end),alpha)
        if self.queue_mode and g.startswith('sched_'):
            call=self.call_by_end.get((vessel,end))
            apply=call is not None
            if apply and self.mode=='L2c_post_discharge_queue':apply=self.current_time>=getattr(call,self.fit['queue_gate'])
            if apply:
                forecast=self.fit['C_rate']*(call.end-call.start)/3600
                if self.mode=='L2c_post_discharge_queue' and call.start<=self.current_time<=call.end:
                    forecast*=max(0,(call.end-self.current_time)/(call.end-call.start))
                aq=min(.90,max(.10,forecast/max(self.q_eligible(vessel,end),1)))
                if g=='sched_overdue':aq*=self.fit['overdue_ratio']
                alpha=.5*(alpha+aq) if self.mode=='L2b_blended_queue' else aq
        switch=inf;later=mu
        if self.age_mode and key[4] in ARRIVALS:
            early,later=self.fit['age_rates'][key[4]]
            if self.mode=='L3a_age72_step':mu=later if key[5] else early;switch=inf;later=mu
            else:
                switch=max(0,(key[5]-self.current_time)/3600)
                mu=early if switch>0 else later
                if switch==0:switch=inf
        return alpha,mu,max(0,(when-self.current_time)/3600),switch,later
    def _pair(self,x,c,xkey,ckey):
        ax,mx,tx,sx,lx=x;ac,mc,tc,sc,lc=c
        if self.mode=='L3b_age72_exact_piecewise':return piecewise_probability(ax,mx,tx,sx,lx,ac,mc,tc,sc,lc)
        p=probability(ax,mx,tx,ac,mc,tc)
        if self.mode=='L1b_cutoff_plus_unclamped' and ax and ac and isfinite(xkey[1]) and isfinite(ckey[1]):
            old=1 if tx>tc else .5 if tx==tc else 0
            new=1 if xkey[1]>ckey[1] else .5 if xkey[1]==ckey[1] else 0
            p+=ax*ac*exp(-(mx+mc+GAMMA)*tx)*(old-new)
        return min(1,max(0,p))
    def _place_survival(self,event):
        self._q_cache={}
        ckey=self._survival_key(event.container_id);c=self._state(ckey)
        states={};risks={};urgencies={};costs=[None]*len(self._stacks)
        floors=[inf]*len(self._stacks);fragilities=[0.0]*len(self._stacks)
        for i,(_,_,_,maximum) in enumerate(self._stacks):
            if self._heights[i]>=maximum:continue
            cost=0.0;floor=inf;fragile=0.0
            for key in self._survival_lists[i]:
                risk=risks.get(key)
                if risk is None:
                    x=self._state(key);states[key]=x
                    risk=self._pair(x,c,key,ckey);risks[key]=risk
                    if self.mode=='L4b_floor_degrade':urgencies[key]=probability(x[0],x[1],x[2],0,0,inf)
                cost+=risk
                if self.mode=='L4a_gap':floor=min(floor,key[1])
                if self.mode=='L4b_floor_degrade':fragile=max(fragile,urgencies[key])
            costs[i]=cost;floors[i]=floor;fragilities[i]=fragile
        lowest={}
        if self.mode=='L5b_reloc_2step':
            for block,indices in self._block_indices.items():
                lowest[block]=sorted((i for i in indices if costs[i] is not None),key=lambda i:(self._heights[i],self._stacks[i][1],self._stacks[i][2]))[:2]
        uc=probability(c[0],c[1],c[2],0,0,inf) if self.mode=='L4b_floor_degrade' else 0
        best=None;chosen=None
        for i in self.scan_order:
            direct=costs[i]
            if direct is None:continue
            block,bay,row,_=self._stacks[i];height=self._heights[i];score=direct
            if self.mode=='L4a_gap' and height and floors[i]>ckey[1]:
                floor_tau=max(0,(floors[i]-self.current_time)/3600)
                score+=self.value*min((floor_tau-c[2])/240,1)
            elif self.mode=='L4b_floor_degrade':score+=self.value*max(0,uc-fragilities[i])
            elif self.mode=='L5b_reloc_2step':
                target=next((j for j in lowest[block] if j!=i),None)
                score*=1+self.value*(costs[target] if target is not None else 0)
            elif self.mode=='L5c_catcher_reserve' and ckey[0] not in ('missed_load','expired_no_rot') and bay==1 and row<=self.value:score+=.50
            rank=score,height
            if best is None or rank<best:best,chosen=rank,i
        if chosen is None:raise RuntimeError('No non-full stack')
        return self._finish_placement(chosen)


class Traced(PlacementStrategy):
    def __init__(self,base):self.base=base
    def initialize(self,layout,initial):
        self.origins={c['container_id']:'initial' for c in initial['containers']}
        self.trace=[];self.buckets=defaultdict(lambda:dict(found=0,reshuffles=0))
        self.base.initialize(layout,initial)
    def on_event(self,e):
        self.event=e
        if e.type in ARRIVALS:self.origins[e.container_id]=e.type
        self.base.on_event(e)
    def place_container(self,yard,e):
        p=self.base.place_container(yard,e)
        self.trace.append((e.event_id,p.block,p.bay,p.row,p.tier));return p
    def on_container_retrieved(self,cid,pos,reshuffles):
        period='early' if parse_time(self.event.timestamp)<CUT else 'late'
        for key in (period,'exit/'+self.event.type,'origin/'+self.origins[cid]):
            self.buckets[key]['found']+=1;self.buckets[key]['reshuffles']+=reshuffles
        self.base.on_container_retrieved(cid,pos,reshuffles)

class RepeatedSimulator(Simulator):
    def __init__(self,yard,strategy):super().__init__(yard,strategy);self.moves=Counter();self.repeated=0
    def _do_retrieval(self,target,above):
        for cid in above:
            self.repeated+=self.moves[cid]>0;self.moves[cid]+=1
        super()._do_retrieval(target,above)


def evaluate(name,base,inputs,config=None):
    if (OUT/(name+'.json')).exists():
        result=json.loads((OUT/(name+'.json')).read_text());print('REUSED COMPLETED',name,flush=True);return result
    layout,initial,events=inputs
    yard=YardState(layout);yard.load_initial_state(initial)
    wrapper=Traced(base);wrapper.initialize(layout,initial)
    sim=RepeatedSimulator(yard,wrapper)
    started=time.perf_counter();stats=sim.run(events);elapsed=time.perf_counter()-started
    assert stats.errors==stats.hard_constraint_violations==stats.placement_failures==0
    assert stats.retrieval_not_found==1333 and stats.final_yard_count==6311
    assert wrapper.buckets['early']['found']==5042 and wrapper.buckets['late']['found']==3832
    h=hashlib.sha256(json.dumps(wrapper.trace,separators=(',',':')).encode()).hexdigest()
    r=dict(name=name,config=config,total=stats.total_reshuffles,excess=stats.total_reshuffles-1325,
           official_rate=stats.reshuffles_per_retrieval,score=stats.quantitative_score(),
           early=wrapper.buckets['early']['reshuffles'],late=wrapper.buckets['late']['reshuffles'],
           early_rate=wrapper.buckets['early']['reshuffles']/5042,late_rate=wrapper.buckets['late']['reshuffles']/3832,
           LOAD=wrapper.buckets['exit/LOAD']['reshuffles'],TRUCK_DLVR=wrapper.buckets['exit/TRUCK_DLVR']['reshuffles'],
           I=wrapper.buckets['origin/initial']['reshuffles'],D=wrapper.buckets['origin/DISCHARGE']['reshuffles'],R=wrapper.buckets['origin/TRUCK_RECV']['reshuffles'],
           repeated=sim.repeated,seconds=elapsed,summary=stats.summary(),slices=dict(wrapper.buckets),
           placements=len(wrapper.trace),trace_sha256=h,cache_guard_fallbacks=base.cache_guard_fallbacks)
    assert r['total']==r['early']+r['late']==r['LOAD']+r['TRUCK_DLVR']==r['I']+r['D']+r['R']
    save(name+'.json',r)
    print(name,'early',r['early'],'late',r['late'],'total',r['total'],'sec',round(elapsed,2),flush=True)
    return r


def self_tests(fit,rotations):
    # Independent quadrature with exact scheduled point masses and age switches.
    for sx,sc in ((inf,inf),(3,9),(9,3),(0,7)):
        ax,ac,mx,mc,lx,lc,tx,tc=.6,.4,.03,.02,.008,.006,5,11
        def integrated(mu,later,s,t):return mu*min(s,t)+(later*max(0,t-s) if isfinite(s) else 0)
        h=1500;bounds=sorted(set([0,h,tx,tc]+[s for s in (sx,sc) if isfinite(s) and s>0]))
        numerical=0
        for a,b in zip(bounds,bounds[1:]):
            n=max(1000,int((b-a)*100));dt=(b-a)/n
            for i in range(n):
                t=a+(i+.5)*dt
                ux=lx if t>=sx else mx
                surv=exp(-integrated(mx,lx,sx,t)-integrated(mc,lc,sc,t)-GAMMA*t)
                numerical+=ux*surv*(1-ax if t>tx else 1)*(1-ac if t>tc else 1)*dt
        numerical+=ax*exp(-integrated(mx,lx,sx,tx)-integrated(mc,lc,sc,tx)-GAMMA*tx)*(1-ac*(1 if tx>tc else .5 if tx==tc else 0))
        actual=piecewise_probability(ax,mx,tx,sx,lx,ac,mc,tc,sc,lc)
        assert abs(actual-numerical)<2e-7,(sx,sc,actual,numerical)
        if sx==sc==inf:assert actual==probability(ax,mx,tx,ac,mc,tc)
    # Raw order is retained when both clamped jump times are zero.
    strat=LabStrategy('L1b_cutoff_plus_unclamped',fit,rotations)
    assert strat._pair((1,0,0,inf,0),(1,0,0,inf,0),('g',1),('g',2))==1
    assert strat._pair((1,0,0,inf,0),(1,0,0,inf,0),('g',2),('g',1))==0
    # Synthetic two-call schedule checks progress, rollover, counts and age keys.
    def stamp(day,h=0):return f'2025-01-{day:02d}T{h:02d}:00:00'
    synthetic=OUT/'synthetic_schedule.json'
    synthetic.write_text(json.dumps(dict(vessels=[dict(vessel_id='V',ports=['P1','P2'],rotations=[
        dict(load_start=stamp(2),load_end=stamp(2,12)),dict(load_start=stamp(10),load_end=stamp(10,12))])])))
    calls=[Call('V',1,parse_time(stamp(2)),parse_time(stamp(2,12)),START,START),Call('V',2,parse_time(stamp(10)),parse_time(stamp(10,12)),START,START)]
    layout={'blocks':{'A':{'bays':2,'rows':1,'tiers':5}}}
    initial={'containers':[dict(container_id='resident',vessel_id='V',departure_time=stamp(11),port_of_discharge='P1',weight_class='HEAVY',position=dict(block='A',bay=1,row=1,tier=1))]}
    for mode in ('L1a_all_loads','L1a_found_only'):
        s=LabStrategy(mode,fit,calls,schedule_path=synthetic);s.initialize(layout,initial)
        e=Event(1,stamp(2,1),'LOAD','absent',vessel_id='V',port_of_discharge='P2',weight_class='LIGHT')
        s.on_event(e);key=s._survival_key('resident')
        assert (key[0]=='sched_overdue')==(mode=='L1a_all_loads')
        if mode=='L1a_all_loads':assert key[3]==calls[1].end
    s=LabStrategy('L2a_direct_queue',fit,calls,schedule_path=synthetic);s.initialize(layout,initial)
    yard=YardState(layout);yard.load_initial_state(initial)
    assert s.q_eligible('V',calls[0].end)==1
    e=Event(1,stamp(1),'DISCHARGE','new',vessel_id='V',port_of_discharge='P2',departure_time=stamp(11))
    s.on_event(e);assert s.q_eligible('V',calls[0].end)==1  # Incoming is not a resident yet.
    pos=s.place_container(yard,e);assert yard.place_container(e.to_container(),pos)
    assert s.q_eligible('V',calls[0].end)==2
    Simulator(yard,s).run([Event(2,stamp(1,1),'LOAD','resident',vessel_id='V',port_of_discharge='P1',weight_class='HEAVY')])
    assert s.q_eligible('V',calls[0].end)==1
    s.on_event(Event(3,stamp(1,2),'LOAD','missed',vessel_id='V'))
    s.on_event(Event(4,stamp(1,3),'DISCHARGE','missed',vessel_id='V',departure_time=stamp(11)))
    assert s.q_eligible('V',calls[0].end)==1
    # Cached/live sequences exercise callbacks, changing queue and age-72 boundary.
    modes=['L1a_all_loads','L1a_found_only','L1b_cutoff_plus_unclamped','L2a_direct_queue','L2b_blended_queue','L2c_post_discharge_queue','L3a_age72_step','L3b_age72_exact_piecewise']
    for mode in modes:
        traces=[]
        for cached in (True,False):
            s=LabStrategy(mode,fit,calls,schedule_path=synthetic,use_cache=cached);yard=YardState(layout);yard.load_initial_state(initial);s.initialize(layout,initial)
            trace=[]
            for day,cid in ((1,'a'),(3,'b'),(5,'c')):
                e=Event(day,stamp(day),'DISCHARGE',cid,vessel_id='V',port_of_discharge='P1',departure_time=stamp(11))
                s.on_event(e);p=s.place_container(yard,e);trace.append((p.block,p.bay,p.row,p.tier));assert yard.place_container(e.to_container(),p)
                if day==1:Simulator(yard,s).run([Event(2,stamp(2,1),'LOAD','resident',vessel_id='V',port_of_discharge='P1',weight_class='HEAVY')])
            traces.append((trace,yard.snapshot()))
        assert traces[0]==traces[1],mode
    save('self_tests.json',dict(passed=True,numerical_piecewise_cases=4,raw_phi_cases=2,synthetic_cached_live_modes=len(modes),
                               count_and_rollover_checks=True,notes='Age hazards apply literally to all new-origin cohorts, including missed_load; initial residents retain frozen mu.'))
    print('SELF TESTS PASSED',flush=True)


def table_text(rows):
    columns=['name','early_rate','late_rate','total','excess','LOAD','TRUCK_DLVR','I','D','R','repeated','seconds']
    out=[' | '.join(columns),' | '.join('---' for _ in columns)]
    for r in rows:
        out.append(' | '.join(f"{r[c]:.4f}" if c.endswith('_rate') else f"{r[c]:.2f}" if c=='seconds' else str(r[c]) for c in columns))
    return '\n'.join(out)+'\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    def no_test_access(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            value=str(args[0]).replace('\\','/')
            if '/data/test/' in value or value.startswith('data/test/'):raise RuntimeError('Test access forbidden in Phase 6B')
    sys.addaudithook(no_test_access)
    protected=['solution/__init__.py','solution/my_strategy.py','solution/survival_params.py','results/results.json']
    before={p:digest(p) for p in protected}
    checkpoint=json.loads((ROOT/'results/phase5b/phase5b_manifest.json').read_text())
    assert before['solution/my_strategy.py']==checkpoint['files']['solution/my_strategy.py']
    assert before['solution/survival_params.py']==checkpoint['files']['solution/survival_params.py']
    layout=json.loads((ROOT/'data/yard_layout.json').read_text());initial=json.loads((ROOT/'data/train/initial_state.json').read_text())
    events=read_events(str(ROOT/'data/train/events.jsonl'));schedule=json.loads((ROOT/'data/vessel_schedule.json').read_text())
    inputs=layout,initial,events
    control=evaluate('control_exact_copy',Base(),inputs)
    assert control['total']==6408 and control['trace_sha256']==checkpoint['train_placement_sha256'] and control['repeated']==1515
    calls,fit,oracle=calibrate(initial,events,schedule)
    print('FIT',fit,flush=True)
    self_tests(fit,calls)
    neutral=evaluate('neutral_lab_equivalence',LabStrategy('neutral',fit,calls),inputs)
    assert neutral['trace_sha256']==control['trace_sha256']
    diagnostic=evaluate('D_oracle_alpha',LabStrategy('D_oracle_alpha',fit,calls,oracle=oracle),inputs,dict(NONDEPLOYABLE=True,completed_full_train_rotations=len(oracle)))
    configs=[('L1a_all_loads',None),('L1a_found_only',None),('L1b_cutoff_plus_unclamped',None),
             ('L2a_direct_queue',None),('L2b_blended_queue',None),('L2c_post_discharge_queue',None),
             ('L3a_age72_step',None),('L3b_age72_exact_piecewise',None)]
    configs += [('L4a_gap',v) for v in (.02,.05,.10,.20)]
    configs += [('L4b_floor_degrade',v) for v in (.05,.15,.30)]
    configs += [('L5a_reverse_scan',None)]
    configs += [('L5b_reloc_2step',v) for v in (.25,.50,1.0)]
    configs += [('L5c_catcher_reserve',v) for v in (2,4)]
    assert len(configs)==21
    rows=[]
    for mode,value in configs:
        name=mode+(f'_{value:.2f}' if isinstance(value,float) else '_'+str(value) if value is not None else '')
        r=evaluate(name,LabStrategy(mode,fit,calls,value=value),inputs,dict(mode=mode,value=value))
        rows.append(r);save('ablations.json',rows);(OUT/'tables.txt').write_text(table_text([control,diagnostic]+rows))
    after={p:digest(p) for p in protected};assert before==after
    save('verification.json',dict(protected_hashes_before=before,protected_hashes_after=after,unchanged=True,test_access_guard_enabled=True,test_data_accessed=False,
        engine_copy_sha256=digest('scripts/phase6b_engine_snapshot.py'),source_and_copy_identical=(ROOT/'solution/my_strategy.py').read_bytes()==(ROOT/'scripts/phase6b_engine_snapshot.py').read_bytes(),
        exact_copy_control_6408=True,neutral_lab_trace_match=True,placements=10385,trace_sha256=control['trace_sha256'],
        isolated_runs=len(rows),oracle_runs=1,control_runs=2,fit_cutoff_exclusive='2025-01-13T00:00:00',
        all_errors_violations_failures=0,fit_configuration=fit,
        interpretation_notes=['Oracle alpha is a non-deployable yardstick, not a mathematical upper bound on reshuffle improvement.',
          'L1 order uses lexicographic port + weight rank; exact on this schedule and avoids port-ID heuristics.',
          'L2 live counters exclude missed and passed categories; L2 does not otherwise apply L1 rollover.',
          'L3 replaces mu for all new-origin cohorts including missed_load, as specified; initial mu is unchanged.',
          'L5 reverse order keeps block ascending and reverses bay,row via (block,-bay,-row).',
          'L5 two-step uses second-lowest non-full stack if the candidate is lowest; no alternative means zero extra multiplier.',
          'No combinations, default changes, or test evaluations were performed.']))
    print(table_text([control,diagnostic]+rows),flush=True)

if __name__=='__main__':main()
