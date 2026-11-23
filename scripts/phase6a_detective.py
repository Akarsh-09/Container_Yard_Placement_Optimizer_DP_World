"""Train-only structural detective work; never opens test or alters solution.

Feature/outcome analyses stop strictly before Jan 13. Whole-train mechanics
are descriptive context only. The observer must reproduce the frozen trace.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from bisect import bisect_left
import datetime
import hashlib
import json
from math import sqrt, isfinite
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solution.my_strategy import MyStrategy, parse_time, WEIGHT_RANK
from src.event_reader import read_events
from src.simulator import Simulator
from src.yard_state import YardState

OUT=ROOT/'results/phase6a'
START=parse_time('2025-01-01T00:00:00')
CUT=parse_time('2025-01-13T00:00:00')
ARRIVALS={'DISCHARGE','TRUCK_RECV'}
KS=(1,2,4,6,12)
AGE_LIMITS=(0,24,72,120,240,float('inf'))
AGE_NAMES=('0-24h','24-72h','72-120h','120-240h','>240h')

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def period(t):return 'Days1-12' if t<CUT else 'Days13-20'
def percentiles(values):
    a=sorted(values)
    if not a:return dict(n=0)
    result=dict(n=len(a))
    for p in (5,10,25,50,75,90,95):
        at=(len(a)-1)*p/100;i=int(at)
        result['p'+str(p)]=a[i]+(a[min(i+1,len(a)-1)]-a[i])*(at-i)
    return result
def age_bucket(hours):
    return '<24h' if hours<24 else '24-72h' if hours<72 else '72-168h' if hours<168 else '>=168h'
def lead_bucket(hours):
    return 'past' if hours<0 else '0-24h' if hours<24 else '24-72h' if hours<72 else '72-168h' if hours<168 else '168-288h' if hours<288 else '>=288h'
def relation(raw,etd):
    if raw>1e11:return 'missing'
    return 'matches' if abs(raw-etd)<1 else 'later' if raw>etd else 'earlier'
def correlation(xs,ys):
    if len(xs)<2:return None
    x=statistics.mean(xs);y=statistics.mean(ys)
    a=sum((u-x)**2 for u in xs);b=sum((v-y)**2 for v in ys)
    return sum((u-x)*(v-y) for u,v in zip(xs,ys))/sqrt(a*b) if a*b else None
def rank(values):
    out=[0]*len(values)
    order=sorted(range(len(values)),key=lambda i:values[i]);a=0
    while a<len(order):
        b=a+1
        while b<len(order) and values[order[b]]==values[order[a]]:b+=1
        for i in order[a:b]:out[i]=(a+b-1)/2
        a=b
    return out

class Audit:
    def __init__(self,layout,initial,schedule,yard):
        self.layout=layout;self.initial=initial;self.yard=yard
        self.rotations=[];self.by_vessel=defaultdict(list)
        for vessel in schedule['vessels']:
            for ro in vessel.get('rotations',[]):
                row=dict(vessel=vessel['vessel_id'],number=ro['rotation_number'],
                         start=parse_time(ro['load_start']),end=parse_time(ro['load_end']),etd=parse_time(ro['etd']))
                self.by_vessel[row['vessel']].append(row)
                if START<=row['start'] and row['end']<CUT:self.rotations.append(row)
        self.rotations.sort(key=lambda r:(r['start'],r['vessel'],r['number']))
        for rows in self.by_vessel.values():rows.sort(key=lambda r:r['end'])
        self.snap_index=0;self.queues={};self.load_events=Counter();self.found_loads=Counter()
        self.records={c['container_id']:dict(c,origin='initial',entry=START,end=CUT,exit=None,exit_order=None) for c in initial['containers']}
        self.progress={};self.passed={};self.passed_exposures=Counter();self.passed_placements=Counter()
        self.scan_exposures=Counter();self.unique_scan_placements=set();self.unique_passed_placements=set();self.load_reversals=Counter()
        self.trace=[];self.event=None
        self.last_landing={};self.moves=Counter();self.repeated=Counter();self.landing_prefix=Counter()
        self.simultaneous=Counter();self.empty_retrieval=defaultdict(Counter);self.empty_own=defaultdict(Counter)
        self.heights={};self.empties=Counter()
        for b,info in layout['blocks'].items():
            for bay in range(1,info['bays']+1):
                for row in range(1,info['rows']+1):
                    h=yard.get_stack_height(b,bay,row);self.heights[(b,bay,row)]=h
                    self.empties[b]+=h==0
        self.empty_placements=defaultdict(Counter);self.all_placements=Counter()
        self.affected=set()
    def sync_stack(self,b,bay,row):
        key=b,bay,row;old=self.heights[key];new=self.yard.get_stack_height(b,bay,row)
        self.empties[b]+=(new==0)-(old==0);self.heights[key]=new
    def sync_block(self,b):
        info=self.layout['blocks'][b]
        for bay in range(1,info['bays']+1):
            for row in range(1,info['rows']+1):self.sync_stack(b,bay,row)
    @staticmethod
    def key(ro):return ro['vessel'],ro['number']
    def window(self,vessel,t):
        return next((r for r in self.by_vessel.get(vessel,()) if r['start']<=t<=r['end'] and r['end']<CUT),None)
    def before_event(self,e):
        t=parse_time(e.timestamp);self.event=e
        while self.snap_index<len(self.rotations) and self.rotations[self.snap_index]['start']<=t:
            ro=self.rotations[self.snap_index];queue={}
            for cid,(pos,c) in self.yard.get_all_containers().items():
                if c.vessel_id==ro['vessel']:
                    queue[cid]=dict(tier=pos.tier,height=self.yard.get_stack_height(pos.block,pos.bay,pos.row))
            self.queues[self.key(ro)]=queue;self.snap_index+=1
        if t>=CUT:return
        if e.type in ARRIVALS:
            self.records[e.container_id]=dict(vars(e),origin=e.type,entry=t,end=CUT,exit=None,exit_order=None)
        elif e.type=='LOAD':
            ro=self.window(e.vessel_id,t)
            if ro:
                key=self.key(ro);self.load_events[key]+=1
                if self.yard.get_container_position(e.container_id) is not None:self.found_loads[key]+=1
                category=(e.port_of_discharge,WEIGHT_RANK.get(e.weight_class,1))
                previous=self.progress.get(key)
                if previous is not None and category<previous:self.load_reversals[key]+=1
                self.progress[key]=max(self.progress.get(key,category),category)
    def on_retrieved(self,cid):
        e=self.event;t=parse_time(e.timestamp)
        if t<CUT:
            r=self.records[cid];assert r['exit'] is None
            r.update(end=t,exit=e.type,exit_order=e.event_id)
    def placed(self,s,yard,e,p):
        self.trace.append((e.event_id,p.block,p.bay,p.row,p.tier))
        cohort,when=s._survival_key(e.container_id)
        hours=max(0,(when-s.current_time)/3600)
        jump='inf' if not isfinite(hours) else '<24h' if hours<24 else '24-72h' if hours<=72 else '>72h'
        self.all_placements[cohort]+=1
        if p.tier==1:self.empty_placements[cohort][jump]+=1
        if s.current_time>=CUT:return
        active={r['vessel']:r for r in self.rotations if r['start']<=s.current_time<=r['end']}
        passed_windows=set();any_scanned=False
        for i,(b,bay,row,maximum) in enumerate(s._stacks):
            if s._heights[i]>=maximum:continue
            for cid in s._contents[i]:
                meta=s._meta[cid];ro=active.get(meta.vessel)
                if not ro:continue
                key=self.key(ro);self.scan_exposures[key]+=1;any_scanned=True
                progress=self.progress.get(key)
                r=self.records[cid];category=(r.get('port_of_discharge',''),WEIGHT_RANK.get(r.get('weight_class','MEDIUM'),1))
                if progress is not None and category<progress:
                    self.passed_exposures[key]+=1;passed_windows.add(key)
                    self.passed.setdefault((key,cid),dict(first_event=e.event_id,first_time=s.current_time))
        if any_scanned:self.unique_scan_placements.add(e.event_id)
        if passed_windows:self.unique_passed_placements.add(e.event_id)
        for key in passed_windows:self.passed_placements[key]+=1
    def retrieval_start(self,e):
        t=parse_time(e.timestamp);p=period(t)
        for b,n in self.empties.items():self.empty_retrieval[(p,b)]['zero' if n==0 else 'positive']+=1
        pos=self.yard.get_container_position(e.container_id)
        if pos:self.empty_own[(p,pos.block)]['zero' if self.empties[pos.block]==0 else 'positive']+=1
    def blockers(self,target,above):
        p=period(parse_time(self.event.timestamp));self.simultaneous[(p,len(above))]+=1
        self.affected={self.yard.get_container_position(target).block}
        for cid in above:
            if self.moves[cid]:
                self.repeated[(p,'all')]+=1
                landing=self.last_landing[cid]
                for k in KS:
                    if landing['rank']<=k:self.repeated[(p,k)]+=1
            self.moves[cid]+=1
    def landing(self,pos):
        p=period(parse_time(self.event.timestamp))
        rank=(pos.bay-1)*self.layout['blocks'][pos.block]['rows']+pos.row
        for k in KS:
            if rank<=k:self.landing_prefix[(p,k)]+=1
        return dict(rank=rank,block=pos.block,height=pos.tier-1,event_id=self.event.event_id)

class ObservedStrategy(MyStrategy):
    def __init__(self,audit):super().__init__();self.audit=audit
    def on_event(self,e):self.audit.before_event(e);super().on_event(e)
    def place_container(self,yard,e):
        p=super().place_container(yard,e);self.audit.placed(self,yard,e,p);return p
    def on_container_retrieved(self,cid,pos,reshuffles):
        self.audit.on_retrieved(cid);super().on_container_retrieved(cid,pos,reshuffles)

class ObservedSimulator(Simulator):
    def _handle_placement(self,e):
        super()._handle_placement(e)
        p=self.yard.get_container_position(e.container_id)
        if p:self.strategy.audit.sync_stack(p.block,p.bay,p.row)
    def _handle_retrieval(self,e):
        self.strategy.audit.retrieval_start(e);super()._handle_retrieval(e)
    def _do_retrieval(self,target,above):
        a=self.strategy.audit;a.blockers(target,above)
        super()._do_retrieval(target,above)
        for block in a.affected:a.sync_block(block)
    def _find_reshuffle_position(self,block,container):
        pos=super()._find_reshuffle_position(block,container)
        if pos:
            a=self.strategy.audit;a.last_landing[container.container_id]=a.landing(pos);a.affected.add(pos.block)
        return pos

def rotation_analysis(a):
    rows=[];features=defaultdict(lambda:dict(loaded=0,not_loaded=0,expected_loaded=0.0));excluded_other_exits=0
    ages=defaultdict(list)
    for ro in a.rotations:
        key=a.key(ro);queue=a.queues[key]
        loaded={cid for cid in queue if a.records[cid]['exit']=='LOAD' and ro['start']<=a.records[cid]['end']<=ro['end']}
        q=len(queue);alpha=len(loaded)/q if q else None
        other=sum(a.records[cid]['exit'] is not None and a.records[cid]['end']<=ro['end'] and cid not in loaded for cid in queue)
        rows.append(dict(vessel_id=ro['vessel'],rotation_number=ro['number'],load_window_hours=(ro['end']-ro['start'])/3600,
                         yard_queue_at_load_start=q,actual_loads_from_queue=len(loaded),empirical_alpha=alpha,
                         total_load_events_in_window=a.load_events[key],found_load_events_in_window=a.found_loads[key],
                         absent_load_events_in_window=a.load_events[key]-a.found_loads[key],
                         loads_of_containers_not_in_start_queue=a.found_loads[key]-len(loaded),other_exits_from_queue=other,
                         queue_still_resident_at_window_end=q-len(loaded)-other))
        if alpha is not None and alpha<.70:
            excluded_other_exits+=other
            comparison_alpha=len(loaded)/(q-other)
            for cid,position in queue.items():
                r=a.records[cid]
                if cid not in loaded and r['exit'] is not None and r['end']<=ro['end']:continue
                age=(ro['start']-r['entry'])/3600
                fs=dict(origin=r['origin'],raw_etd_relation=relation(parse_time(r.get('departure_time','')),ro['etd']),
                        age_at_load_start=age_bucket(age),port=r.get('port_of_discharge',''),
                        weight=r.get('weight_class','MEDIUM'),stack_tier=str(position['tier']))
                for f,v in fs.items():
                    cell=features[(f,v)];cell['loaded' if cid in loaded else 'not_loaded']+=1;cell['expected_loaded']+=comparison_alpha
                ages['loaded' if cid in loaded else 'not_loaded'].append(age)
    valid=[r for r in rows if r['yard_queue_at_load_start']]
    correlations={}
    for name,xs in [('inverse_queue',[1/r['yard_queue_at_load_start'] for r in valid]),
                    ('window_hours_over_queue',[r['load_window_hours']/r['yard_queue_at_load_start'] for r in valid]),
                    ('rotation_number',[r['rotation_number'] for r in valid]),
                    ('window_hours',[r['load_window_hours'] for r in valid]),
                    ('found_LOAD_share',[r['found_load_events_in_window']/r['total_load_events_in_window'] for r in valid])]:
        ys=[r['empirical_alpha'] for r in valid]
        correlations[name]=dict(pearson=correlation(xs,ys),spearman=correlation(rank(xs),rank(ys)),n=len(valid))
    out=[]
    for (feature,level),c in sorted(features.items()):
        n=c['loaded']+c['not_loaded'];out.append(dict(feature=feature,level=level,n=n,**c,load_rate=c['loaded']/n,
            within_rotation_expected_rate=c['expected_loaded']/n,within_rotation_lift=(c['loaded']-c['expected_loaded'])/n))
    return dict(rotations=rows,correlations=correlations,low_alpha_feature_comparison=out,
                feature_comparison_excludes_other_window_exits=excluded_other_exits,
                age_hours={k:percentiles(v) for k,v in ages.items()},
                notes=['Queues are public live residents immediately before events at load_start; only complete windows before Jan13 are included.',
                       'Feature-table not_loaded means STILL RESIDENT at window end; competing non-LOAD exits are excluded from that comparison and counted separately. Rotation alpha still uses the complete start queue.',
                       'Within-rotation feature lift subtracts LOAD/(LOAD+still resident) in that rotation before pooling, reducing vessel/rotation composition confounding.',
                       'Initial age is time since the initial snapshot, not unknown true dwell. Alpha uses all vessel residents, unlike earlier active-cohort-specific alpha.'])

def exit_analysis(a):
    dwell={};hazards=[]
    for origin in ('DISCHARGE','TRUCK_RECV'):
        records=[r for r in a.records.values() if r['origin']==origin]
        dwell[origin]=percentiles([(r['end']-r['entry'])/3600 for r in records if r['exit']=='TRUCK_DLVR'])
        for j,(lo,hi) in enumerate(zip(AGE_LIMITS,AGE_LIMITS[1:])):
            exposure=truck=loads=at_risk=0
            for r in records:
                duration=(r['end']-r['entry'])/3600
                hours=max(0,min(duration,hi)-lo);exposure+=hours;at_risk+=hours>0
                if lo<=duration<hi:
                    truck+=r['exit']=='TRUCK_DLVR';loads+=r['exit']=='LOAD'
            hazards.append(dict(origin=origin,age=AGE_NAMES[j],container_hours=exposure,containers_exposed=at_risk,
                                truck_exits=truck,competing_LOAD_exits=loads,truck_hazard_per_hour=truck/exposure if exposure else None))
    features=defaultdict(Counter)
    for r in a.records.values():
        if r['origin']!='initial':continue
        vid=r.get('vessel_id','');rotations=a.by_vessel.get(vid,())
        first=next((ro for ro in rotations if ro['end']>=START),None)
        raw=parse_time(r.get('departure_time',''));p=r['position']
        height=max(c['position']['tier'] for c in a.initial['containers'] if c['position']['block']==p['block'] and c['position']['bay']==p['bay'] and c['position']['row']==p['row'])
        fs=dict(raw_etd_minus_t0=lead_bucket((raw-START)/3600),has_rotations=str(bool(rotations)),
                raw_etd_vs_initial_S3=relation(raw,first['etd']) if first else 'unscheduled',
                initial_tier=str(p['tier']),initial_height=str(height),weight=r.get('weight_class','MEDIUM'))
        outcome=r['exit'] or 'unretrieved_Day12'
        for f,v in fs.items():features[(f,v)][outcome]+=1
    rows=[]
    for (f,v),counts in sorted(features.items()):
        n=sum(counts.values());rows.append(dict(feature=f,level=v,n=n,LOAD=counts['LOAD'],TRUCK_DLVR=counts['TRUCK_DLVR'],
                                             unretrieved_Day12=counts['unretrieved_Day12'],
                                             LOAD_rate=counts['LOAD']/n,TRUCK_DLVR_rate=counts['TRUCK_DLVR']/n))
    return dict(truck_exit_dwell_hours=dwell,cause_specific_age_hazards=hazards,initial_feature_outcomes=rows,
                notes=['Dwell quantiles condition on observed truck exits by Day12 and are censoring-biased descriptions.',
                       'Hazard denominator includes all origin arrivals while resident, ending at any exit or Jan13 cutoff; numerator only TRUCK_DLVR. No future exit-mode labels enter risk sets.',
                       'Initial outcomes are known through Day12 only; unretrieved is right-censoring, not permanent storage.'])

def mechanics(a):
    repeated=[]
    for p in ('Days1-12','Days13-20','all_train'):
        periods=('Days1-12','Days13-20') if p=='all_train' else (p,)
        total=sum(a.repeated[(x,'all')] for x in periods)
        for k in KS:
            n=sum(a.repeated[(x,k)] for x in periods)
            repeated.append(dict(period=p,k=k,repeated_moves_from_prefix=n,total_repeated_moves=total,
                                 fraction=n/total if total else None,total_relocation_landings_in_prefix=sum(a.landing_prefix[(x,k)] for x in periods)))
    hist=[dict(period=p,blockers=n,retrievals=count) for (p,n),count in sorted(a.simultaneous.items())]
    empty=[]
    for (p,b),counts in sorted(a.empty_retrieval.items()):
        own=a.empty_own[(p,b)]
        empty.append(dict(period=p,block=b,all_retrieval_events_zero_empty=counts['zero'],all_retrieval_events_positive_empty=counts['positive'],
                          own_block_found_retrievals_zero_empty=own['zero'],own_block_found_retrievals_positive_empty=own['positive']))
    return dict(repeated_blocker_prefix=repeated,blockers_per_found_retrieval=hist,block_empty_stack_frequency=empty,
                notes=['Prefix is fixed lexicographic (bay,row) rank within each block, not a dynamic height-sorted rank.',
                       'Each repeated move is attributed to that blocker last relocation landing; k prefixes are nested and must not be added.',
                       'All-event empty frequencies include absent retrievals; own-block frequencies include only found targets. Whole-train mechanics are descriptive, not fitted.'])

def cutoff_analysis(a):
    rows=[]
    for ro in a.rotations:
        key=a.key(ro);exposed=[(cid,data) for (r,cid),data in a.passed.items() if r==key]
        later=sum(a.records[cid]['exit']=='LOAD' and data['first_event']<a.records[cid]['exit_order'] and a.records[cid]['end']<=ro['end'] for cid,data in exposed)
        rows.append(dict(vessel_id=ro['vessel'],rotation_number=ro['number'],scanned_current_vessel_resident_exposures=a.scan_exposures[key],
                         passed_resident_exposures=a.passed_exposures[key],placements_with_passed_residents=a.passed_placements[key],
                         unique_passed_container_rotation_pairs=len(exposed),later_LOAD_in_same_window=later))
    return dict(rows=rows,unique_placements_scanning_loading_vessel=len(a.unique_scan_placements),
                unique_placements_with_passed_residents=len(a.unique_passed_placements),
                load_category_reversals=sum(a.load_reversals.values()),
                total_resident_exposures=sum(a.scan_exposures.values()),passed_resident_exposures=sum(a.passed_exposures.values()),
                unique_passed_container_rotation_pairs=len(a.passed),later_LOAD_in_same_window=sum(r['later_LOAD_in_same_window'] for r in rows),
                notes=['Fit-period only, complete loading windows. Residents in full stacks are excluded, matching candidate scan eligibility.',
                       'Passed is STRICT category < maximum (port, HEAVY/MEDIUM/LIGHT rank) in an already observed LOAD event in the same window, including absent LOADs. Equal categories are not passed.',
                       'Passed exposure counts can repeat a resident at many arrivals. Outcome confirmation uses fit events only.'])

def empty_analysis(a):
    rows=[]
    for cohort,n in sorted(a.all_placements.items()):
        c=a.empty_placements[cohort];rows.append(dict(cohort=cohort,all_placements=n,empty_stack_placements=sum(c.values()),
           empty_fraction=sum(c.values())/n,by_jump_hours={k:c[k] for k in ('<24h','24-72h','>72h','inf')}))
    return dict(rows=rows,total_placements=sum(a.all_placements.values()),total_empty_stack_placements=sum(sum(c.values()) for c in a.empty_placements.values()),
                note='Whole-train placement geometry only. Time-to-jump comes from the actual incoming online descriptor; missing/missed/expired-no-rotation jumps are infinity.')

def format_tables(result):
    lines=[]
    def table(title,headers,rows):
        lines.extend(['',title,' | '.join(headers),' | '.join('---' for _ in headers)])
        for row in rows:lines.append(' | '.join(str(v) for v in row))
    q=result['Q1']
    table('Q1: completed rotations, fit period',['vessel','rotation','window h','queue','loaded from queue','alpha','all LOAD events','absent LOADs'],
          [[r['vessel_id'],r['rotation_number'],f"{r['load_window_hours']:.3f}",r['yard_queue_at_load_start'],r['actual_loads_from_queue'],f"{r['empirical_alpha']:.4f}",r['total_load_events_in_window'],r['absent_load_events_in_window']] for r in q['rotations']])
    table('Q1 correlations (13 rotations)',['feature','Pearson','Spearman'],[[k,f"{v['pearson']:.4f}" if v['pearson'] is not None else 'NA',f"{v['spearman']:.4f}" if v['spearman'] is not None else 'NA'] for k,v in q['correlations'].items()])
    table('Q1: within low-alpha rotations, all requested feature levels',['feature','level','loaded','stayed in yard','loaded %','within-rotation lift pp'],
          [[r['feature'],r['level'],r['loaded'],r['not_loaded'],f"{100*r['load_rate']:.2f}",f"{100*r['within_rotation_lift']:+.2f}"] for r in q['low_alpha_feature_comparison']])
    q=result['Q2']
    table('Q2: observed truck-exit dwell hours (right-censored sampling caveat)',['origin','n','p5','p10','p25','p50','p75','p90','p95'],
          [[origin,r['n']]+[f"{r['p'+str(p)]:.2f}" for p in (5,10,25,50,75,90,95)] for origin,r in q['truck_exit_dwell_hours'].items()])
    table('Q2: cause-specific truck hazard',['origin','age','exposure h','truck exits','competing LOADs','truck exits / 1000 h'],
          [[r['origin'],r['age'],f"{r['container_hours']:.1f}",r['truck_exits'],r['competing_LOAD_exits'],f"{1000*r['truck_hazard_per_hour']:.4f}" if r['truck_hazard_per_hour'] is not None else 'NA'] for r in q['cause_specific_age_hazards']])
    table('Q2: initial t=0 feature outcomes by Day12',['feature','level','n','LOAD','TRUCK_DLVR','unretrieved','LOAD %','TRUCK %'],
          [[r['feature'],r['level'],r['n'],r['LOAD'],r['TRUCK_DLVR'],r['unretrieved_Day12'],f"{100*r['LOAD_rate']:.2f}",f"{100*r['TRUCK_DLVR_rate']:.2f}"] for r in q['initial_feature_outcomes']])
    q=result['Q3']
    table('Q3: repeated blockers, last relocation landing in fixed prefix',['period','k','repeated moves from prefix','all repeated moves','%','all relocations landing in prefix'],
          [[r['period'],r['k'],r['repeated_moves_from_prefix'],r['total_repeated_moves'],f"{100*r['fraction']:.2f}",r['total_relocation_landings_in_prefix']] for r in q['repeated_blocker_prefix']])
    table('Q3: simultaneous blockers',['blockers','Days1-12 retrievals','Days13-20 retrievals','total found retrievals'],
          [[n]+[sum(r['retrievals'] for r in q['blockers_per_found_retrieval'] if r['blockers']==n and r['period']==p) for p in ('Days1-12','Days13-20')]+[sum(r['retrievals'] for r in q['blockers_per_found_retrieval'] if r['blockers']==n)] for n in range(5)])
    table('Q3: block emptiness at retrieval time (all events vs own-block found targets)',['period','block','all: zero empty','all: >=1 empty','own found: zero empty','own found: >=1 empty'],
          [[r['period'],r['block'],r['all_retrieval_events_zero_empty'],r['all_retrieval_events_positive_empty'],r['own_block_found_retrievals_zero_empty'],r['own_block_found_retrievals_positive_empty']] for r in q['block_empty_stack_frequency']])
    q=result['Q4']
    table('Q4: loading-progress exposure',['vessel','rotation','resident scans','passed scans','placements with passed','unique passed pairs','later LOADs'],
          [[r['vessel_id'],r['rotation_number'],r['scanned_current_vessel_resident_exposures'],r['passed_resident_exposures'],r['placements_with_passed_residents'],r['unique_passed_container_rotation_pairs'],r['later_LOAD_in_same_window']] for r in q['rows']])
    table('Q4: global distinct counts',['measurement','count'],[[k,q[k]] for k in ('unique_placements_scanning_loading_vessel','unique_placements_with_passed_residents','total_resident_exposures','passed_resident_exposures','unique_passed_container_rotation_pairs','later_LOAD_in_same_window','load_category_reversals')])
    q=result['Q5']
    table('Q5: empty-stack consumption, whole-train geometry',['cohort','all placements','empty placements','empty %','<24h','24-72h','>72h','inf'],
          [[r['cohort'],r['all_placements'],r['empty_stack_placements'],f"{100*r['empty_fraction']:.3f}"]+[r['by_jump_hours'][k] for k in ('<24h','24-72h','>72h','inf')] for r in q['rows']])
    for name in ('Q1','Q2','Q3','Q4','Q5'):
        lines.append('\n'+name+' NOTES')
        notes=result[name].get('notes',[result[name].get('note','')])
        lines.extend(notes)
    return '\n'.join(lines)+'\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_paths=('solution/my_strategy.py','solution/survival_params.py')
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    expected=json.loads((ROOT/'results/phase5b/phase5b_manifest.json').read_text())
    assert all(h==expected['files'][p] for p,h in hashes.items())
    layout=json.loads((ROOT/'data/yard_layout.json').read_text())
    initial=json.loads((ROOT/'data/train/initial_state.json').read_text())
    schedule=json.loads((ROOT/'data/vessel_schedule.json').read_text())
    events=read_events(str(ROOT/'data/train/events.jsonl'))
    yard=YardState(layout);yard.load_initial_state(initial)
    a=Audit(layout,initial,schedule,yard);strategy=ObservedStrategy(a);strategy.initialize(layout,initial)
    started=time.perf_counter();stats=ObservedSimulator(yard,strategy).run(events);elapsed=time.perf_counter()-started
    trace=hashlib.sha256(json.dumps(a.trace,separators=(',',':')).encode()).hexdigest()
    assert stats.total_reshuffles==6408 and trace==expected['train_placement_sha256']
    assert len(a.trace)==10385 and stats.errors==stats.hard_constraint_violations==stats.placement_failures==0
    assert not a.load_reversals
    assert len(a.records)==4800+sum(e.type in ARRIVALS and parse_time(e.timestamp)<CUT for e in events)
    assert sum(r["exit"] is not None for r in a.records.values())==5042
    assert sum(a.repeated.values())>=1515
    assert sum(a.repeated[(p,'all')] for p in ('Days1-12','Days13-20'))==1515
    assert sum(n*c for (p,n),c in a.simultaneous.items())==6408
    assert sum(a.all_placements.values())==10385
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    result=dict(fit_cutoff_exclusive='2025-01-13T00:00:00',test_data_accessed=False,source_hashes_before_and_after=hashes,
                frozen_replay=dict(reshuffles=stats.total_reshuffles,placements=len(a.trace),trace_sha256=trace,
                                   instrumentation_seconds=elapsed,errors=stats.errors,cache_guard_fallbacks=strategy.cache_guard_fallbacks),
                Q1=rotation_analysis(a),Q2=exit_analysis(a),Q3=mechanics(a),Q4=cutoff_analysis(a),Q5=empty_analysis(a))
    save('diagnostics.json',result)
    (OUT/'tables.txt').write_text(format_tables(result))
    print('FROZEN REPLAY',result['frozen_replay'],flush=True)
    for r in result['Q1']['rotations']:print('Q1',r,flush=True)
    print('CORRELATIONS',result['Q1']['correlations'],flush=True)
    print('Q2 DWELL',result['Q2']['truck_exit_dwell_hours'],flush=True)
    for r in result['Q2']['cause_specific_age_hazards']:print('Q2 HAZARD',r,flush=True)
    print('Q3 PREFIX',[r for r in result['Q3']['repeated_blocker_prefix'] if r['period']=='all_train'],flush=True)
    print('Q3 HISTOGRAM',result['Q3']['blockers_per_found_retrieval'],flush=True)
    print('Q4', {k:v for k,v in result['Q4'].items() if k!='rows'},flush=True)
    for r in result['Q5']['rows']:print('Q5',r,flush=True)
    print(format_tables(result),flush=True)

if __name__=='__main__':main()
