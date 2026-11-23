"""Train-only oracle and placement-objective laboratory. Never edits solution.

No oracle is deployable. The feature-screen module is separate and never sees
oracle labels as features. Existing official output is hashed, never rewritten.
"""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from math import inf,exp
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.event_reader import read_events
from src.simulator import Simulator
from src.yard_state import YardState
from src.placement_interface import PlacementStrategy
OUT=ROOT/'results/phase6c'
CUT=1736726400.0
ARRIVALS={'DISCHARGE','TRUCK_RECV'}
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).replace('\\','/')
        if '/data/test/' in p or p.startswith('data/test/'):raise RuntimeError('Phase6C forbids test access')
sys.addaudithook(guard)
source=ROOT/'solution/my_strategy.py'
snapshot=ROOT/'scripts/phase6c_engine_snapshot.py';snapshot.write_bytes(source.read_bytes())
assert snapshot.read_bytes()==source.read_bytes()
spec=importlib.util.spec_from_file_location('solution._phase6c_engine_snapshot',snapshot)
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
Base=E.MyStrategy

class Experiment(Base):
    def __init__(self,mode,value=0,low=0,labels=None):
        super().__init__();self.mode=mode;self.value=value;self.low=low;self.labels=labels
        self.gamma=1/value if mode=='C3_discount' else E.DISCOUNT_RATE
    def _survival_key(self,cid):
        g,when=super()._survival_key(cid)
        if not self.mode.startswith('O'):return g,when
        outcome,actual=self.labels.get(cid,(None,inf))
        if self.mode=='O1_unretrieved_only' and outcome is None:return '__permanent',inf
        if self.mode=='O2_exit_mode_only':
            if outcome is None:return '__permanent',inf
            if outcome=='LOAD':return '__jump',when
            return '__truck:'+g,inf
        if self.mode=='O3_exact_load_only' and outcome=='LOAD':return '__jump',actual
        if self.mode=='O4_exact_truck_only' and outcome=='TRUCK_DLVR':return '__jump',actual
        return g,when
    def _survival_values(self,key):
        g,when=key
        if g=='__permanent':return 0,0,inf
        if g=='__jump':return 1,0,max(0,(when-self.current_time)/3600)
        if g.startswith('__truck:'):return 0,E.SURVIVAL_PARAMETERS[g[8:]][1],inf
        return super()._survival_values(key)
    def _place_survival(self,event):
        if self.mode.startswith('O'):return super()._place_survival(event)
        ac,mc,tc=self._survival_values(self._survival_key(event.container_id))
        u=E.survival_block_probability(ac,mc,tc,0,0,inf,self.gamma)
        probabilities={};best=None;chosen=None
        for i,(block,bay,row,maximum) in enumerate(self._stacks):
            h=self._heights[i]
            if h>=maximum:continue
            d=0.0
            for key in self._survival_lists[i]:
                p=probabilities.get(key)
                if p is None:
                    ax,mx,tx=self._survival_values(key)
                    p=E.survival_block_probability(ax,mx,tx,ac,mc,tc,self.gamma);probabilities[key]=p
                d+=p
            score=d
            if self.mode in ('C1_upward','C2_shield_low'):
                score+=self.value*u*(maximum-1-h)
                if self.mode=='C2_shield_low':score+=self.low*(1-u)*h
            elif self.mode=='C4_quadratic':score+=self.value*d*d
            rank=(score,-h if u>.25 else h) if self.mode in ('C1_upward','C2_shield_low') else (score,h)
            if best is None or rank<best:best,chosen=rank,i
        if chosen is None:raise RuntimeError('No non-full stack')
        return self._finish_placement(chosen)

class Trace(PlacementStrategy):
    def __init__(self,base):self.base=base
    def initialize(self,layout,initial):
        self.origins={c['container_id']:'initial' for c in initial['containers']};self.trace=[];self.buckets=defaultdict(lambda:dict(found=0,reshuffles=0))
        self.base.initialize(layout,initial)
    def on_event(self,e):
        self.event=e
        if e.type in ARRIVALS:self.origins[e.container_id]=e.type
        self.base.on_event(e)
    def place_container(self,yard,e):
        p=self.base.place_container(yard,e);self.trace.append((e.event_id,p.block,p.bay,p.row,p.tier));return p
    def on_container_retrieved(self,cid,pos,n):
        for key in ('early' if E.parse_time(self.event.timestamp)<CUT else 'late','exit/'+self.event.type,'origin/'+self.origins[cid]):
            self.buckets[key]['found']+=1;self.buckets[key]['reshuffles']+=n
        self.base.on_container_retrieved(cid,pos,n)
class ObservedSimulator(Simulator):
    def __init__(self,yard,s):super().__init__(yard,s);self.moves=Counter();self.repeated=0
    def _do_retrieval(self,target,above):
        for cid in above:self.repeated+=self.moves[cid]>0;self.moves[cid]+=1
        super()._do_retrieval(target,above)

def evaluate(name,base,inputs):
    path=OUT/(name+'.json')
    if path.exists():return json.loads(path.read_text())
    layout,initial,events=inputs
    yard=YardState(layout);yard.load_initial_state(initial);s=Trace(base);s.initialize(layout,initial)
    simulator=ObservedSimulator(yard,s);start=time.perf_counter();stats=simulator.run(events);elapsed=time.perf_counter()-start
    assert stats.errors==stats.hard_constraint_violations==stats.placement_failures==base.cache_guard_fallbacks==0
    assert s.buckets['early']['found']==5042 and s.buckets['late']['found']==3832
    r=dict(name=name,total=stats.total_reshuffles,excess=stats.total_reshuffles-1325,early=s.buckets['early']['reshuffles'],late=s.buckets['late']['reshuffles'],
           early_rate=s.buckets['early']['reshuffles']/5042,late_rate=s.buckets['late']['reshuffles']/3832,
           LOAD=s.buckets['exit/LOAD']['reshuffles'],TRUCK_DLVR=s.buckets['exit/TRUCK_DLVR']['reshuffles'],
           I=s.buckets['origin/initial']['reshuffles'],D=s.buckets['origin/DISCHARGE']['reshuffles'],R=s.buckets['origin/TRUCK_RECV']['reshuffles'],
           repeated=simulator.repeated,seconds=elapsed,summary=stats.summary(),official_rate=stats.reshuffles_per_retrieval,score=stats.quantitative_score(),
           trace_sha256=hashlib.sha256(json.dumps(s.trace,separators=(',',':')).encode()).hexdigest(),placements=len(s.trace),cache_guard_fallbacks=base.cache_guard_fallbacks)
    assert r['total']==r['early']+r['late']==r['LOAD']+r['TRUCK_DLVR']==r['I']+r['D']+r['R']
    save(name+'.json',r);print(name,r['early'],r['late'],r['total'],round(elapsed,2),flush=True)
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=['solution/__init__.py','solution/my_strategy.py','solution/survival_params.py','results/results.json']
    before={p:digest(p) for p in paths}
    checkpoint=json.loads((ROOT/'results/phase5b/phase5b_manifest.json').read_text())
    assert before['solution/my_strategy.py']==checkpoint['files']['solution/my_strategy.py']
    layout=json.loads((ROOT/'data/yard_layout.json').read_text());initial=json.loads((ROOT/'data/train/initial_state.json').read_text());events=read_events(str(ROOT/'data/train/events.jsonl'))
    inputs=layout,initial,events
    control=evaluate('control',Base(),inputs)
    assert control['total']==6408 and control['trace_sha256']==checkpoint['train_placement_sha256']
    # Offline labels confined to oracle instances; absent targets are not exits.
    active={c['container_id'] for c in initial['containers']};labels={}
    for e in events:
        if e.type in ARRIVALS:active.add(e.container_id)
        elif e.container_id in active:
            active.remove(e.container_id);labels[e.container_id]=(e.type,E.parse_time(e.timestamp))
    oracles=[]
    for name in ('O1_unretrieved_only','O2_exit_mode_only','O3_exact_load_only','O4_exact_truck_only'):
        oracles.append(evaluate(name,Experiment(name,labels=labels),inputs));save('oracles.json',oracles)
    configs=[('C1_upward',v,0) for v in (0,.02,.05,.10,.20,.35)]
    configs += [('C2_shield_low',v,low) for v in (.05,.15) for low in (.02,.05)]
    configs += [('C3_discount',v,0) for v in (48,72,96)]
    configs += [('C4_quadratic',v,0) for v in (.25,.50,1)]
    assert len(configs)==16
    rows=[]
    for mode,v,low in configs:
        name=f'{mode}_{v:.2f}'+(f'_{low:.2f}' if low else '')
        r=evaluate(name,Experiment(mode,v,low),inputs);rows.append(r);save('controls.json',rows)
        if mode=='C4_quadratic':assert r['trace_sha256']==control['trace_sha256']
    after={p:digest(p) for p in paths};assert before==after
    save('lab_verification.json',dict(protected_before=before,protected_after=after,unchanged=True,test_data_accessed=False,
         copy_sha256=digest('scripts/phase6c_engine_snapshot.py'),control_trace_exact=True,oracle_runs=4,control_experiments=16,
         all_errors_violations_failures=0,quadratic_traces_identical=True,
         C2='User-confirmed 2x2 C1 upward shielding + positive low-urgency height cost; no separate purity penalty.',
         quadratic_note='For D>=0, D+lambda*D^2 is strictly increasing at these positive lambdas. Rankings and tie-breaking cannot improve.',
         oracle_note='Partial revelations can overlap and interact; their effects do not add to an exact decomposition of the H1 gap.'))
if __name__=='__main__':main()
