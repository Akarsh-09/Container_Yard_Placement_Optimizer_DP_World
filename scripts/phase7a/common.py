"""Phase 7A train-only instrumentation. Never imports prior experiment runners."""
import sys, json, hashlib, time, importlib.util, subprocess
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
OUT=ROOT/'results/phase7a';OUT.mkdir(exist_ok=True)
def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and isinstance(args[0],(str,bytes)):
        p=str(args[0]).replace('\\','/')
        if '/data/test' in p or p.startswith('data/test'):raise RuntimeError('Phase7A forbids test access')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('solution._phase7a_snapshot',ROOT/'scripts/phase7a/engine_snapshot.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
from src.simulator import Simulator
from src.yard_state import YardState
from src.event_reader import read_events
from src.placement_interface import PlacementStrategy
CUT=E.parse_time('2025-01-13T00:00:00');START=E.parse_time('2025-01-01T00:00:00');ARRIVALS={'DISCHARGE','TRUCK_RECV'}
TRACE='74a6945f533e5cbe55d16fecf93d09fdb0b886da151bd06bfa4fad464fb2f414'
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def inputs():return (json.loads((ROOT/'data/yard_layout.json').read_text()),json.loads((ROOT/'data/train/initial_state.json').read_text()),read_events(str(ROOT/'data/train/events.jsonl')))
def integrity():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    # Do not even stat or hash test files. Git's tracked blob index supplies their baseline.
    protected={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths if p and not p.startswith('data/test/') and (p.startswith(('solution/','src/','data/','baselines/')) or p in ('results/results.json','docs/design.md','validate_submission.sh'))}
    test_index=subprocess.check_output(['git','ls-files','-s','--','data/test'],cwd=ROOT).decode()
    return dict(files=protected,test_index=test_index)
class Trace(PlacementStrategy):
    def __init__(self,base):self.base=base
    def initialize(self,layout,initial):
        self.origins={c['container_id']:'initial' for c in initial['containers']};self.trace=[];self.buckets=defaultdict(lambda:dict(found=0,reshuffles=0));self.base.initialize(layout,initial)
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
    def __init__(self,yard,s):super().__init__(yard,s);self.moves=Counter();self.repeated=0;self.empty_retrievals=0
    def _do_retrieval(self,target,above):
        pos=self.yard.get_container_position(target);bi=self.yard.blocks[pos.block]
        self.empty_retrievals+=any(self.yard.get_stack_height(pos.block,b,r)==0 for b in range(1,bi.bays+1) for r in range(1,bi.rows+1))
        for cid in above:
            if cid:self.repeated+=self.moves[cid]>0;self.moves[cid]+=1
        super()._do_retrieval(target,above)
def evaluate(name,base,inp,early_only=False,persist=True):
    layout,initial,events=inp
    if early_only:events=[e for e in events if E.parse_time(e.timestamp)<CUT]
    yard=YardState(layout);yard.load_initial_state(initial);s=Trace(base);s.initialize(layout,initial)
    sim=ObservedSimulator(yard,s);start=time.perf_counter();stats=sim.run(events);elapsed=time.perf_counter()-start
    assert stats.errors==stats.hard_constraint_violations==stats.placement_failures==base.cache_guard_fallbacks==0,(name,stats.summary())
    assert s.buckets['early']['found']==5042
    if not early_only:assert s.buckets['late']['found']==3832
    r=dict(name=name,total=stats.total_reshuffles,excess=stats.total_reshuffles-1325,early=s.buckets['early']['reshuffles'],late=s.buckets['late']['reshuffles'],
       early_found=s.buckets['early']['found'],late_found=s.buckets['late']['found'],early_rate=s.buckets['early']['reshuffles']/5042,late_rate=s.buckets['late']['reshuffles']/3832,
       LOAD=s.buckets['exit/LOAD']['reshuffles'],TRUCK_DLVR=s.buckets['exit/TRUCK_DLVR']['reshuffles'],I=s.buckets['origin/initial']['reshuffles'],D=s.buckets['origin/DISCHARGE']['reshuffles'],R=s.buckets['origin/TRUCK_RECV']['reshuffles'],repeated=sim.repeated,
       seconds=elapsed,official_rate=stats.reshuffles_per_retrieval,summary=stats.summary(),trace_sha256=hashlib.sha256(json.dumps(s.trace,separators=(',',':')).encode()).hexdigest(),placements=len(s.trace),empty_retrievals=sim.empty_retrievals,errors=stats.errors,violations=stats.hard_constraint_violations,failures=stats.placement_failures,count_guard_fallbacks=base.cache_guard_fallbacks)
    if persist:save(name+'.json',r)
    print(name,r['early'],r['late'],r['total'],round(elapsed,2),flush=True)
    return r
