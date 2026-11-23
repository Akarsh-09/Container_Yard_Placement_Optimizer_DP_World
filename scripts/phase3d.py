"""Bounded 14-run train-only horizon/discount/multi-rotation comparison."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import phase3
from phase3b import PeriodExitObserver, display_config
from phase3b_diag import load_train_inputs
from solution.my_strategy import MyStrategy
OUT=ROOT/'results/phase3d'
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def early(r): return r['slices']['period/1-12']['reshuffles']
def late(r): return r['slices']['period/13-20']['reshuffles']
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    phase3.OUT=OUT
    inputs=load_train_inputs()
    configs={}; runs=[]
    def run(name,config):
        configs[name]=config
        r,_=phase3.evaluate(name,MyStrategy(**config),inputs,display_config(config),observer_class=PeriodExitObserver)
        runs.append(r); save('runs.json',runs)
        return r
    historical=[]
    for r in json.loads((ROOT/'results/phase3c/core.json').read_text()):
        if r['config']['survival_rotation_sigma_hours']!=0: continue
        r=dict(r,name='historical_'+r['name'])
        h=r['config']['survival_horizon_hours']
        config=dict(r['config'],survival_horizon_hours=float('inf') if h=='infinity' else h)
        configs[r['name']]=config
        historical.append(r)
    for h in (12,24,48,72,96):
        run(f'A_H{h}',dict(mode='survival_risk',survival_horizon_hours=h,survival_rotation_sigma_hours=0))
    for d in (24,48,96,144,240):
        run(f'B_D{d}',dict(mode='survival_risk',survival_horizon_hours=float('inf'),survival_discount_hours=d,survival_rotation_sigma_hours=0))
    hard=min(historical+[r for r in runs if r['name'].startswith('A_')],key=early)
    discount=min([r for r in runs if r['name'].startswith('B_')],key=early)
    for label,parent in (('hard',hard),('discount',discount)):
        r=run('C_multi_'+label,dict(configs[parent['name']],use_multi_rotation=True))
        r['parent']=parent['name']
        r['eligible']=early(r)<early(parent) and late(r)<=late(parent)
    cores=historical+[r for r in runs if not r['name'].startswith('C_') or r['eligible']]
    top=sorted(cores,key=early)[:2]
    expanded=[]
    for i,parent in enumerate(top):
        r=run(f'D_expanded_{i}',dict(configs[parent['name']],use_expanded_survival=True))
        r['parent']=parent['name']
        r['eligible']=early(parent)-early(r)>=25 and late(r)<late(parent)
        expanded.append(r)
    save('runs.json',runs)
    best_core=min(cores,key=early)
    eligible=[best_core]+[r for r in expanded if r['eligible']]
    winner=min(eligible,key=early)
    config=configs[winner['name']]
    save('selection.json',{'winner':winner,'config':display_config(config),'top2_core':[r['name'] for r in top],
                           'best_hard':hard['name'],'best_discount':discount['name'],
                           'default_unchanged':True,'selection':'Early count only; multi must improve early and not regress late; expanded must save >=25 early and improve late.'})
    cached,a=phase3.evaluate('winning_cached',MyStrategy(**config),inputs,display_config(config),observer_class=PeriodExitObserver)
    live,b=phase3.evaluate('winning_live',MyStrategy(**dict(config,use_cache=False)),inputs,display_config(dict(config,use_cache=False)),observer_class=PeriodExitObserver)
    assert a==b and cached['trace_sha256']==winner['trace_sha256']
    save('cache_equivalence.json',{'identical_placements':len(a),'cached_seconds':cached['wall_sec'],'live_seconds':live['wall_sec'],
                                 'trace_sha256':cached['trace_sha256'],'winner':winner['name']})
if __name__=='__main__': main()
