"""Online feature extraction and censor-safe chronological classification audit.

Only data/train and the public layout/schedule are read. Container/vessel IDs
are ledger keys, never predictors. Full-hour future batch counts are forbidden.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from math import exp,inf
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.event_reader import read_events
from solution.my_strategy import MyStrategy,parse_time
OUT=ROOT/'results/phase6c'
START=parse_time('2025-01-01T00:00:00');CUT=parse_time('2025-01-13T00:00:00');END=parse_time('2025-01-21T00:00:00')
ARRIVALS={'DISCHARGE','TRUCK_RECV'}
CATEGORICAL={'origin','weight_class','port_of_discharge','day_of_week','initial_block','cohort'}
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).replace('\\','/')
        if '/data/test/' in p or p.startswith('data/test/'):raise RuntimeError('Test access forbidden')
sys.addaudithook(guard)

def extract():
    initial=json.loads((ROOT/'data/train/initial_state.json').read_text());layout=json.loads((ROOT/'data/yard_layout.json').read_text())
    schedule=json.loads((ROOT/'data/vessel_schedule.json').read_text());events=read_events(str(ROOT/'data/train/events.jsonl'))
    vessels={v['vessel_id']:v for v in schedule['vessels']}
    native=MyStrategy();native.initialize(layout,initial)
    ncap=sum(b['bays']*b['rows']*b['tiers'] for b in layout['blocks'].values())
    active={c['container_id']:dict(c,origin='initial') for c in initial['containers']};seen=set(active);missed=set()
    vq=Counter();vpq=Counter();vpwq=Counter();batch=Counter()
    for c in active.values():
        v=c.get('vessel_id','');p=c.get('port_of_discharge','');w=c.get('weight_class','MEDIUM')
        vq[v]+=1;vpq[v,p]+=1;vpwq[v,p,w]+=1
    heights=Counter()
    for c in initial['containers']:
        p=c['position'];k=p['block'],p['bay'],p['row'];heights[k]=max(heights[k],p['tier'])
    rows=[];labels={};features=[]
    def make(c,t,origin,batch_prefix):
        v=c.get('vessel_id','');port=c.get('port_of_discharge','');weight=c.get('weight_class','MEDIUM');raw=parse_time(c.get('departure_time',''))
        rotations=vessels.get(v,{}).get('rotations',[])
        r=next((r for r in rotations if parse_time(r['load_end'])>=t),None)
        meta=native._make_meta(c.get('departure_time',''),v,port,weight)
        when=parse_time(r['load_start'])+meta.fraction*(parse_time(r['load_end'])-parse_time(r['load_start'])) if r else inf
        if c['container_id'] in missed:g='missed_load'
        elif not r:g='expired_no_rot' if raw<t else 'unscheduled_active'
        elif raw<t:g='sched_overdue'
        elif origin=='initial':g='sched_init_active'
        else:
            lead=(raw-t)/3600;bucket='<24h' if lead<24 else '24-72h' if lead<=72 else '>72h'
            g=('sched_disch_active' if origin=='DISCHARGE' else 'sched_truck_active')+'|'+bucket
        a,mu=native._survival_values((g,when))[:2]  # alpha/mu do not depend on clock.
        tau=max(0,(when-t)/3600)
        def any_exit(h):return 1-exp(-mu*h)*(1-a if tau<=h else 1)
        def truck(h):return 1-exp(-mu*h)-(a*(exp(-mu*tau)-exp(-mu*h)) if tau<=h else 0)
        baseline={'exit72':any_exit(72),'load_s3':a*exp(-mu*tau) if a and tau<inf else 0,'truck120':truck(120)}
        dt=datetime.fromtimestamp(t,timezone.utc);p=c.get('position',{})
        f=dict(origin=origin,weight_class=weight,port_of_discharge=port,
          raw_etd_minus_t=(raw-t)/3600 if raw<1e11 else 0,
          s3_load_start_minus_t=(parse_time(r['load_start'])-t)/3600 if r else 0,
          raw_etd_vs_s3_etd=(raw-parse_time(r['etd']))/3600 if r and raw<1e11 else 0,
          s3_exists=int(r is not None),raw_etd_missing=int(raw>=1e11),
          in_discharge_window=int(any(parse_time(q['discharge_start'])<=t<=parse_time(q['discharge_end']) for q in rotations)),
          time_of_day=dt.hour+dt.minute/60+dt.second/3600,day_of_week=str(dt.weekday()),
          arrival_batch_size=batch_prefix,vessel_queue_size=vq[v],vessel_port_queue_size=vpq[v,port],
          vessel_port_weight_queue_size=vpwq[v,port,weight],vessel_rotation_index=r['rotation_number'] if r else 0,
          vessel_total_rotations=len(rotations),yard_occupancy=len(active)/ncap,
          initial_tier=p.get('tier',0) if origin=='initial' else 0,
          initial_stack_height=heights[p['block'],p['bay'],p['row']] if origin=='initial' else 0,
          initial_block=p.get('block','NA') if origin=='initial' else 'NA',cohort=g)
        rows.append(dict(cid=c['container_id'],time=t,origin=origin,baseline=baseline,
                         s3_start=parse_time(r['load_start']) if r else None,s3_end=parse_time(r['load_end']) if r else None))
        features.append(f)
    for c in initial['containers']:make(c,START,'initial',0)
    for e in events:
        t=parse_time(e.timestamp)
        if e.type in ARRIVALS:
            c=vars(e);key=e.vessel_id,int(t//3600);batch[key]+=1
            make(c,t,e.type,batch[key])
            active[e.container_id]=dict(c,origin=e.type);seen.add(e.container_id)
            vq[e.vessel_id]+=1;vpq[e.vessel_id,e.port_of_discharge]+=1;vpwq[e.vessel_id,e.port_of_discharge,e.weight_class]+=1
        else:
            c=active.pop(e.container_id,None)
            if c:
                labels[e.container_id]=(e.type,t)
                v=c.get('vessel_id','');p=c.get('port_of_discharge','');w=c.get('weight_class','MEDIUM')
                vq[v]-=1;vpq[v,p]-=1;vpwq[v,p,w]-=1
            elif e.type=='LOAD' and e.container_id not in seen:missed.add(e.container_id)
    assert len(rows)==15185
    return rows,features,labels


def export_table(rows,feature_rows,outcomes):
    # Targets are outcome columns, never features; incomplete horizons are null.
    with (OUT/'arrival_initial_features.jsonl').open('w') as stream:
        for i,(r,f) in enumerate(zip(rows,feature_rows)):
            t=r['time'];boundary=CUT if t<CUT else END
            kind,actual=outcomes.get(r['cid'],(None,inf));targets={}
            for target,hours in (('exit72',72),('load_s3',0),('truck120',120)):
                eligible=(t+hours*3600<boundary) if target!='load_s3' else (r['s3_end'] is None or r['s3_end']<boundary)
                if not eligible:targets[target]=None
                elif target=='exit72':targets[target]=int(actual<=t+72*3600)
                elif target=='truck120':targets[target]=int(kind=='TRUCK_DLVR' and actual<=t+120*3600)
                else:targets[target]=int(r['s3_start'] is not None and kind=='LOAD' and r['s3_start']<=actual<=r['s3_end'])
            stream.write(json.dumps(dict(row=i,timestamp=datetime.fromtimestamp(t,timezone.utc).isoformat(),
                split='fit' if t<CUT else 'validation',features=f,frozen_predictions=r['baseline'],targets=targets))+'\n')


def auc(y,p):
    y=np.asarray(y);p=np.asarray(p);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    order=np.argsort(p,kind='mergesort');sortedp=p[order];ys=y[order]
    starts=np.r_[0,np.flatnonzero(sortedp[1:]!=sortedp[:-1])+1];ends=np.r_[starts[1:],len(y)]
    sums=np.add.reduceat(ys,starts)
    rank_sum=np.sum(sums*(starts+ends+1)/2)
    return float((rank_sum-n1*(n1+1)/2)/(n1*n0))
def metrics(y,p):
    p=np.asarray(p,float);return dict(AUC=auc(y,p),Brier=float(np.mean((p-y)**2)),n=len(y),positives=int(np.sum(y)))
def univariate(train,y,valid,categorical):
    prior=float(y.mean())
    if categorical:
        levels=sorted(set(train));lookup={v:i for i,v in enumerate(levels)}
        a=np.array([lookup[v] for v in train]);b=np.array([lookup.get(v,-1) for v in valid]);k=len(levels)
    else:
        train=np.asarray(train,float);valid=np.asarray(valid,float)
        edges=np.unique(np.quantile(train,np.linspace(0,1,11)[1:-1]))
        a=np.searchsorted(edges,train,side='right');b=np.searchsorted(edges,valid,side='right');k=len(edges)+1
    n=np.bincount(a,minlength=k);yes=np.bincount(a,weights=y,minlength=k)
    pred=(yes+20*prior)/(n+20)
    pfit=pred[a];pval=np.array([pred[v] if v>=0 else prior for v in b])
    joint=np.vstack([n-yes,yes])/len(y);marg=n/len(y);py=np.array([1-prior,prior]);mi=0
    for j in range(2):
        good=joint[j]>0
        if py[j]>0:mi+=float(np.sum(joint[j,good]*np.log2(joint[j,good]/(marg[good]*py[j]))))
    return pfit,pval,mi

def design(fit,valid,features,indices_train,indices_val,baseline_train,baseline_val,baseline_only=False):
    traincols=[];valcols=[];names=[]
    if not baseline_only:
        for name in features:
            a=[fit[i][name] for i in indices_train];b=[fit[i][name] for i in indices_val]
            if name in CATEGORICAL:
                for value in sorted(set(a)):
                    traincols.append(np.array([x==value for x in a],float));valcols.append(np.array([x==value for x in b],float));names.append(name+'='+value)
            else:
                a=np.asarray(a,float);b=np.asarray(b,float);mean=float(a.mean());sd=float(a.std()) or 1
                traincols.append((a-mean)/sd);valcols.append((b-mean)/sd);names.append(name)
        # Circular encodings are deterministic transforms of admissible clock time.
        if 'time_of_day' in features:
            for trig in (np.sin,np.cos):
                traincols.append(trig(np.array([fit[i]['time_of_day'] for i in indices_train])*np.pi/12))
                valcols.append(trig(np.array([fit[i]['time_of_day'] for i in indices_val])*np.pi/12));names.append('clock_'+trig.__name__)
    for a,b,name in ((baseline_train,baseline_val,'baseline_logit'),):
        a=np.clip(a,1e-6,1-1e-6);b=np.clip(b,1e-6,1-1e-6)
        a=np.log(a/(1-a));b=np.log(b/(1-b));mean=float(a.mean());sd=float(a.std()) or 1
        traincols.append((a-mean)/sd);valcols.append((b-mean)/sd);names.append(name)
    return np.column_stack(traincols),np.column_stack(valcols),names

def logistic(x,y,z):
    x=np.column_stack([np.ones(len(x)),x]);z=np.column_stack([np.ones(len(z)),z]);w=np.zeros(x.shape[1]);w[0]=np.log((y.sum()+.5)/(len(y)-y.sum()+.5))
    penalty=np.ones(len(w));penalty[0]=0;ridge=1/len(y)
    def objective(v):
        logits=x@v
        return float(np.mean(np.logaddexp(0,logits)-y*logits)+.5*ridge*np.sum(penalty*v*v))
    for iteration in range(35):
        pred=1/(1+np.exp(-np.clip(x@w,-35,35)))
        grad=x.T@(pred-y)/len(y)+ridge*penalty*w
        hessian=(x.T*(pred*(1-pred)))@x/len(y)+np.diag(ridge*penalty+1e-8)
        step=np.linalg.solve(hessian,grad)
        scale=1.0;old=objective(w)
        while scale>1e-7 and objective(w-scale*step)>old:scale*=.5
        w-=scale*step
        if np.max(np.abs(scale*step))<1e-6:break
    return 1/(1+np.exp(-np.clip(z@w,-35,35))),dict(iterations=iteration+1,coefficients=len(w),ridge_sum_loss_penalty=1.0)

def tree(x,y,z,maxdepth=3,minleaf=100):
    prior=float(y.mean());nodes=0
    def fit(ids,depth):
        nonlocal nodes
        nodes+=1;n=len(ids);positives=float(y[ids].sum());prediction=(positives+20*prior)/(n+20)
        if depth==maxdepth or n<2*minleaf:return ('leaf',prediction)
        best=None;gain=0;parent=positives*(1-positives/n)
        for j in range(x.shape[1]):
            values=x[ids,j]
            if np.min(values)==np.max(values):continue
            for threshold in np.unique(np.quantile(values,np.linspace(.1,.9,9))):
                mask=values<=threshold;nl=int(mask.sum());nr=n-nl
                if min(nl,nr)<minleaf:continue
                yl=float(y[ids[mask]].sum());yr=positives-yl
                candidate=parent-yl*(1-yl/nl)-yr*(1-yr/nr)
                if candidate>gain:gain=candidate;best=(j,float(threshold),ids[mask],ids[~mask])
        if best is None:return ('leaf',prediction)
        j,threshold,left,right=best
        return (j,threshold,fit(left,depth+1),fit(right,depth+1))
    root=fit(np.arange(len(y)),0)
    def predict(row,node):
        if node[0]=='leaf':return node[1]
        return predict(row,node[2] if row[node[0]]<=node[1] else node[3])
    return np.array([predict(row,root) for row in z]),dict(maxdepth=maxdepth,minleaf=minleaf,nodes=nodes)

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    rows,feature_rows,outcomes=extract();features=list(feature_rows[0])
    records=[];full=[];screens=[];supports=[];incremental=[]
    for target in ('exit72','load_s3','truck120'):
        train=[];valid=[];labels=[]
        for i,r in enumerate(rows):
            t=r['time'];boundary=CUT if t<CUT else END
            horizon=72 if target=='exit72' else 120
            eligible=(t+horizon*3600<boundary) if target!='load_s3' else (r['s3_end'] is None or r['s3_end']<boundary)
            if not eligible:continue
            kind,actual=outcomes.get(r['cid'],(None,inf))
            if target=='exit72':label=int(actual<=t+72*3600)
            elif target=='truck120':label=int(kind=='TRUCK_DLVR' and actual<=t+120*3600)
            else:label=int(r['s3_start'] is not None and kind=='LOAD' and r['s3_start']<=actual<=r['s3_end'])
            (train if t<CUT else valid).append(i);labels.append((i,label))
        labelmap=dict(labels);yt=np.array([labelmap[i] for i in train],float);yv=np.array([labelmap[i] for i in valid],float)
        pt=np.array([rows[i]['baseline'][target] for i in train]);pv=np.array([rows[i]['baseline'][target] for i in valid])
        base=metrics(yv,pv)
        supports.append(dict(target=target,fit_rows=len(train),fit_positives=int(yt.sum()),validation_rows=len(valid),validation_positives=int(yv.sum()),
                             fit_initial_rows=sum(rows[i]['origin']=='initial' for i in train),validation_initial_rows=sum(rows[i]['origin']=='initial' for i in valid)))
        full.append(dict(target=target,model='frozen_11_cohort_survival',**base))
        for name in features:
            a=[feature_rows[i][name] for i in train];b=[feature_rows[i][name] for i in valid]
            pfit,pvalid,mi=univariate(a,yt,b,name in CATEGORICAL)
            m=metrics(yv,pvalid)
            screens.append(dict(target=target,feature=name,fit_mutual_information_bits=mi,fit_AUC=auc(yt,pfit),**m,AUC_lift_vs_frozen=m['AUC']-base['AUC']))
            xi,zi,_=design(feature_rows,feature_rows,[name],train,valid,pt,pv)
            pi,_=logistic(xi,yt,zi);mi_metrics=metrics(yv,pi)
            incremental.append(dict(target=target,feature=name,**mi_metrics,AUC_lift_vs_frozen=mi_metrics['AUC']-base['AUC']))
        x,z,names=design(feature_rows,feature_rows,features,train,valid,pt,pv,True)
        p,meta=logistic(x,yt,z);full.append(dict(target=target,model='calibrated_baseline_logistic',**metrics(yv,p),configuration=meta))
        x,z,names=design(feature_rows,feature_rows,features,train,valid,pt,pv)
        p,meta=logistic(x,yt,z);full.append(dict(target=target,model='all_features_plus_baseline_logistic',**metrics(yv,p),configuration=meta))
        p,meta=tree(x,yt,z);full.append(dict(target=target,model='all_features_plus_baseline_depth3_tree',**metrics(yv,p),configuration=meta))
        best=max((r for r in screens if r['target']==target),key=lambda r:r['AUC'])
        print('FEATURE TARGET',target,'BASE',base,'BEST',best,flush=True)
    export_table(rows,feature_rows,outcomes)
    result=dict(fit_cutoff_exclusive='2025-01-13T00:00:00',observation_end_exclusive='2025-01-21T00:00:00',
       supports=supports,model_comparison=full,univariate=screens,baseline_plus_one_feature=incremental,
       single_feature_lift_over_003=[r for r in screens if r['AUC_lift_vs_frozen']>.03],
       add_one_feature_lift_over_003=[r for r in incremental if r['AUC_lift_vs_frozen']>.03],seconds=time.perf_counter()-started,
       package_policy='NumPy already installed; no packages installed.',
       notes=['Only complete 72h/120h observation horizons within each chronological split are used; even observed positives from incomplete horizons are excluded to avoid outcome-dependent sampling.',
              'S3 target requires the assigned loading window to finish within its split. No-S3 rows are structurally known negatives.',
              'Initial rows occur only at Day1; validation contains new arrivals only. Initial-specific features have no direct late validation support.',
              'Arrival batch size is the observed prefix in the current hour/vessel, including the current arrival, never the completed future hourly batch.',
              'Queue and occupancy features are observed before arrival. Initial queues use the full initial snapshot.',
              'Baseline horizon probabilities use frozen alpha/mu and its one S3 jump; background risk approximates truck risk because fit off-S3 LOADs were zero.',
              'All model settings fixed before validation: logistic L2=1 in summed loss; depth3 tree minleaf100; univariate fit-decile/category probabilities shrink by 20 prior observations.',
              'Mutual information is empirical fit-only discretized MI. Feature screens and three targets create multiple comparisons; AUC lifts are diagnostics, not discoveries or a Bayes bound.',
              'Container and vessel ID strings/numbers are excluded as predictors. Keys are used only for online inventory/label linkage.',
              'No estimator output is used in a placement experiment or deployed to solution.'])
    save('feature_screen.json',result)
    print('MODEL COMPARISON',full,flush=True)
if __name__=='__main__':main()
