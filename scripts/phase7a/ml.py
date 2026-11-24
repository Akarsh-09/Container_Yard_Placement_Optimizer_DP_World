"""NumPy-only, predeclared dynamic pairwise screen and conditional plug-in."""
from common import *
from policies import parts
from math import inf,log,isfinite
from bisect import bisect_right
import numpy as np
FEATURES=['p_frozen_240','logit_p_frozen_240','p_bg_240','p_jump_240','p_tot_48','is_initial_x','is_overdue_x','is_missed_load_x','is_expired_x','resident_age_hours_x','overdue_hours_x','tau_x_capped','has_finite_tau_x','vessel_total_rotations_x','vessel_queue_ratio_x','raw_etd_minus_s3_hours_x','is_truck_recv_c','is_missed_load_c','lead_time_hours_c','tau_c_capped','has_finite_tau_c','vessel_total_rotations_c','vessel_queue_ratio_c','tau_diff_clipped','tau_x_lt_tau_c','tau_x_eq_tau_c','same_vessel','same_rotation','same_port','port_weight_diff','age_x_minus_lead_c']
def dynamic(base,cid,info,queues):
    meta=base._meta[cid];key=base._survival_key(cid);v=base._survival_values(key);rot=base._next_by_vessel.get(meta.vessel)
    age=(base.current_time-base._arrival_times.get(cid,START))/3600
    lead=(meta.departure-base._arrival_times.get(cid,base.current_time))/3600
    window=(rot.end-rot.start)/3600 if rot else 0
    # raw ETD minus S3 expected per-port/weight loading time (not rotation end).
    raw_s3=(meta.departure-key[1])/3600 if isfinite(key[1]) and meta.departure<1e11 else 0.
    return dict(cid=cid,v=v,key=key,meta=meta,age=age,lead=lead,initial=cid in base._initial_ids,overdue=meta.departure<base.current_time,missed=key[0]=='missed_load',expired=key[0]=='expired_no_rot',overdue_hours=max(0,(base.current_time-meta.departure)/3600),nrot=len(base._rotations.get(meta.vessel,())),queue=queues.get(meta.vessel,0)/max(window,1.),raw_s3=raw_s3,truck=base._origins.get(cid)=='TRUCK_RECV',port=info.port_of_discharge,weight=E.WEIGHT_RANK.get(info.weight_class,1),rotation=(meta.vessel,rot.end) if rot else None)
def features(x,c):
    p=E.survival_block_probability(*x['v'],*c['v']);j,bg=parts(x['v'],c['v'],1/240);p48=E.survival_block_probability(*x['v'],*c['v'],1/48);clip=min(1-1e-4,max(1e-4,p));tx=x['v'][2];tc=c['v'][2]
    diff=0. if tx==tc else 480. if not isfinite(tx) else -480. if not isfinite(tc) else max(-480,min(480,tx-tc))
    same=x['meta'].vessel==c['meta'].vessel
    return [p,log(clip/(1-clip)),bg,j,p48,x['initial'],x['overdue'],x['missed'],x['expired'],x['age'],x['overdue_hours'],min(tx,480),isfinite(tx),x['nrot'],x['queue'],x['raw_s3'],c['truck'],c['missed'],c['lead'],min(tc,480),isfinite(tc),c['nrot'],c['queue'],diff,tx<tc,tx==tc,same,x['rotation'] is not None and x['rotation']==c['rotation'],x['port']==c['port'],x['meta'].fraction-c['meta'].fraction if same else 0.,x['age']-c['lead']]
def queues_for(base,cids):return Counter(base._meta[cid].vessel for cid in cids if cid not in base.observed_missed_loads)
class Sample(E.MyStrategy):
    def __init__(self,counts,rng):super().__init__();self.counts=counts;self.rng=rng;self.rows=[];self.links=[]
    def _place_survival(self,event):
        n=self.counts.get(event.event_id,0)
        if n:
            cids=list(self._cid_index);q=queues_for(self,cids);incoming=dynamic(self,event.container_id,event,q)
            chosen=self.rng.integers(0,len(cids),size=n);cache={}
            for idx in chosen:
                cid=cids[idx]
                if cid not in cache:cache[cid]=dynamic(self,cid,self.yard.get_container_info(cid),q)
                self.rows.append(features(cache[cid],incoming));self.links.append((self.current_time,cid,event.container_id))
        return super()._place_survival(event)
    def place_container(self,yard,event):self.yard=yard;return super().place_container(yard,event)
class LabelSimulator(ObservedSimulator):
    def __init__(self,yard,s):super().__init__(yard,s);self.exits=defaultdict(list)
    def _do_retrieval(self,cid,above):
        super()._do_retrieval(cid,above)
        if self.yard.get_container_position(cid) is None:self.exits[cid].append(E.parse_time(self.strategy.event.timestamp))
def dataset(inp):
    layout,initial,events=inp;rng=np.random.default_rng(707);counts=Counter()
    for lo,hi,n in ((START,E.parse_time('2025-01-09T00:00:00'),150000),(CUT,E.parse_time('2025-01-17T00:00:00'),80000)):
        eligible=[e.event_id for e in events if e.type in ARRIVALS and lo<=E.parse_time(e.timestamp)<=hi]
        counts.update(np.asarray(eligible)[rng.integers(0,len(eligible),n)].tolist())
    base=Sample(counts,rng);s=Trace(base);yard=YardState(layout);yard.load_initial_state(initial);s.initialize(layout,initial)
    sim=LabelSimulator(yard,s);stats=sim.run(events)
    assert stats.total_reshuffles==6408 and hashlib.sha256(json.dumps(s.trace,separators=(',',':')).encode()).hexdigest()==TRACE
    X=np.array(base.rows,float);y=[];ts=[]
    split_exits={True:{cid:[v for v in times if v<CUT] for cid,times in sim.exits.items()},False:{cid:[v for v in times if CUT<=v<E.parse_time('2025-01-21T00:00:00')] for cid,times in sim.exits.items()}}
    for t,x,c in base.links:
        boundary=CUT if t<CUT else E.parse_time('2025-01-21T00:00:00')
        # No fit-label lookup includes an outcome at or beyond CUT.
        horizon=min(t+96*3600,boundary)
        def observed(cid):
            times=split_exits[t<CUT].get(cid,[]);k=bisect_right(times,t)
            return times[k] if k<len(times) and times[k]<boundary and times[k]<=horizon else inf
        tx,tc=observed(x),observed(c);y.append(int(tx<tc and tx<=horizon));ts.append(t)
    y=np.array(y,float);ts=np.array(ts);fit=ts<CUT
    assert fit.sum()==150000 and (~fit).sum()==80000 and np.isfinite(X).all()
    np.savez_compressed(OUT/'pair_dataset.npz',X=X,y=y,time=ts,fit=fit)
    save('dataset_manifest.json',dict(seed=707,features=FEATURES,fit_pairs=int(fit.sum()),validation_pairs=int((~fit).sum()),fit_initial=int(X[fit,5].sum()),validation_initial=int(X[~fit,5].sum()),sampling='Uniform eligible arrival event, then uniform real resident; sampling with replacement; no IDs used as predictors',target='Y96: resident actual exit precedes incoming actual exit within 96 hours',fit_outcomes_exclusive='2025-01-13T00:00:00',validation_outcomes_exclusive='2025-01-21T00:00:00',queue_definition='Current live same-vessel residents excluding historically missed LOAD IDs; divide by full next loading-window hours, minimum denominator 1',rotation_definition='Equality of public vessel and next rotation end; both must have an assigned rotation',port_weight_diff='Resident minus incoming normalized port/weight loading-order fraction, same vessel only',raw_etd_minus_s3='Raw ETD minus fraction-adjusted expected S3 loading timestamp',control_trace_exact=True))
    return X,y,fit

def sigmoid(a):return 1/(1+np.exp(-np.clip(a,-35,35)))
def auc(y,p):
    order=np.argsort(p,kind='stable');ps=p[order];ys=y[order];starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1];ends=np.r_[starts[1:],len(y)];positives=y.sum()
    if positives==0 or positives==len(y):return None
    return float((np.sum(np.add.reduceat(ys,starts)*(starts+ends+1)/2)-positives*(positives+1)/2)/(positives*(len(y)-positives)))
def metric(y,p):return dict(n=len(y),positives=int(y.sum()),AUC=auc(y,p),Brier=float(np.mean((y-p)**2)))
def logistic(x,y,l2=10,intercept=True):
    a=np.column_stack([np.ones(len(x)),x]) if intercept else x
    w=np.zeros(a.shape[1]);penalty=np.full(len(w),l2);penalty[0]=0 if intercept else l2
    for iteration in range(50):
        p=sigmoid(a@w);g=a.T@(p-y)+penalty*w;h=(a.T*(p*(1-p)))@a+np.diag(penalty+1e-8);step=np.linalg.solve(h,g)
        old=float(np.sum(np.logaddexp(0,a@w)-y*(a@w))+.5*np.sum(penalty*w*w));scale=1
        while scale>1e-7:
            v=w-scale*step;z=a@v;loss=np.sum(np.logaddexp(0,z)-y*z)+.5*np.sum(penalty*v*v)
            if loss<=old:break
            scale*=.5
        w-=scale*step
        if max(abs(scale*step))<1e-7:break
    return w
class GBDT:
    def fit(self,x,y):
        self.edges=[np.unique(np.quantile(x[:,j],np.arange(1,16)/16)) for j in range(x.shape[1])]
        bins=self.bin(x);self.trees=[];self.gain=np.zeros(x.shape[1]);self.prior=float(log(y.mean()/(1-y.mean())));pred=np.full(len(y),self.prior)
        for iteration in range(30):
            p=sigmoid(pred);g=y-p;h=p*(1-p)
            def build(ids,depth):
                G=g[ids].sum();H=h[ids].sum();value=float(G/(H+1e-8))
                if depth==3 or len(ids)<400:return dict(value=value)
                best=None;gain=0.
                for j in range(x.shape[1]):
                    b=bins[ids,j];cnt=np.bincount(b,minlength=16);gs=np.bincount(b,weights=g[ids],minlength=16);hs=np.bincount(b,weights=h[ids],minlength=16)
                    ns=cnt.cumsum()[:-1];gl=gs.cumsum()[:-1];hl=hs.cumsum()[:-1];score=gl**2/(hl+1e-8)+(G-gl)**2/(H-hl+1e-8)-G*G/(H+1e-8);score[(ns<200)|(len(ids)-ns<200)]=-inf;k=int(np.argmax(score))
                    if score[k]>gain:gain=float(score[k]);best=(j,k)
                if best is None:return dict(value=value)
                j,k=best;self.gain[j]+=gain;mask=bins[ids,j]<=k
                return dict(feature=j,threshold=k,left=build(ids[mask],depth+1),right=build(ids[~mask],depth+1))
            tree=build(np.arange(len(y)),0);self.trees.append(tree);pred+=.1*self.tree_predict(bins,tree)
        return self
    def bin(self,x):return np.column_stack([np.searchsorted(e,x[:,j],side='left') for j,e in enumerate(self.edges)]).astype(np.uint8)
    def tree_predict(self,bins,tree):
        result=np.empty(len(bins))
        def visit(node,ids):
            if 'value' in node:result[ids]=node['value'];return
            mask=bins[ids,node['feature']]<=node['threshold'];visit(node['left'],ids[mask]);visit(node['right'],ids[~mask])
        visit(tree,np.arange(len(bins)));return result
    def predict(self,x):
        b=self.bin(x);return sigmoid(self.prior+sum(.1*self.tree_predict(b,t) for t in self.trees))
class Plugin(E.MyStrategy):
    def __init__(self,predict):super().__init__();self.predict=predict
    def place_container(self,yard,event):self.yard=yard;return super().place_container(yard,event)
    def _place_survival(self,event):
        cids=list(self._cid_index);q=queues_for(self,cids);c=dynamic(self,event.container_id,event,q)
        x=np.array([features(dynamic(self,cid,self.yard.get_container_info(cid),q),c) for cid in cids]);ps=self.predict(x);mapping=dict(zip(cids,ps));best=None;chosen=None
        for i,(b,bay,row,m) in enumerate(self._stacks):
            h=self._heights[i]
            if h>=m:continue
            rank=(sum(mapping[cid] for cid in self._contents[i]),h,b,bay,row)
            if best is None or rank<best:best,chosen=rank,i
        return self._finish_placement(chosen)
def main():
    before=integrity();inp=inputs()
    X,y,fit=dataset(inp);x=X[fit];z=X[~fit];yt=y[fit];yv=y[~fit]
    # All model parameters, transforms, and quantiles are fit before validation scoring.
    mean=x.mean(axis=0);sd=x.std(axis=0);sd[sd==0]=1
    w=logistic((x-mean)/sd,yt)
    # Exactly one parameter, temperature/slope only; no free intercept.
    cal=logistic(x[:,1:2],yt,l2=0,intercept=False)
    tree=GBDT().fit(x,yt)
    np.savez_compressed(OUT/'logistic_models.npz',mean=mean,sd=sd,w=w,cal=cal)
    save('gbdt_model.json',dict(edges=[a.tolist() for a in tree.edges],trees=tree.trees,prior=tree.prior,gain=tree.gain.tolist(),hyperparameters=dict(trees=30,max_depth=3,learning_rate=.1,min_leaf=200,quantile_bins=16)))
    preds={'M0_frozen_raw':z[:,0],'M0_frozen_cal':sigmoid(z[:,1]*cal[0]),'M1_pairwise_logistic':sigmoid(w[0]+((z-mean)/sd)@w[1:]),'M2_pairwise_gbdt':tree.predict(z)}
    rows=[]
    for name,p in preds.items():
        for label,mask in [('overall',np.ones(len(z),bool)),('initial',z[:,5]==1),('new',z[:,5]==0)]:rows.append(dict(model=name,slice=label,**metric(yv[mask],p[mask])))
    base=next(r for r in rows if r['model']=='M0_frozen_cal' and r['slice']=='overall')
    eligible=[r for r in rows if r['slice']=='overall' and r['model'].startswith(('M1','M2')) and r['AUC']>=base['AUC']+.015 and r['Brier']<base['Brier']]
    winner=max(eligible,key=lambda r:r['AUC'])['model'] if eligible else None
    result=dict(metrics=rows,gate_passed=bool(winner),winner=winner,gate='AUC lift >= .015 and lower Brier than one-parameter calibrated baseline; largest validation AUC wins among gate passers',calibration_slope=float(cal[0]),logistic_top_features=sorted([dict(feature=k,standardized_coefficient=float(v)) for k,v in zip(FEATURES,w[1:])],key=lambda a:abs(a['standardized_coefficient']),reverse=True),gbdt_top_features=sorted([dict(feature=k,split_gain=float(v)) for k,v in zip(FEATURES,tree.gain)],key=lambda a:a['split_gain'],reverse=True),fit_only=True)
    save('ml_screen.json',result);print(json.dumps(result),flush=True)
    if winner:
        predict=(lambda a:sigmoid(w[0]+((a-mean)/sd)@w[1:])) if winner.startswith('M1') else tree.predict
        evaluate(winner+'_plugin',Plugin(predict),inp)
    assert before==integrity();save('integrity_after_track3.json',dict(unchanged=True,test_access_blocked=True))
if __name__=='__main__':main()
