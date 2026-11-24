"""Exact requested ADP features and four compaction policies."""
from common import E
from math import exp,isfinite,inf
NAMES=['jump_240','bg_240','jump_48','bg_48','init_risk','overdue_risk','max_risk','strict_inv','compat_gap','is_empty','u_empty','h1_active','u_tier5','longstay_compact','same_vessel_top','is_reloc_target']
THETA0=[1.,1.]+[0.]*14
def parts(x,c,gamma):
    ax,mx,tx=x;ac,mc,tc=c;rate=mx+mc+gamma
    jx=ax>0 and isfinite(tx);jc=ac>0 and isfinite(tc)
    ex=exp(-rate*tx) if jx else 1.;ec=exp(-rate*tc) if jc else 1.
    bg=mx/rate*(1-(ax*ex if jx else 0)-(ac*ec if jc else 0)+(ax*ac*min(ex,ec) if jx and jc else 0)) if mx else 0.
    jump=ax*ex*(1-ac*(1. if tx>tc else .5 if tx==tc else 0.)) if jx and jc else ax*ex if jx else 0.
    return jump,bg
class Policy(E.MyStrategy):
    def __init__(self,mode='ADP',theta=None):super().__init__();self.mode=mode;self.theta=list(theta or THETA0)
    def _place_survival(self,event):
        cid=event.container_id;ck=self._survival_key(cid);c=self._survival_values(ck)
        uc=E.survival_block_probability(*c,0,0,inf)
        low=ck[0]=='missed_load' or uc<.15
        pair={};urg={};best=None;chosen=None
        lowest={}
        if self.mode=='ADP' and self.theta[15]:
            for b,ids in self._block_indices.items():
                eligible=[i for i in ids if self._heights[i]<self._stacks[i][3]]
                if eligible:lowest[b]=min(eligible,key=lambda i:self._heights[i])
        for i,(b,bay,row,maximum) in enumerate(self._stacks):
            h=self._heights[i]
            if h>=maximum:continue
            residents=self._contents[i];keys=self._survival_lists[i]
            # Compute all pair anchors once per descriptor, not once per resident.
            vals=[];taus=[];us=[]
            for key in keys:
                if key not in pair:
                    x=self._survival_values(key);j,bg=parts(x,c,1/240);j48,bg48=parts(x,c,1/48)
                    pair[key]=(j,bg,j48,bg48,E.survival_block_probability(*x,*c),x[2])
                    urg[key]=E.survival_block_probability(*x,0,0,inf)
                v=pair[key];vals.append(v);taus.append(v[5]);us.append(urg[key])
            d=sum(v[4] for v in vals)
            if self.mode=='ADP':
                w=self.theta
                # d + deviations gives exact frozen floating-point order at theta0.
                score=d
                if w[1]!=1:score+=(w[1]-1)*sum(v[1] for v in vals)
                if w[2]:score+=w[2]*sum(v[2] for v in vals)
                if w[3]:score+=w[3]*sum(v[3] for v in vals)
                if w[4]:score+=w[4]*sum(v[4] for x,v in zip(residents,vals) if x in self._initial_ids)
                if w[5]:score+=w[5]*sum(v[4] for x,v in zip(residents,vals) if self._meta[x].departure<self.current_time)
                if w[6]:score+=w[6]*max((v[4] for v in vals),default=0.)
                if w[7]:score+=w[7]*sum(t<c[2] for t in taus)
                if w[8] and taus and isfinite(c[2]) and min(taus)>c[2]:score+=w[8]*min((min(taus)-c[2])/240,1.)
                if h==0:score+=w[9]+w[10]*uc
                if h==1 and max(us,default=0)>.25:score+=w[11]
                if h==maximum-1:score+=w[12]*uc
                score+=w[13]*(1-uc)*h
                if residents and self._meta[residents[-1]].vessel==self._meta[cid].vessel:score+=w[14]
                if lowest.get(b)==i:score+=w[15]
            elif self.mode=='CD1_aggressive_compact':score=d-.12*h if low else d
            elif self.mode=='CD2_drain_and_shield':
                score=d+(.30 if h==1 and max(us,default=0)>.25 else 0)+(.40*uc if h==0 else 0)-(.10*(1-uc)*h if low else 0)
            elif self.mode=='CD3_sink_blocks':score=d+(-.20 if b in ('B09','B10') else .20) if low else d+(.20 if uc>=.25 and b in ('B09','B10') else 0)
            elif self.mode=='CD4_init_quarantine':
                score=d+.8*sum(v[4] for x,v in zip(residents,vals) if x in self._initial_ids)
                if any(x in self._initial_ids and u>.15 for x,u in zip(residents,us)):score+=.15
            else:raise ValueError(self.mode)
            rank=(score,-h if self.mode=='CD1_aggressive_compact' and low else h,b,bay,row)
            if best is None or rank<best:best,chosen=rank,i
        if chosen is None:raise RuntimeError('No non-full stack')
        return self._finish_placement(chosen)

# Vectorized stack aggregation; descriptors retain scalar frozen probability math.
import numpy as np
class VectorPolicy(Policy):
    def initialize(self,layout,initial):
        super().initialize(layout,initial)
        n=len(self._stacks);m=max(x[3] for x in self._stacks)
        self.ids=np.zeros((n,m),int);self.init=np.zeros((n,m),bool);self.depart=np.full((n,m),inf)
        self.top_vessel=np.full(n,'',object);self.code={};self.descriptors=[None];self.pending=set();self.vgen=-1;self._anchor_data=[(0.,0.,inf)]
        self.maximum=np.array([s[3] for s in self._stacks]);self.sinks=np.array([s[0] in ('B09','B10') for s in self._stacks])
    def _refresh_indices(self,yard,indices):
        indices=list(indices);super()._refresh_indices(yard,indices);self.pending.update(indices)
    def _refresh(self,yard):
        self.pending.update(self._keys_dirty)
        super()._refresh(yard)
        indices=range(len(self._stacks)) if self.vgen!=self._generation else self.pending
        for i in indices:
            self.ids[i]=0;self.init[i]=False;self.depart[i]=inf
            for j,(cid,key) in enumerate(zip(self._contents[i],self._survival_lists[i])):
                if key not in self.code:self.code[key]=len(self.descriptors);self.descriptors.append(key)
                self.ids[i,j]=self.code[key];self.init[i,j]=cid in self._initial_ids;self.depart[i,j]=self._meta[cid].departure
            self.top_vessel[i]=self._meta[self._contents[i][-1]].vessel if self._contents[i] else ''
        self.vgen=self._generation;self.pending.clear()
    def _place_survival(self,event):
        ck=self._survival_key(event.container_id);c=self._survival_values(ck);uc=E.survival_block_probability(*c,0,0,inf);low=ck[0]=='missed_load' or uc<.15
        # Batch probability arithmetic retains math.exp and the frozen operation order.
        n=len(self.descriptors)
        if not hasattr(self,'_anchor_data'):self._anchor_data=[(0.,0.,inf)]
        for key in self.descriptors[len(self._anchor_data):]:
            a,m=E.SURVIVAL_PARAMETERS[key[0]];self._anchor_data.append((a,m,key[1]))
        data=np.asarray(self._anchor_data);ax=data[:,0];mx=data[:,1];tau=np.maximum(0,(data[:,2]-self.current_time)/3600)
        ac,mc,tc=c;jx=(ax>0)&np.isfinite(tau);jc=ac>0 and isfinite(tc)
        def probability(gamma):
            rate=mx+mc+gamma;ex=np.ones(n);ex[jx]=[exp(-r*t) for r,t in zip(rate[jx],tau[jx])]
            ec=np.array([exp(-r*tc) for r in rate]) if jc else np.ones(n)
            integral=np.ones(n);integral-=np.where(jx,ax*ex,0)
            if jc:integral-=ac*ec;integral+=np.where(jx,ax*ac*np.minimum(ex,ec),0)
            bg=mx/rate*integral
            precedes=np.where(tau>tc,1.,np.where(tau==tc,.5,0.)) if jc else np.zeros(n)
            atom=np.where(jx,ax*ex*(1-ac*precedes),0)
            return np.clip(bg+atom,0,1),atom,bg
        p,jump,bg=probability(1/240);_,j48,b48=probability(1/48)
        rate=mx+1/240;ex=np.ones(n);ex[jx]=[exp(-r*t) for r,t in zip(rate[jx],tau[jx])]
        u=np.clip(mx/rate*(1-np.where(jx,ax*ex,0))+np.where(jx,ax*ex,0),0,1)
        pv=p[self.ids];h=np.asarray(self._heights);d=pv.sum(axis=1);score=d.copy();umax=u[self.ids].max(axis=1)
        if self.mode=='ADP':
            w=self.theta
            score+=(w[1]-1)*bg[self.ids].sum(axis=1)
            if w[2]:score+=w[2]*j48[self.ids].sum(axis=1)
            if w[3]:score+=w[3]*b48[self.ids].sum(axis=1)
            if w[4]:score+=w[4]*(pv*self.init).sum(axis=1)
            if w[5]:score+=w[5]*(pv*(self.depart<self.current_time)).sum(axis=1)
            if w[6]:score+=w[6]*pv.max(axis=1)
            if w[7]:score+=w[7]*((tau[self.ids]<c[2])&(self.ids!=0)).sum(axis=1)
            if w[8] and isfinite(c[2]):
                mint=tau[self.ids].min(axis=1);gap=np.minimum((mint-c[2])/240,1);score+=w[8]*np.where((mint>c[2])&(self.ids[:,0]!=0),gap,0)
            score+=(h==0)*(w[9]+w[10]*uc)+w[11]*((h==1)&(umax>.25))+w[12]*uc*(h==self.maximum-1)+w[13]*(1-uc)*h
            if w[14]:score+=w[14]*((self.ids[:,0]!=0)&(self.top_vessel==self._meta[event.container_id].vessel))
            if w[15]:
                for b,indices in self._block_indices.items():
                    valid=[i for i in indices if h[i]<self.maximum[i]]
                    if valid:score[min(valid,key=lambda i:h[i])]+=w[15]
        elif self.mode=='CD1_aggressive_compact':
            if low:score-=.12*h
        elif self.mode=='CD2_drain_and_shield':score+=.30*((h==1)&(umax>.25))+.40*uc*(h==0)-(.10*(1-uc)*h if low else 0)
        elif self.mode=='CD3_sink_blocks':score+=np.where(self.sinks,-.20,.20) if low else .20*self.sinks if uc>=.25 else 0
        elif self.mode=='CD4_init_quarantine':score+=.8*(pv*self.init).sum(axis=1)+.15*((self.init&(u[self.ids]>.15)).any(axis=1))
        else:raise ValueError(self.mode)
        score[h>=self.maximum]=inf
        # np.lexsort's final key is primary; stable index is block/bay/row order.
        hh=-h if self.mode=='CD1_aggressive_compact' and low else h
        chosen=int(np.lexsort((np.arange(len(h)),hh,score))[0])
        if not isfinite(score[chosen]):raise RuntimeError('No non-full stack')
        return self._finish_placement(chosen)
Policy=VectorPolicy
