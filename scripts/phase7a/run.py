from common import *
from policies import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
def worker_init():
    global WORKER_INPUTS
    WORKER_INPUTS=inputs()
def worker_eval(payload):
    name,w=payload
    return evaluate(name,Policy(theta=w),WORKER_INPUTS,early_only=True,persist=False)

def main():
    before=integrity();save('integrity_before.json',before)
    assert (ROOT/'scripts/phase7a/engine_snapshot.py').read_bytes()==(ROOT/'solution/my_strategy.py').read_bytes()
    inp=inputs()
    control=json.loads((OUT/'control.json').read_text()) if (OUT/'control.json').exists() else evaluate('control',E.MyStrategy(),inp);assert control['total']==6408 and control['trace_sha256']==TRACE
    zero=json.loads((OUT/'ADP_theta0.json').read_text()) if (OUT/'ADP_theta0.json').exists() else evaluate('ADP_theta0',Policy(),inp);assert zero['trace_sha256']==TRACE and zero['total']==6408
    # All optimization calls run only the strict prefix using the unchanged real simulator.
    early=json.loads((OUT/'ADP_theta0_prefix.json').read_text()) if (OUT/'ADP_theta0_prefix.json').exists() else evaluate('ADP_theta0_prefix',Policy(),inp,early_only=True)
    ref=json.loads((OUT/'control_prefix.json').read_text()) if (OUT/'control_prefix.json').exists() else evaluate('control_prefix',E.MyStrategy(),inp,early_only=True)
    assert early['trace_sha256']==ref['trace_sha256'] and early['total']==3236
    save('simulator_verification.json',dict(internal_fast_simulator_used=False,optimization_simulator='unchanged src.simulator.Simulator + src.yard_state.YardState',prefix_trace_identical=True,prefix_trace_sha256=early['trace_sha256'],prefix_reshuffles=3236,full_trace_sha256=TRACE))
    log=json.loads((OUT/'optimizer_log.json').read_text()) if (OUT/'optimizer_log.json').exists() else [];theta=np.array(THETA0);n=0
    def objective(w,stage):
        nonlocal n
        n+=1
        if n<=len(log):
            old=log[n-1];assert old['stage']==stage and np.array_equal(old['theta'],w.tolist())
            return old['objective']
        r=evaluate(f'opt_{n:03}',Policy(theta=w.tolist()),inp,early_only=True,persist=False)
        penalty=5*float(np.sum((w[1:]-np.array(THETA0)[1:])**2));j=r['total']+penalty
        log.append(dict(evaluation=n,stage=stage,theta=w.tolist(),reshuffles=r['total'],penalty=penalty,objective=j,seconds=r['seconds']))
        save('optimizer_log.json',log);return j
    best=objective(theta,'initial')
    # One deterministic cyclic sweep at each requested step; both signs are evaluated.
    # Reserve the remaining 59 calls for coupled ES refinement.
    for step in (.25,.08):
        for k in range(1,16):
            for sign in (1,-1):
                trial=theta.copy();trial[k]=np.clip(trial[k]+sign*step,-2,3)
                j=objective(trial,f'coordinate_{step}')
                if j<best:theta,best=trial,j
    stage1=theta.copy();stage1j=best
    save('stage1_weights.json',dict(names=NAMES,theta=theta.tolist(),objective=best,evaluations=n))
    evaluate('ADP_stage1',Policy(theta=theta.tolist()),inp)
    rng=np.random.default_rng(42);sigma=.06
    # Children in each ES generation share a frozen parent and are independent.
    # Numerical worker processes only; result acceptance stays ordered and deterministic.
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        while n<120:
            parent=theta.copy();trials=[];stage=f'ES_sigma_{sigma:.9f}'
            for _ in range(min(6,120-n)):
                trial=parent.copy();trial[1:]=np.clip(trial[1:]+rng.normal(0,sigma,15),-2,3);trials.append(trial)
            futures={}
            for offset,trial in enumerate(trials,1):
                number=n+offset
                if number>len(log):futures[number]=pool.submit(worker_eval,(f'opt_{number:03}',trial.tolist()))
            children=[]
            for trial in trials:
                n+=1
                if n<=len(log):
                    old=log[n-1];assert old['stage']==stage and np.array_equal(old['theta'],trial.tolist());j=old['objective']
                else:
                    r=futures[n].result();penalty=5*float(np.sum((trial[1:]-np.array(THETA0)[1:])**2));j=r['total']+penalty
                    log.append(dict(evaluation=n,stage=stage,theta=trial.tolist(),reshuffles=r['total'],penalty=penalty,objective=j,seconds=r['seconds']))
                    save('optimizer_log.json',log)
                children.append((j,trial))
            j,w=min(children,key=lambda z:z[0])
            if j<best:theta,best=w,j
            else:sigma*=.85
    save('final_weights.json',dict(names=NAMES,theta=theta.tolist(),objective=best,evaluations=n,bounds=[-2,3],seed=42,stage1_objective=stage1j))
    evaluate('ADP_final',Policy(theta=theta.tolist()),inp)
    for mode in ('CD1_aggressive_compact','CD2_drain_and_shield','CD3_sink_blocks','CD4_init_quarantine'):evaluate(mode,Policy(mode),inp)
    after=integrity();assert before==after
    save('integrity_after_tracks12.json',dict(unchanged=True,before=before,after=after,test_access_blocked_by_audit_hook=True))
if __name__=='__main__':main()
