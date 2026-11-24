"""Assemble reviewable Phase 7A evidence without touching production artifacts."""
from common import *
from policies import NAMES
from html import escape

def read(name):return json.loads((OUT/name).read_text())
def main():
    before=read('integrity_before.json');after=integrity();assert before==after
    control=read('control.json');assert control['trace_sha256']==TRACE and control['total']==6408
    log=read('optimizer_log.json');assert len(log)==120 and [r['evaluation'] for r in log]==list(range(1,121))
    assert all(r['theta'][0]==1 for r in log)
    ml=read('ml_screen.json');sparse=read('sparse_verification.json');assert sparse['passed']
    names=['control','ADP_stage1','ADP_final','CD1_aggressive_compact','CD2_drain_and_shield','CD3_sink_blocks','CD4_init_quarantine']
    if ml['gate_passed']:names.append(ml['winner']+'_plugin')
    runs=[read(n+'.json') for n in names]
    for r in runs:
        assert r['early_found']==5042 and r['late_found']==3832
        assert r['total']==r['early']+r['late']==r['LOAD']+r['TRUCK_DLVR']==r['I']+r['D']+r['R']
        assert r['errors']==r['violations']==r['failures']==r['count_guard_fallbacks']==0
        r['breakthrough']=r['total']<=6150 and r['early']<3236 and r['late']<3172
        r['promotion_bar']=r['breakthrough'] and r['seconds']<20
    promotions=[r['name'] for r in runs[1:] if r['promotion_bar']]
    best=min(runs,key=lambda r:r['total'])
    recommendation=(f"Advance {', '.join(promotions)} to a separate Phase 7B integration and independent robustness review." if promotions else 'Keep the saved Phase 5B checkpoint as final. No Phase 7A candidate clears the complete promotion bar.')
    rollout=('A tightly bounded Track 4 single-block common-random-number rollout is warranted as a separate exploration: train-only, action-dependent future relocation dynamics remain untested by these three tracks. Do not replace Phase 5B on this evidence.' if not promotions else 'Prioritize Phase 7B validation before spending on a Track 4 rollout.')
    manifest=dict(protected_files_unchanged=True,test_contents_never_opened=True,test_access_blocked_by_audit_hook=True,protected_before=before,protected_after=after,
       snapshot_exact=(ROOT/'scripts/phase7a/engine_snapshot.py').read_bytes()==(ROOT/'solution/my_strategy.py').read_bytes(),scaffold_tests='88 passed',sparse_checks=sparse,
       fitted_outcomes_cutoff_exclusive='2025-01-13T00:00:00',optimizer_evaluations=120,fast_simulator_used=False,full_train_evaluations_use_real_unchanged_simulator_and_yard=True)
    manifest['phase7a_code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts/phase7a').glob('*.py'))}
    save('integrity_final.json',manifest)
    result=dict(control_trace_sha256=TRACE,runs=runs,stage1=read('stage1_weights.json'),final=read('final_weights.json'),optimizer_log=log,ml=ml,dataset=read('dataset_manifest.json'),simulator_verification=read('simulator_verification.json'),integrity=manifest,promotions=promotions,recommendation=recommendation,track4=rollout)
    save('report.json',result)
    def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    metrics=table(['Policy','D1–12 / 5,042','Rate','D13–20 / 3,832','Rate','Total','Official rate','Excess >1,325','Seconds','Empty at retrieval /8,874'],[[r['name'],r['early'],f"{r['early_rate']:.6f}",r['late'],f"{r['late_rate']:.6f}",r['total'],f"{r['official_rate']:.6f}",r['excess'],f"{r['seconds']:.2f}",r['empty_retrievals']] for r in runs])
    cohorts=table(['Policy','LOAD','TRUCK_DLVR','I','D','R','Repeated','Errors','Violations','Failures','Count guard'],[[r['name']]+[r[k] for k in ['LOAD','TRUCK_DLVR','I','D','R','repeated','errors','violations','failures','count_guard_fallbacks']] for r in runs])
    weights=table(['Feature','Stage 1 θ','Final θ'],[[f'f{i+1}: {n}',f"{result['stage1']['theta'][i]:.9f}",f"{result['final']['theta'][i]:.9f}"] for i,n in enumerate(NAMES)])
    mltable=table(['Model','Residents','Pairs','Positive','AUC','Brier'],[[r['model'],r['slice'],r['n'],r['positives'],f"{r['AUC']:.6f}",f"{r['Brier']:.6f}"] for r in ml['metrics']])
    features=table(['Feature','Standardized logistic coefficient'],[[r['feature'],f"{r['standardized_coefficient']:.6f}"] for r in ml['logistic_top_features'][:10]])+table(['Feature','GBDT split gain'],[[r['feature'],f"{r['split_gain']:.2f}"] for r in ml['gbdt_top_features'][:10]])
    svg='<svg viewBox="0 0 700 245" role="img" aria-label="Full train reshuffles by policy">'
    low=0;high=max(r['total'] for r in runs)+100
    for i,r in enumerate(runs):
        y=15+i*30;width=(r['total']-low)/(high-low)*440
        svg+=f'<text x="0" y="{y+14}" fill="currentColor" font-size="11">{escape(r["name"])}</text><rect x="190" y="{y}" width="{width:.2f}" height="19" fill="var(--chart-1,#4787df)"/><text x="{196+width:.2f}" y="{y+14}" fill="currentColor" font-size="12">{r["total"]}</text>'
    target=190+6150/high*440
    svg+=f'<line x1="{target:.2f}" x2="{target:.2f}" y1="5" y2="225" stroke="var(--warning,#d99030)" stroke-dasharray="3 3"/><text x="190" y="240" fill="currentColor" font-size="11">Bars start at zero; dashed line = 6,150 target</text></svg>'
    bestj=[];current=float('inf')
    for r in log:current=min(current,r['objective']);bestj.append(current)
    ymin=min(bestj)-2;ymax=max(bestj)+2
    pts=' '.join(f'{30+i*640/119:.1f},{150-(j-ymin)/(ymax-ymin)*130:.1f}' for i,j in enumerate(bestj))
    convergence=f'<svg viewBox="0 0 700 180" role="img" aria-label="Best training objective over 120 evaluations"><polyline points="{pts}" fill="none" stroke="var(--chart-2,#71b98d)" stroke-width="2"/><text x="30" y="175" fill="currentColor" font-size="12">1 → 120 evaluations; best J {bestj[0]:.3f} → {bestj[-1]:.3f}</text></svg>'
    notes='''<p>All candidates ran on the unchanged real Simulator and YardState. Runtime is the simulation loop wall time, excluding input loading and strategy initialization. No internal fast simulator was used. The full and strict-prefix θ₀ traces match the frozen control exactly. The score uses sum(p_tot) plus weighted deviations to preserve the control’s floating-point summation order; mathematically this is θᵀΦ with θ₁ fixed at one.</p>
<p>Optimization: seed 42; 120 objective evaluations; one deterministic sweep of both signs over coordinates 2–16 at step 0.25, then 0.08; remaining calls use (1+6)-ES, with a final partial batch to respect the cap. Bounds [-2,3]. J = early reshuffles + 5||θ₂:₁₆−θ₀₂:₁₆||². Accept only strict decreases. σ starts at .06 and decays by .85 on stalls. No Days 13–20 metrics enter selection.</p>
<p>Track 2: CD2’s compaction term applies to low-urgency arrivals, defined consistently as missed_load or u&lt;.15. CD1 changes height tie-breaking only for those arrivals. CD3 uses exactly B09/B10; B08 is also 720 slots in the supplied layout, but the named sink blocks were preserved.</p>
<p>Pair sampling: seed 707, uniform eligible arrival followed by uniform current real resident, with replacement. 150,000 fit pairs from Days 1–8 and 80,000 validation pairs from Days 13–16. Labels use actual successful exits, strict chronological outcome boundaries, and complete 96-hour horizons. Missing exit within a horizon means survival through the horizon; no censored row is assigned an outcome beyond its split. Initial residents remain in validation. See dataset_manifest.json for precise feature conventions.</p>
<p>M0 calibration has exactly one fitted parameter: slope of frozen logit, no intercept. M1 has an unpenalized intercept and L2=10 in summed loss; fit-only numeric standardization. M2 uses 30 depth-3 histogram Newton boosting trees, learning rate .1, min leaf 200, 16 fit-only quantile bins. Model hyperparameters were fixed before validation. A feature implementation correction changed port_weight_diff from weight-only rank to the prescribed combined frozen port/weight order; all final screen results use the corrected feature with unchanged hyperparameters. Repeated pairs and shared states are dependent, so these point estimates are diagnostics rather than independent-sample confidence claims.</p>
<p>Initial implementation diagnostics caught and corrected a stale overdue-descriptor cache before optimization. The accepted θ₀ and prefix hashes are exact. Scalar/vector agreement and synthetic sparse-stack tests cover full sparse stacks, physical gaps, ghost stacks, failed target removal, metadata restoration, and cached/live multistep equivalence.</p>'''
    body=f'<h1>Phase 7A exploration sprint</h1><p>{escape(recommendation)}</p><p>{escape(rollout)}</p><p>Control: 6,408; trace SHA-256 <code>{TRACE}</code>. Protected files remain unchanged; test contents were never opened or run.</p>{svg}<h2>Full-train real-simulator results</h2>{metrics}<p>The ADP finalist saves 50 early and 39 late reshuffles (89 total, 1.39%), but remains 169 above the 6,150 target and exceeds the 20-second runtime limit. CD2 increases empty-stack retrieval opportunities from 15 to 92 while worsening total reshuffles. CD4 cuts initial-origin reshuffles by 378 but adds 534 DISCHARGE-origin and 295 TRUCK_RECV-origin reshuffles. These policies shift costs rather than remove enough of them.</p><p>Rates divide reshuffles by found retrievals for each period; official rate divides by all 10,207 retrieval events. Empty counts are measured immediately before each found retrieval in its own block. Repeated counts are moves of real containers previously relocated.</p><details><summary>Cohort slices and operational guards</summary>{cohorts}</details><h2>Track 1 — direct closed-loop ADP</h2>{convergence}<p>Stage 1 J: {result["stage1"]["objective"]:.6f}; final J: {result["final"]["objective"]:.6f}. Complete convergence log: optimizer_log.json.</p><details><summary>All 16 learned weights</summary>{weights}</details><h2>Track 3 — dynamic pairwise screen</h2>{mltable}<p>Conditional plug-in gate: {"passed; winner "+ml["winner"] if ml["gate_passed"] else "failed; no plug-in simulator run"}.</p><details><summary>Top learned features</summary>{features}</details><details><summary>Protocol, numerical conventions, and verification</summary>{notes}</details>'
    css='''<style>body{font-family:var(--font-sans,system-ui);color:var(--foreground,#202834);line-height:1.5;margin:0}h1{font-size:22px}h2{font-size:18px;margin-top:25px}p{font-size:13px}code{font-size:11px;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:11px;width:100%;white-space:nowrap}td,th{text-align:right;padding:7px;border-bottom:1px solid var(--border,#bbb)}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}summary{cursor:pointer;font-size:13px;margin:14px 0}svg{display:block;width:100%;max-width:700px}details{border-bottom:1px solid var(--border,#bbb)}</style>'''
    (OUT/'report.html').write_text('<!doctype html><html><meta charset="utf-8">'+css+'<body>'+body+'</body></html>')
    # Compact inline comparison, with the complete report linked in the final answer.
    compact='<h2>Phase 7A: full-train reshuffles</h2>'+svg+f'<p>{escape(recommendation)}</p>'+table(['Policy','D1–12','D13–20','Total','Seconds'],[[r['name'],r['early'],r['late'],r['total'],f"{r['seconds']:.2f}"] for r in runs])
    (OUT/'summary.html').write_text('<!doctype html><html><meta charset="utf-8">'+css+'<body>'+compact+'</body></html>')
    print(recommendation,flush=True)
if __name__=='__main__':main()
