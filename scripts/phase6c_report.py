"""Compile Phase 6C saved results, detailed screens and comparison artifact."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/phase6c'
def read(p):return json.loads((OUT/p).read_text())
def table(headers,rows):return '\n'.join([' | '.join(headers),' | '.join('---' for _ in headers)]+[' | '.join(str(v) for v in row) for row in rows])+'\n'
def main():
    control=read('control.json');oracles=read('oracles.json');runs=read('controls.json');features=read('feature_screen.json')
    h1=json.loads((ROOT/'results/phase3/H1_hindsight_LDT.json').read_text())
    h=dict(name='H1_reference',total=1961,early=h1['slices']['period/1-12']['reshuffles'],late=h1['slices']['period/13-20']['reshuffles'],
           early_rate=h1['slices']['period/1-12']['found_rate'],late_rate=h1['slices']['period/13-20']['found_rate'])
    for r in oracles:r['gap_closed_fraction']=(6408-r['total'])/(6408-1961)
    summary=dict(control=control,H1=h,oracles=oracles,controls=runs,feature_models=features['model_comparison'],
                 feature_supports=features['supports'],add_one_feature_lifts=features['add_one_feature_lift_over_003'],
                 single_feature_lifts=features['single_feature_lift_over_003'],
                 deployable_improvements_over_200=[r['name'] for r in runs if r['total']<6208],
                 improves_both_periods=[r['name'] for r in runs if r['early']<3236 and r['late']<=3172])
    (OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    text='PART 1: non-deployable information revelations (not additive)\n'+table(['oracle','early rate','late rate','total','saved','fraction of 4447 gap','LOAD','TRUCK','repeated','seconds'],
        [[r['name'],f"{r['early_rate']:.4f}",f"{r['late_rate']:.4f}",r['total'],6408-r['total'],f"{r.get('gap_closed_fraction',0):.3f}",r['LOAD'],r['TRUCK_DLVR'],r['repeated'],f"{r['seconds']:.2f}"] for r in [control]+oracles])
    text+='\nPART 2: chronologically validated models\n'+table(['target','model','AUC','Brier','n','positives'],[[r['target'],r['model'],f"{r['AUC']:.4f}",f"{r['Brier']:.5f}",r['n'],r['positives']] for r in features['model_comparison']])
    text+='\nPART 2: all univariate MI and validation screens\n'+table(['target','feature','fit MI bits','fit AUC','val AUC','val Brier','AUC lift'],[[r['target'],r['feature'],f"{r['fit_mutual_information_bits']:.5f}",f"{r['fit_AUC']:.4f}",f"{r['AUC']:.4f}",f"{r['Brier']:.5f}",f"{r['AUC_lift_vs_frozen']:+.4f}"] for r in features['univariate']])
    text+='\nPART 2: baseline plus one admissible feature\n'+table(['target','feature','val AUC','val Brier','AUC lift'],[[r['target'],r['feature'],f"{r['AUC']:.4f}",f"{r['Brier']:.5f}",f"{r['AUC_lift_vs_frozen']:+.4f}"] for r in features['baseline_plus_one_feature']])
    text+='\nPART 3: placement experiments\n'+table(['name','early rate','late rate','total','excess','LOAD','TRUCK','I','D','R','repeated','seconds'],
        [[r['name'],f"{r['early_rate']:.4f}",f"{r['late_rate']:.4f}",r['total'],r['excess'],r['LOAD'],r['TRUCK_DLVR'],r['I'],r['D'],r['R'],r['repeated'],f"{r['seconds']:.2f}"] for r in [control]+runs])
    (OUT/'tables.txt').write_text(text)
    html='''<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;font:14px var(--font-sans);color:var(--foreground)}select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:5px}svg{width:100%;height:500px}text{font:11px var(--font-sans);fill:var(--foreground)}p{font-size:12px;line-height:1.5;color:var(--muted-foreground)}</style></head><body><select id="view"><option value="oracle">Information-revelation diagnostics · full-train reshuffles</option><option value="early_rate">Placement controls · early found-retrieval rate</option><option value="late_rate">Placement controls · late found-retrieval rate</option><option value="total">Placement controls · full-train reshuffles</option></select><svg id="chart" viewBox="0 0 760 540" role="img" aria-label="Phase6C train-only diagnostics and controls"></svg><p id="note"></p><script>const control=CONTROL,oracles=ORACLES,runs=RUNS,h1=H1DATA,ns='http://www.w3.org/2000/svg';function add(t,a,v){let e=document.createElementNS(ns,t);for(let[k,x]of Object.entries(a))e.setAttribute(k,x);if(v!==undefined)e.textContent=v;document.querySelector('#chart').appendChild(e)}function draw(){document.querySelector('#chart').replaceChildren();let view=document.querySelector('#view').value,key=view==='oracle'?'total':view,rows=view==='oracle'?[control,...oracles,h1]:[control,...runs],max=Math.max(...rows.map(r=>r[key]))*1.12,gap=view==='oracle'?65:30;rows.forEach((r,i)=>{let y=10+i*gap,w=r[key]/max*460;add('text',{x:0,y:y+15},r.name);add('rect',{x:220,y,width:w,height:22,rx:3,fill:i===0?'var(--chart-2)':view==='oracle'?'var(--chart-4)':r.early<control.early&&r.late<=control.late?'var(--chart-3)':'var(--chart-1)'});add('text',{x:226+w,y:y+15},key==='total'?r[key]:r[key].toFixed(4))});document.querySelector('#note').textContent=view==='oracle'?'Oracles reveal future training outcomes and are non-deployable. Partial-revelation improvements overlap; H1 is a reference heuristic, not a proven optimum.':'All placement controls use frozen cohort parameters. None saves >200 reshuffles or improves both periods; the submitted strategy is unchanged.'}document.querySelector('#view').onchange=draw;draw();</script></body></html>'''
    (OUT/'comparison.html').write_text(html.replace('CONTROL',json.dumps(control)).replace('ORACLES',json.dumps(oracles)).replace('RUNS',json.dumps(runs)).replace('H1DATA',json.dumps(h)))
    print('Wrote report.json, tables.txt, comparison.html')
if __name__=='__main__':main()
