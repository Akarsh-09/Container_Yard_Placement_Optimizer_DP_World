"""Summarize saved Phase 6B runs without executing any strategy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase6b'
def main():
    rows=json.loads((OUT/'ablations.json').read_text())
    assert len(rows)==21
    control=json.loads((OUT/'control_exact_copy.json').read_text())
    good=[r for r in rows if r['early']<control['early'] and r['late']<=control['late']]
    summary=dict(control=control,both_period_improvements=[r['name'] for r in sorted(good,key=lambda r:r['early'])],
                 early_ranking=[dict(name=r['name'],early=r['early'],late=r['late'],total=r['total']) for r in sorted(rows,key=lambda r:r['early'])],
                 rows=rows)
    (OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    html='''<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;font:14px var(--font-sans);color:var(--foreground)}select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:5px}svg{width:100%;height:760px}text{font:11px var(--font-sans);fill:var(--foreground)}p{font-size:12px;color:var(--muted-foreground);line-height:1.5}</style></head><body><select id="metric"><option value="early_rate">Days 1–12 · found retrieval rate</option><option value="late_rate">Days 13–20 · found retrieval rate</option><option value="total">Full-train reshuffles</option></select><svg viewBox="0 0 760 800" id="chart" role="img" aria-label="Isolated train-only structural lever ablations"></svg><p>All constants fit before Jan13. Oracle alpha is non-deployable. Green denotes improvement in both periods; no combinations or submitted-code changes.</p><script>const rows=ROWS,control=CONTROL,oracle=ORACLE,ns='http://www.w3.org/2000/svg';function add(tag,a,t){let e=document.createElementNS(ns,tag);for(let[k,v]of Object.entries(a))e.setAttribute(k,v);if(t!==undefined)e.textContent=t;document.querySelector('#chart').appendChild(e)}function draw(){document.querySelector('#chart').replaceChildren();let key=document.querySelector('#metric').value,all=[control,oracle].concat(rows),max=Math.max(...all.map(r=>r[key]))*1.10;all.forEach((r,i)=>{let y=10+i*33,w=r[key]/max*485,good=r.early<control.early&&r.late<=control.late;add('text',{x:0,y:y+15},r.name);add('rect',{x:205,y,width:w,height:23,rx:3,fill:i===0?'var(--chart-2)':i===1?'var(--chart-4)':good?'var(--chart-3)':'var(--chart-1)'});add('text',{x:211+w,y:y+15},key==='total'?r[key]:r[key].toFixed(4))})}document.querySelector('#metric').onchange=draw;draw();</script></body></html>'''
    oracle=json.loads((OUT/'D_oracle_alpha.json').read_text())
    (OUT/'comparison.html').write_text(html.replace('ROWS',json.dumps(rows)).replace('CONTROL',json.dumps(control)).replace('ORACLE',json.dumps(oracle)))
    for r in [control,oracle]+rows:
        print(r['name'],r['early'],f"{r['early_rate']:.4f}",r['late'],f"{r['late_rate']:.4f}",r['total'],r['excess'],r['LOAD'],r['TRUCK_DLVR'],r['I'],r['D'],r['R'],r['repeated'],f"{r['seconds']:.2f}")
    print('IMPROVES BOTH',summary['both_period_improvements'])
if __name__=='__main__':main()
