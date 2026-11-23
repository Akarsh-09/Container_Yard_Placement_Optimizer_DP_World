"""Summarize the bounded sweep and produce an interactive comparison."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase3d'
def main():
    runs=json.loads((OUT/'runs.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    rows=[]
    for r in runs:
        s=r['slices']
        rows.append(dict(name=r['name'],config=r['config'],total=r['summary']['total_reshuffles'],
                         early=s['period/1-12']['reshuffles'],late=s['period/13-20']['reshuffles'],
                         early_rate=s['period/1-12']['found_rate'],late_rate=s['period/13-20']['found_rate'],
                         LOAD=s['exit/LOAD']['reshuffles'],TRUCK_DLVR=s['exit/TRUCK_DLVR']['reshuffles'],
                         initial=s['origin/initial']['reshuffles'],DISCHARGE=s['origin/DISCHARGE']['reshuffles'],
                         TRUCK_RECV=s['origin/TRUCK_RECV']['reshuffles'],repeat=r['rehandled_blocker_moves'],
                         seconds=r['wall_sec'],score=r['score_exact'],official_rate=r['official_rate_exact'],
                         excess=r['excess_above_LB'],eligible=r.get('eligible'),parent=r.get('parent')))
    (OUT/'tables.json').write_text(json.dumps(rows,indent=2)+'\n')
    html='''<!doctype html><html><head><meta charset="utf-8"><style>
    body{font:14px var(--font-sans);color:var(--foreground);margin:0}select{padding:7px}svg{width:100%;height:590px}text{font:12px var(--font-sans);fill:var(--foreground)}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid var(--border);text-align:right}.scroll{overflow-x:auto}h3{margin:12px 0}p{color:var(--muted-foreground)}
    </style></head><body><p>Selection uses Days 1–12. Days 13–20 is a repeatedly consulted continuation period.</p><select id="metric"><option value="early_rate">Days 1–12 reshuffles / found retrieval</option><option value="late_rate">Days 13–20 reshuffles / found retrieval</option><option value="total">Full-train reshuffles</option></select><svg id="chart" viewBox="0 0 800 650"></svg><details><summary>All experiment and cohort metrics</summary><div class="scroll"><table id="table"></table></div></details><script>
    const rows=ROWS,winner=WINNER,ns='http://www.w3.org/2000/svg';
    const refs=[{name:'Expected baseline',total:7026,early_rate:3423/5042,late_rate:3603/3832},{name:'3C Core H120',total:6626,early_rate:3293/5042,late_rate:3333/3832},{name:'3C Expanded H120',total:6595,early_rate:3286/5042,late_rate:3309/3832}];
    function add(tag,a,t){let e=document.createElementNS(ns,tag);for(let[k,v]of Object.entries(a))e.setAttribute(k,v);if(t!==undefined)e.textContent=t;document.querySelector('#chart').appendChild(e);return e}
    function draw(){document.querySelector('#chart').replaceChildren();let key=document.querySelector('#metric').value,all=refs.concat(rows),max=Math.max(...all.map(r=>r[key]))*1.12;all.forEach((r,i)=>{let y=10+i*36,w=r[key]/max*520;add('text',{x:0,y:y+16},r.name);let bar=add('rect',{x:170,y,width:w,height:24,rx:3,fill:r.name===winner?'var(--chart-3)':i<3?'var(--chart-2)':'var(--chart-1)'});let title=document.createElementNS(ns,'title');title.textContent=JSON.stringify(r.config||{});bar.appendChild(title);add('text',{x:176+w,y:y+16},key==='total'?r[key]:r[key].toFixed(4))})}
    let cols=['name','total','early_rate','late_rate','excess','LOAD','TRUCK_DLVR','initial','DISCHARGE','TRUCK_RECV','repeat','seconds'],table=document.querySelector('#table'),head=table.insertRow();cols.forEach(k=>{let th=document.createElement('th');th.textContent=k;head.appendChild(th)});rows.forEach(r=>{let tr=table.insertRow();cols.forEach(k=>{let v=r[k];tr.insertCell().textContent=typeof v==='number'&&!Number.isInteger(v)?v.toFixed(k==='seconds'?2:4):v})});document.querySelector('#metric').onchange=draw;draw();</script></body></html>'''
    (OUT/'comparison.html').write_text(html.replace('ROWS',json.dumps(rows)).replace('WINNER',json.dumps(selection['winner']['name'])))
    for r in rows:
        print(r['name'],r['early'],f"{r['early_rate']:.4f}",r['late'],f"{r['late_rate']:.4f}",r['total'],r['excess'],r['LOAD'],r['TRUCK_DLVR'],r['initial'],r['DISCHARGE'],r['TRUCK_RECV'],r['repeat'],f"{r['seconds']:.2f}")
if __name__=='__main__':main()
