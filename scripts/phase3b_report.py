"""Flatten Phase 3B autopsy/ablation evidence and produce an inline chart."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/phase3b"


def main():
    diagnostic=json.loads((OUT/"diagnostics.json").read_text())
    runs=json.loads((OUT/"ablations.json").read_text())
    flat=[]
    for r in runs:
        c=r["config"]
        flat.append(dict(name=r["name"],config=c,total=r["summary"]["total_reshuffles"],
                         early=r["slices"]["period/1-12"]["reshuffles"],
                         late=r["slices"]["period/13-20"]["reshuffles"],
                         early_rate=r["slices"]["period/1-12"]["found_rate"],
                         late_rate=r["slices"]["period/13-20"]["found_rate"],
                         early_lift=r["early_lift"],late_lift=r["late_lift"],
                         LOAD=r["slices"]["exit/LOAD"]["reshuffles"],
                         TRUCK_DLVR=r["slices"]["exit/TRUCK_DLVR"]["reshuffles"],
                         initial=r["slices"]["origin/initial"]["reshuffles"],
                         DISCHARGE=r["slices"]["origin/DISCHARGE"]["reshuffles"],
                         TRUCK_RECV=r["slices"]["origin/TRUCK_RECV"]["reshuffles"],
                         repeat=r["rehandled_blocker_moves"],seconds=r["wall_sec"]))
    marginals={}
    cells=diagnostic["A2"]["strategy_placement_cells"]
    for period in ("Days1-12","Days13-20"):
        selected=[r for r in cells if r["period"]==period]
        marginals[period]={}
        for col in ("belief","blocker_origin","missed_load_at_arrival","blocker_unretrieved_at_period_end",
                    "target_origin","target_state_at_placement","target_exit"):
            counts=collections.Counter()
            for row in selected: counts[str(row[col])]+=row["count"]
            marginals[period][col]=dict(counts)
    tables={"ablations":flat,"A2_marginals":marginals,"default_unchanged":True}
    (OUT/"tables.json").write_text(json.dumps(tables,indent=2)+"\n")
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
    body{margin:0;font:14px var(--font-sans);color:var(--foreground)}select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:5px;margin-left:8px}svg{width:100%;display:block}.lift{height:610px}.sources{height:120px}text{fill:var(--foreground);font:12px var(--font-sans)}.axis{stroke:var(--border);stroke-width:1}.up{fill:var(--chart-1)}.down{fill:var(--chart-2)}.s0{fill:var(--chart-1);background:var(--chart-1)}.s1{fill:var(--chart-2);background:var(--chart-2)}.s2{fill:var(--chart-3);background:var(--chart-3)}.small{font-size:12px;color:var(--muted-foreground);line-height:1.5}.legend{display:flex;gap:16px;font-size:12px}.dot{width:9px;height:9px;display:inline-block;margin-right:5px;border-radius:50%}h3{font-size:15px;margin:18px 0 10px}table{font-size:12px;border-collapse:collapse;white-space:nowrap}td,th{padding:5px;text-align:right;border-bottom:1px solid var(--border)}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}summary{cursor:pointer;margin:12px 0}
    </style></head><body><label>Reshuffles saved versus the unchanged 7,026 policy<select id="period"><option value="early_lift">Days 1–12 · selection period</option><option value="late_lift">Days 13–20 · continuation</option></select></label><svg id="lift" class="lift" viewBox="0 0 700 610" role="img" aria-label="Isolated ablation gains"></svg><p class="small">Right of zero: fewer reshuffles. Left: regression. All mechanisms were tested separately; the default is unchanged.</p><h3>Blocker’s most recent landing</h3><div class="legend"><span><i class="dot s0"></i>Initial, unmoved</span><span><i class="dot s1"></i>Strategy placement</span><span><i class="dot s2"></i>Simulator relocation</span></div><svg id="sources" class="sources" viewBox="0 0 700 120" role="img" aria-label="Blocker source proportions by period"></svg><p class="small">Last-landing provenance is not counterfactual causal responsibility: an earlier placement may cause a later relocation.</p><details><summary>All isolated results and cohort counts</summary><div class="scroll"><table id="table"></table></div></details><script>
    const rows=ROWS,sources=SOURCES,ns='http://www.w3.org/2000/svg';function add(root,tag,attrs,text){let e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;root.appendChild(e);return e}function draw(){let root=document.querySelector('#lift');root.replaceChildren();let key=document.querySelector('#period').value,extent=Math.max(1,...rows.map(r=>Math.abs(r[key]))),scale=240/extent,zero=410;add(root,'line',{x1:zero,x2:zero,y1:4,y2:600,class:'axis'});rows.forEach((r,i)=>{let y=12+i*34,v=r[key],width=Math.abs(v)*scale;add(root,'text',{x:0,y:y+15},r.name);let rect=add(root,'rect',{x:v>=0?zero:zero-width,y,width,height:21,rx:3,class:v>=0?'up':'down'});let title=document.createElementNS(ns,'title');title.textContent=JSON.stringify(r.config);rect.appendChild(title);add(root,'text',{x:v>=0?zero+width+6:zero-width-6,y:y+15,'text-anchor':v>=0?'start':'end'},(v>0?'+':'')+v)})}let sroot=document.querySelector('#sources');Object.entries(sources).forEach(([p,data],i)=>{let y=15+i*50,total=Object.values(data).reduce((a,b)=>a+b,0),x=125;add(sroot,'text',{x:0,y:y+18},p);['initial_unmoved','placed_by_strategy','placed_by_relocation'].forEach((k,j)=>{let width=data[k]/total*520;let rect=add(sroot,'rect',{x,y,width,height:28,class:'s'+j});let title=document.createElementNS(ns,'title');title.textContent=k+': '+data[k]+' ('+(data[k]/total*100).toFixed(1)+'%)';rect.appendChild(title);add(sroot,'text',{x:x+width/2,y:y+19,'text-anchor':'middle'},data[k]);x+=width})});let columns=['name','total','early_rate','late_rate','LOAD','TRUCK_DLVR','initial','DISCHARGE','TRUCK_RECV','repeat','seconds'],table=document.querySelector('#table'),head=table.insertRow();columns.forEach(k=>{let th=document.createElement('th');th.textContent=k;head.appendChild(th)});rows.forEach(r=>{let tr=table.insertRow();columns.forEach(k=>{let td=tr.insertCell(),v=r[k];td.textContent=typeof v==='number'&&!Number.isInteger(v)?v.toFixed(k==='seconds'?2:4):v})});document.querySelector('#period').onchange=draw;draw();
    </script></body></html>'''
    html=html.replace('ROWS',json.dumps(flat)).replace('SOURCES',json.dumps(diagnostic["A2"]["sources"]))
    (OUT/"comparison.html").write_text(html)
    print("Wrote tables.json and comparison.html")


if __name__=="__main__": main()
