"""Flatten the fit audit and twelve survival/control evaluations."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/phase3c"


def main():
    audit=json.loads((OUT/"audit.json").read_text())
    runs=sum((json.loads((OUT/name).read_text()) for name in ("core.json","expanded.json","controls.json")),[])
    rows=[]
    for r in runs:
        rows.append(dict(name=r["name"],config=r["config"],total=r["summary"]["total_reshuffles"],
                         early=r["slices"]["period/1-12"]["reshuffles"],
                         late=r["slices"]["period/13-20"]["reshuffles"],
                         early_rate=r["slices"]["period/1-12"]["found_rate"],
                         late_rate=r["slices"]["period/13-20"]["found_rate"],
                         LOAD=r["slices"]["exit/LOAD"]["reshuffles"],
                         TRUCK_DLVR=r["slices"]["exit/TRUCK_DLVR"]["reshuffles"],
                         initial=r["slices"]["origin/initial"]["reshuffles"],
                         DISCHARGE=r["slices"]["origin/DISCHARGE"]["reshuffles"],
                         TRUCK_RECV=r["slices"]["origin/TRUCK_RECV"]["reshuffles"],
                         repeat=r["rehandled_blocker_moves"],seconds=r["wall_sec"],
                         score=r["score_exact"],excess=r["excess_above_LB"]))
    landings=collections.defaultdict(collections.Counter)
    for r in audit["relocation_landings_full_train_context_only"]:
        for key in ("height_before_landing","origin","state"):
            landings[key][str(r[key])]+=r["count"]
    (OUT/"tables.json").write_text(json.dumps(dict(experiments=rows,landing_marginals=dict(landings)),indent=2)+"\n")
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
    body{margin:0;font:14px var(--font-sans);color:var(--foreground)}select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:5px;margin-left:8px}svg{width:100%;display:block}.bars{height:490px}.curve{height:285px}text{fill:var(--foreground);font:12px var(--font-sans)}.base{fill:var(--chart-2)}.candidate{fill:var(--chart-1)}.selected{fill:var(--chart-3)}.grid{stroke:var(--border);stroke-width:1}.line{fill:none;stroke-width:2.5}.l0{stroke:var(--chart-1)}.l1{stroke:var(--chart-2)}.p0{fill:var(--chart-1)}.p1{fill:var(--chart-2)}.small{font-size:12px;color:var(--muted-foreground);line-height:1.5}h3{font-size:15px;margin:18px 0 8px}summary{cursor:pointer;margin:14px 0}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:5px;border-bottom:1px solid var(--border);text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}
    </style></head><body><label>Evaluation<select id="metric"><option value="early_rate">Days 1–12 / found retrieval · selection</option><option value="late_rate">Days 13–20 / found retrieval · continuation</option><option value="total">Full-train reshuffles</option></select></label><svg class="bars" id="bars" viewBox="0 0 700 490" role="img" aria-label="Survival and simple-control comparison"></svg><h3>Including pairs where both containers remain at Day 12</h3><p class="small">Blue: finite-cutoff blocking indicator. Orange: previous curve conditional on at least one observed departure.</p><svg class="curve" id="curve" viewBox="0 0 700 285" role="img" aria-label="Conditional versus finite-horizon ordering curves"></svg><p class="small">The finite-cutoff curve describes the sampled horizon event; it does not identify lifetime ordering beyond the cutoff.</p><details><summary>All experiment metrics and cohort counts</summary><div class="scroll"><table id="table"></table></div></details><script>
    const rows=ROWS,pairs=PAIRS,ns='http://www.w3.org/2000/svg';function add(root,tag,attrs,text){let e=document.createElementNS(ns,tag);for(let[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;root.appendChild(e);return e}const baseline=[{name:'Baseline 7,026',early_rate:3423/5042,late_rate:3603/3832,total:7026},{name:'M2a .20',early_rate:3421/5042,late_rate:3350/3832,total:6771}];function draw(){let root=document.querySelector('#bars');root.replaceChildren();let metric=document.querySelector('#metric').value,all=baseline.concat(rows),limit=Math.max(...all.map(r=>r[metric]))*1.14;all.forEach((r,i)=>{let y=7+i*34;add(root,'text',{x:0,y:y+15},r.name);let rect=add(root,'rect',{x:125,y,width:r[metric]/limit*490,height:21,rx:3,class:i<2?'base':r.name==='E_00'?'selected':'candidate'});let title=document.createElementNS(ns,'title');title.textContent=JSON.stringify(r.config||{});rect.appendChild(title);add(root,'text',{x:131+r[metric]/limit*490,y:y+15},metric==='total'?r[metric]:r[metric].toFixed(4))})}let root=document.querySelector('#curve');for(let t of[0,.25,.5,.75,1]){let y=230-t*210;add(root,'line',{x1:45,x2:675,y1:y,y2:y,class:'grid'});add(root,'text',{x:8,y:y+4},t.toFixed(2))}['horizon_rate','conditional_rate'].forEach((key,j)=>{let d='';pairs.forEach((r,i)=>{let x=55+i*76,y=230-r[key]*210;d+=(i?'L':'M')+x+','+y;add(root,'circle',{cx:x,cy:y,r:3,class:'p'+j});if(j===0)add(root,'text',{x,y:260,'text-anchor':'middle',style:'font-size:10px'},r.bin)});add(root,'path',{d,class:'line l'+j})});const cols=['name','total','early_rate','late_rate','LOAD','TRUCK_DLVR','initial','DISCHARGE','TRUCK_RECV','repeat','seconds'];let table=document.querySelector('#table'),head=table.insertRow();cols.forEach(k=>{let th=document.createElement('th');th.textContent=k;head.appendChild(th)});rows.forEach(r=>{let tr=table.insertRow();cols.forEach(k=>{let td=tr.insertCell(),v=r[k];td.textContent=typeof v==='number'&&!Number.isInteger(v)?v.toFixed(k==='seconds'?2:4):v})});document.querySelector('#metric').onchange=draw;draw();</script></body></html>'''
    html=html.replace('ROWS',json.dumps(rows)).replace('PAIRS',json.dumps(audit["pairwise"]["rows"]))
    (OUT/"comparison.html").write_text(html)
    print("Wrote tables.json and comparison.html")


if __name__=="__main__": main()
