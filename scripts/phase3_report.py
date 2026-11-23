"""Assemble Phase 3 evidence tables and a self-contained interactive chart."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/phase3"


def main():
    a=json.loads((OUT/"stage_A.json").read_text())
    b=json.loads((OUT/"stage_B.json").read_text())
    transfer=json.loads((OUT/"stage_B_transfer.json").read_text())
    selection=json.loads((OUT/"selection.json").read_text())
    winner=selection["winner"]
    rows=a+b[1:]+[transfer]
    flat=[]
    for r in rows:
        config=dict(r["config"])
        config.setdefault("lambda_empty",0.0)
        config.setdefault("lambda_gap",0.0)
        flat.append(dict(name=r["name"],**config,reshuffles=r["summary"]["total_reshuffles"],
                         early_found_rate=r["slices"]["period/1-12"]["found_rate"],
                         late_found_rate=r["slices"]["period/13-20"]["found_rate"],
                         early_reshuffles=r["slices"]["period/1-12"]["reshuffles"],
                         late_reshuffles=r["slices"]["period/13-20"]["reshuffles"],
                         excess_above_LB=r["excess_above_LB"],score=r["score_exact"],runtime=r["wall_sec"],
                         LOAD=r["slices"]["exit/LOAD"]["reshuffles"],
                         TRUCK_DLVR=r["slices"]["exit/TRUCK_DLVR"]["reshuffles"],
                         initial=r["slices"]["origin/initial"]["reshuffles"],
                         DISCHARGE=r["slices"]["origin/DISCHARGE"]["reshuffles"],
                         TRUCK_RECV=r["slices"]["origin/TRUCK_RECV"]["reshuffles"],
                         repeated=r["rehandled_blocker_moves"]))
    (OUT/"all_tables.json").write_text(json.dumps(flat,indent=2)+"\n")
    finalists=[min((r for r in a if r["config"]["timing_key"]==timing),
                    key=lambda r:r["slices"]["period/1-12"]["reshuffles"])
               for timing in ("raw","s2_rollover","s3_next_feasible")]
    if winner["name"] not in {r["name"] for r in finalists}: finalists.append(winner)
    bars=[{"name":"Greedy", "total":8036,"early":4138/5042,"late":3898/3832}]
    for r in finalists:
        c=r["config"]
        label=r["name"]+" · "+c["timing_key"].replace("_rollover","").replace("_next_feasible","")
        bars.append({"name":label,"total":r["summary"]["total_reshuffles"],
                     "early":r["slices"]["period/1-12"]["found_rate"],
                     "late":r["slices"]["period/13-20"]["found_rate"]})
    pairwise=json.loads((OUT/"D2_pairwise.json").read_text())
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
    body{margin:0;font:14px var(--font-sans);color:var(--foreground)}select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:5px;margin-left:8px}svg{width:100%;display:block}.bars{height:230px}.curve{height:290px}text{fill:var(--foreground);font:12px var(--font-sans)}.small{color:var(--muted-foreground);font-size:12px;line-height:1.5}.legend{display:flex;gap:16px;font-size:12px;margin:10px 0}.dot{display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:50%}.grid{stroke:var(--border);stroke-width:1}.s0{fill:var(--chart-1);background:var(--chart-1)}.s1{fill:var(--chart-2);background:var(--chart-2)}.s2{fill:var(--chart-3);background:var(--chart-3)}.line0{stroke:var(--chart-1)}.line1{stroke:var(--chart-2)}.line2{stroke:var(--chart-3)}.line{fill:none;stroke-width:2.5}h3{font-size:15px;margin:20px 0 10px}table{font-size:12px;border-collapse:collapse;white-space:nowrap}th,td{text-align:right;padding:6px;border-bottom:1px solid var(--border)}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}summary{cursor:pointer;margin-top:20px}
    </style></head><body><label>Best early-selected configuration per timing key<select id="metric"><option value="total">Full-train reshuffles</option><option value="early">Days 1–12 / found retrieval</option><option value="late">Days 13–20 / found retrieval</option></select></label><svg id="bars" class="bars" viewBox="0 0 700 230" role="img" aria-label="Selected configuration comparison"></svg>
    <h3>Observed pairwise ordering, fit period only</h3><label>Resident cohort<select id="cohort"><option value="all">All eligible pairs</option><option value="initial">Initial containers</option><option value="new_arrival">New arrivals</option><option value="overdue">Overdue</option><option value="not_overdue">Not overdue</option></select></label><div class="legend"><span><i class="dot s0"></i>Raw</span><span><i class="dot s1"></i>S2 rollover</span><span><i class="dot s2"></i>S3 next feasible</span></div><svg id="curve" class="curve" viewBox="0 0 700 290" role="img" aria-label="Probability resident is retrieved first by estimated time difference"></svg><p class="small">Vertical: observed P(resident retrieved first). Horizontal: estimated resident time minus incoming time, hours. Pairs where both containers remain at Day 12 are excluded; these curves are conditional, not full-population calibration.</p><details><summary>All 45 grid evaluations and cohort counts</summary><div class="scroll"><table id="table"></table></div></details>
    <script>const bars=BARS,curves=CURVES,rows=ROWS,ns='http://www.w3.org/2000/svg';function add(root,tag,attrs,text){let e=document.createElementNS(ns,tag);for(let [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;root.appendChild(e);return e}function drawBars(){let root=document.querySelector('#bars');root.replaceChildren();let key=document.querySelector('#metric').value,limit=Math.max(...bars.map(r=>r[key]))*1.15;bars.forEach((r,i)=>{let y=12+i*42;add(root,'text',{x:0,y:y+15},r.name);add(root,'rect',{x:170,y:y,width:r[key]/limit*430,height:22,rx:3,class:i===0?'s1':'s0'});add(root,'text',{x:176+r[key]/limit*430,y:y+16},key==='total'?r[key]:r[key].toFixed(4))})}function drawCurve(){let root=document.querySelector('#curve');root.replaceChildren();for(let t of[0,.25,.5,.75,1]){let y=235-t*215;add(root,'line',{x1:45,x2:675,y1:y,y2:y,class:'grid'});add(root,'text',{x:8,y:y+4},t.toFixed(2))}let group=document.querySelector('#cohort').value;['raw','s2_rollover','s3_next_feasible'].forEach((timing,j)=>{let data=curves[timing+'/'+group],path='',previous=false;data.forEach((r,i)=>{let x=55+i*76;if(j===0)add(root,'text',{x,y:265,'text-anchor':'middle',style:'font-size:10px'},r.bin);if(r.probability===null){previous=false;return}let y=235-r.probability*215;path+=(previous?'L':'M')+x+','+y;previous=true;let c=add(root,'circle',{cx:x,cy:y,r:3,class:'s'+j});let title=document.createElementNS(ns,'title');title.textContent=timing+' '+r.bin+': '+r.probability.toFixed(4)+' (n='+r.n+')';c.appendChild(title)});add(root,'path',{d:path,class:'line line'+j})})}let cols=['name','timing_key','epsilon','sigma_hours','lambda_empty','lambda_gap','reshuffles','early_found_rate','late_found_rate','LOAD','TRUCK_DLVR','initial','DISCHARGE','TRUCK_RECV','repeated','runtime'];let table=document.querySelector('#table');let head=table.insertRow();cols.forEach(c=>{let th=document.createElement('th');th.textContent=c;head.appendChild(th)});rows.forEach(r=>{let tr=table.insertRow();cols.forEach(c=>{let td=tr.insertCell(),v=r[c];td.textContent=typeof v==='number'&&!Number.isInteger(v)?v.toFixed(c==='runtime'?2:4):v})});document.querySelector('#metric').onchange=drawBars;document.querySelector('#cohort').onchange=drawCurve;drawBars();drawCurve();</script></body></html>'''
    html=html.replace('BARS',json.dumps(bars)).replace('CURVES',json.dumps(pairwise["tables"])).replace('ROWS',json.dumps(flat))
    (OUT/"comparison.html").write_text(html)
    print("Wrote all_tables.json and comparison.html")


if __name__=="__main__": main()
