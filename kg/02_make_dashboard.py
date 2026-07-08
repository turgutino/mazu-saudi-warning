# =============================================================================
# MAZU KG Dashboard generator
# Reads kg_data.json and writes a self-contained interactive HTML page
# (data embedded inline so it works by double-click, no server needed).
# Output: dashboard/kg_view.html
# =============================================================================

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
KG_JSON = os.path.join(HERE, "kg_data.json")
OUT_HTML = os.path.join(HERE, "..", "dashboard", "kg_view.html")

with open(KG_JSON, encoding="utf-8") as f:
    kg = json.load(f)

# ── colour maps ─────────────────────────────────────────────────────────
NODE_COLORS = {
    "Indicator":  "#2BC8E2",
    "Hazard":     "#FF5A5A",
    "Mechanism":  "#B07AFF",
    "Event":      "#FFD166",
    "Region":     "#1DDBA0",
    "DataSource": "#8AA0B4",
}
NODE_SHAPES = {
    "Indicator":  "dot",
    "Hazard":     "diamond",
    "Mechanism":  "hexagon",
    "Event":      "star",
    "Region":     "triangle",
    "DataSource": "square",
}
EDGE_COLORS = {
    "contributes_to":  "#FF7D45",
    "triggers":        "#B07AFF",
    "driven_by":       "#9B6DFF",
    "at_risk_of":      "#FF8FA3",
    "exposed_to":      "#4FD6C0",
    "correlates_with": "#3A6EA5",
    "sourced_from":    "#5D7A8C",
    "occurs_at":       "#1DDBA0",
    "manifests_as":    "#FF5A5A",
    "observed_value":  "#FFD166",
}

# degree (for node sizing)
from collections import Counter as _C
deg = _C()
for e in kg["links"]:
    deg[e["source"]] += 1
    deg[e["target"]] += 1

# ── build vis nodes / edges ─────────────────────────────────────────────
vis_nodes = []
for n in kg["nodes"]:
    nt = n.get("ntype", "Indicator")
    tip = [f"<b>{n.get('label', n['id'])}</b>", f"type: {nt}"]
    for key in ("desc", "unit", "source", "date", "value", "hazard", "location", "kind"):
        if n.get(key):
            tip.append(f"{key}: {n[key]}")
    base = {"Hazard": 6, "Mechanism": 4, "Event": 3}.get(nt, 1)
    vis_nodes.append({
        "id": n["id"],
        "label": n.get("label", n["id"]),
        "group": nt,
        "color": NODE_COLORS.get(nt, "#cccccc"),
        "shape": NODE_SHAPES.get(nt, "dot"),
        "title": "<br>".join(tip),
        "value": base + deg.get(n["id"], 0) * 0.5,
    })

vis_edges = []
for _i, e in enumerate(kg["links"]):
    et = e.get("etype", "")
    lbl = ""
    if et == "correlates_with" and "weight" in e:
        lbl = f"r={e['weight']}"
    elif et == "observed_value" and "value" in e:
        lbl = str(e["value"])
    vis_edges.append({
        "id": _i,
        "from": e["source"],
        "to": e["target"],
        "color": {"color": EDGE_COLORS.get(et, "#888"), "opacity": 0.45},
        "etype": et,
        "val": lbl,
        "dashes": et in ("correlates_with", "observed_value"),
        "arrows": "" if et == "correlates_with" else "to",
    })

# counts
from collections import Counter
ncnt = Counter(n.get("ntype") for n in kg["nodes"])
ecnt = Counter(e.get("etype") for e in kg["links"])

NODES_JS = json.dumps(vis_nodes, ensure_ascii=False)
EDGES_JS = json.dumps(vis_edges, ensure_ascii=False)

legend_nodes = "".join(
    f'<span class="lg"><i style="background:{NODE_COLORS[k]}"></i>{k} <b>{ncnt.get(k,0)}</b></span>'
    for k in NODE_COLORS)
legend_edges = "".join(
    f'<span class="lg"><i class="ln" style="background:{EDGE_COLORS[k]}"></i>{k} <b>{ecnt.get(k,0)}</b></span>'
    for k in EDGE_COLORS)

html = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAZU - Saudi Arabia Extreme Events Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  :root{--bg:#0E1B2A;--panel:#16283C;--line:#243B54;--txt:#E6EEF6;--mut:#8AA0B4;--accent:#2BC8E2}
  *{box-sizing:border-box}
  body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--txt);height:100vh;overflow:hidden}
  header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
  header h1{font-size:19px;margin:0;font-weight:700}
  header h1 span{color:var(--accent)}
  header .sub{color:var(--mut);font-size:13px}
  header .stat{margin-left:auto;color:var(--mut);font-size:13px}
  header .stat b{color:var(--txt);font-size:15px}
  .wrap{display:flex;height:calc(100vh - 56px)}
  #net{flex:1;min-width:0;height:100%}
  aside{width:330px;height:100%;border-left:1px solid var(--line);background:var(--panel);padding:16px;overflow-y:auto}
  aside h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
  .legend{display:flex;flex-wrap:wrap;gap:6px 12px;margin-bottom:18px}
  .lg{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--mut)}
  .lg b{color:var(--txt)}
  .lg i{width:11px;height:11px;border-radius:50%;display:inline-block}
  .lg i.ln{width:16px;height:3px;border-radius:2px}
  #detail{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:14px;min-height:120px}
  #detail .dt{font-size:16px;font-weight:700;color:var(--accent);margin-bottom:6px}
  #detail .row{font-size:13px;margin:3px 0;color:var(--mut)}
  #detail .row b{color:var(--txt)}
  #detail .chip{display:inline-block;background:var(--line);color:var(--txt);font-size:11px;padding:2px 8px;border-radius:10px;margin:2px 3px 0 0}
  .hint{color:var(--mut);font-size:12px}
  .filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}
  .filters button{background:var(--line);color:var(--txt);border:none;border-radius:14px;padding:5px 11px;font-size:12px;cursor:pointer}
  .filters button.off{opacity:.35}
</style></head>
<body>
<header>
  <h1><span>MAZU</span> Saudi Arabia Extreme Events Knowledge Graph</h1>
  <div class="sub">multi-hazard · causal · data-grounded</div>
  <div class="stat">Nodes <b>__NNODES__</b> &nbsp; Edges <b>__NEDGES__</b></div>
</header>
<div class="wrap">
  <div id="net"></div>
  <aside>
    <h3>Search</h3>
    <input id="search" placeholder="type a node name (e.g. ivt, Jeddah, ARST)"
      style="width:100%;padding:8px 10px;margin-bottom:16px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--txt);font-size:13px"/>
    <h3>Filter node types</h3>
    <div class="filters" id="filters"></div>
    <h3>Details</h3>
    <div id="detail"><div class="hint">Click any node to see its relationships and metadata.</div></div>
    <h3 style="margin-top:18px">Node legend</h3>
    <div class="legend">__LEGN__</div>
    <h3>Edge legend</h3>
    <div class="legend">__LEGE__</div>
  </aside>
</div>
<script>
const NODES = __NODES__;
const EDGES = __EDGES__;
const nodes = new vis.DataSet(NODES);
const edges = new vis.DataSet(EDGES);
const container = document.getElementById('net');
const data = {nodes, edges};
const options = {
  nodes:{font:{color:'#E6EEF6',size:14,strokeWidth:3,strokeColor:'#0E1B2A'},borderWidth:0,scaling:{min:10,max:28}},
  edges:{smooth:{type:'continuous'},width:1.1},
  physics:{barnesHut:{gravitationalConstant:-12000,centralGravity:0.8,springLength:120,springConstant:0.04,damping:0.1,avoidOverlap:0.8},stabilization:{iterations:450},minVelocity:0.5},
  interaction:{hover:true,tooltipDelay:120,navigationButtons:true,keyboard:false},
  groups:{}
};
const network = new vis.Network(container, data, options);
network.once('stabilizationIterationsDone', ()=>{ network.setOptions({physics:false}); setTimeout(()=>network.fit({animation:false}),80); });

const nodeById = {}; NODES.forEach(n=>nodeById[n.id]=n);
function showDetail(id){
  const n = nodeById[id]; if(!n) return;
  const inc = EDGES.filter(e=>e.to===id);
  const out = EDGES.filter(e=>e.from===id);
  let h = '<div class="dt">'+n.label+'</div>';
  h += '<div class="row"><b>Type:</b> '+n.group+'</div>';
  if(n.title){ n.title.split('<br>').slice(1).forEach(t=>{ if(t && !t.startsWith('type:')) h+='<div class="row">'+t+'</div>';}); }
  const rels = {};
  out.forEach(e=>{ (rels[e.etype]=rels[e.etype]||[]).push('&rarr; '+e.to+(e.val?' ('+e.val+')':'')); });
  inc.forEach(e=>{ (rels[e.etype]=rels[e.etype]||[]).push('&larr; '+e.from+(e.val?' ('+e.val+')':'')); });
  Object.keys(rels).forEach(k=>{ h+='<div class="row" style="margin-top:8px"><b>'+k+'</b></div>'; rels[k].forEach(r=>h+='<span class="chip">'+r+'</span>'); });
  document.getElementById('detail').innerHTML = h;
}
network.on('click', p=>{ if(p.nodes.length){ showDetail(p.nodes[0]); highlight(p.nodes[0]); } else { clearHi(); } });

// hover / click highlight: emphasise a node and its neighbours
const baseColors={}; NODES.forEach(n=>baseColors[n.id]=n.color);
function neighbours(id){ const s=new Set([id]); EDGES.forEach(e=>{ if(e.from===id)s.add(e.to); if(e.to===id)s.add(e.from); }); return s; }
function highlight(id){
  const keep=neighbours(id);
  nodes.update(NODES.map(n=>({id:n.id, color: keep.has(n.id)?baseColors[n.id]:'rgba(90,110,130,0.18)', font:{color: keep.has(n.id)?'#E6EEF6':'rgba(150,165,180,0.25)'}})));
  edges.update(EDGES.map((e,i)=>({id:i, color:{color:(e.from===id||e.to===id)?e.color.color:'rgba(90,110,130,0.08)',opacity:(e.from===id||e.to===id)?0.9:0.15}})));
}
function clearHi(){
  nodes.update(NODES.map(n=>({id:n.id,color:baseColors[n.id],font:{color:'#E6EEF6'}})));
  edges.update(EDGES.map((e,i)=>({id:i,color:{color:e.color.color,opacity:0.5}})));
}
network.on('hoverNode', p=>highlight(p.node));
network.on('blurNode', ()=>clearHi());

// search
document.getElementById('search').addEventListener('keydown', ev=>{
  if(ev.key!=='Enter') return;
  const q=ev.target.value.trim().toLowerCase(); if(!q) return;
  const hit=NODES.find(n=>n.id.toLowerCase().includes(q)||(n.label||'').toLowerCase().includes(q));
  if(hit){ network.focus(hit.id,{scale:1.1,animation:{duration:500}}); network.selectNodes([hit.id]); showDetail(hit.id); highlight(hit.id); }
});

// type filters
const types=[...new Set(NODES.map(n=>n.group))];
const fdiv=document.getElementById('filters'); const active=new Set(types);
types.forEach(t=>{ const b=document.createElement('button'); b.textContent=t; b.onclick=()=>{
  if(active.has(t)){active.delete(t);b.classList.add('off');}else{active.add(t);b.classList.remove('off');}
  nodes.update(NODES.map(n=>({id:n.id,hidden:!active.has(n.group)})));
}; fdiv.appendChild(b); });
</script>
</body></html>"""

html = (html.replace("__NODES__", NODES_JS).replace("__EDGES__", EDGES_JS)
        .replace("__NNODES__", str(len(vis_nodes))).replace("__NEDGES__", str(len(vis_edges)))
        .replace("__LEGN__", legend_nodes).replace("__LEGE__", legend_edges))

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"[SAVED] {os.path.abspath(OUT_HTML)}")
print(f"Nodes: {len(vis_nodes)} | Edges: {len(vis_edges)} | Size: {round(len(html)/1024,1)} KB")
