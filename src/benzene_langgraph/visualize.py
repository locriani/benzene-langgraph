"""Graph export and interactive Kekulé Benzene (C6H6) SVG/HTML renderer."""

from __future__ import annotations

import json
import math
from typing import Any

from benzene_langgraph.chemistry import (
    CARBON_MODEL_ID,
    CARBON_NODES,
    HYDROGEN_FACETS,
    HYDROGEN_MODEL_ID,
    HYDROGEN_NODES,
    RING_POSITIONS,
    KekuleIsomer,
    verify_benzene_topology,
)
from benzene_langgraph.graph import build_benzene_graph


def export_mermaid(compiled_graph: Any) -> str:
    """Export the native LangGraph Mermaid diagram showing all 12 atoms and 15 bonds."""
    return compiled_graph.get_graph().draw_mermaid()


def _serialize_isomer_graph(isomer: KekuleIsomer) -> dict[str, Any]:
    cg = build_benzene_graph(isomer=isomer, cycles=1, mock=True)
    report = verify_benzene_topology(cg, isomer=isomer)
    drawable = cg.get_graph()
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "label": e.data or "sigma_CH",
            "conditional": bool(e.conditional),
        }
        for e in drawable.edges
        if not e.source.startswith("__") and not e.target.startswith("__")
    ]
    sim = cg.invoke({"prompt": "Verify Kekulé double-bond preservation"})
    return {
        "isomer": isomer,
        "edges": edges,
        "ring_bond_orders": report.ring_bond_orders,
        "valencies": report.valencies,
        "total_covalent_edges": report.total_covalent_edges,
        "traversals": sim.get("bond_traversals", []),
    }


def render_benzene_html() -> str:
    """Render an interactive HTML/SVG view directly from the compiled LangGraph."""
    data_a = _serialize_isomer_graph("A")
    data_b = _serialize_isomer_graph("B")

    cx, cy = 220.0, 210.0
    r_carbon = 92.0
    r_hydrogen = 172.0

    coords: dict[str, tuple[float, float]] = {}
    for idx in range(6):
        angle_deg = -90.0 + idx * 60.0
        rad = math.radians(angle_deg)
        c_name = CARBON_NODES[idx]
        h_name = HYDROGEN_NODES[idx]
        coords[c_name] = (round(cx + r_carbon * math.cos(rad), 2), round(cy + r_carbon * math.sin(rad), 2))
        coords[h_name] = (round(cx + r_hydrogen * math.cos(rad), 2), round(cy + r_hydrogen * math.sin(rad), 2))

    atom_meta = {}
    for idx in range(6):
        c_name = CARBON_NODES[idx]
        h_name = HYDROGEN_NODES[idx]
        facet_title, facet_desc = HYDROGEN_FACETS[h_name]
        atom_meta[c_name] = {
            "element": "C",
            "model": CARBON_MODEL_ID,
            "valency": 4,
            "role": f"{RING_POSITIONS[c_name]}-Carbon (sp2 + 2pz)",
            "detail": f"4 covalent bonds: 1x C-H ({h_name}), 1x C-C single (sigma), 2x C=C double (sigma + pi)",
        }
        atom_meta[h_name] = {
            "element": "H",
            "model": HYDROGEN_MODEL_ID,
            "valency": 1,
            "role": facet_title,
            "detail": facet_desc,
        }

    payload_json = json.dumps(
        {
            "coords": coords,
            "atoms": atom_meta,
            "isomers": {"A": data_a, "B": data_b},
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  * {{ box-sizing: border-box; }}
  .wrap {{
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
    gap: 14px;
    align-items: start;
  }}
  @media (max-width: 640px) {{
    .wrap {{ grid-template-columns: 1fr; }}
  }}
  .panel {{
    background: var(--card);
    color: var(--card-foreground, var(--foreground));
    border: 1px solid var(--border);
    border-radius: var(--radius, 10px);
    padding: 16px;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }}
  .btn-group {{
    display: inline-flex;
    gap: 6px;
  }}
  button {{
    font-family: var(--font-sans, system-ui, sans-serif);
    font-size: 12px;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: calc(var(--radius, 8px) - 2px);
    border: 1px solid var(--border);
    background: var(--secondary);
    color: var(--secondary-foreground, var(--foreground));
    cursor: pointer;
  }}
  button.active {{
    background: var(--primary);
    color: var(--primary-foreground, var(--background));
    border-color: var(--primary);
  }}
  .svg-stage {{
    width: 100%;
    height: 420px;
    display: block;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 11.5px;
    color: var(--muted-foreground, var(--foreground));
    margin-top: 6px;
  }}
  .swatch {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .dot {{
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
  }}
  .kicker {{
    font-family: var(--font-mono, monospace);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted-foreground, var(--foreground));
    margin-bottom: 4px;
  }}
  .title-sm {{
    font-size: 15px;
    font-weight: 700;
    margin: 0 0 6px 0;
  }}
  .desc {{
    font-size: 12.5px;
    line-height: 1.45;
    color: var(--muted-foreground, var(--foreground));
    margin: 0 0 12px 0;
  }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 12px;
  }}
  .metric {{
    background: var(--secondary);
    border-radius: calc(var(--radius, 8px) - 2px);
    padding: 9px 10px;
  }}
  .metric-val {{
    font-family: var(--font-mono, monospace);
    font-size: 15px;
    font-weight: 700;
  }}
  .metric-lbl {{
    font-size: 11px;
    color: var(--muted-foreground, var(--foreground));
  }}
  .bond-list {{
    font-family: var(--font-mono, monospace);
    font-size: 11.5px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
  }}
  .bond-pill {{
    padding: 5px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .bond-pill.double {{
    border-color: var(--chart-1, var(--primary));
    background: var(--code-background, var(--secondary));
    font-weight: 600;
  }}
  .atom-node {{ cursor: pointer; }}
  .atom-node circle {{ transition: stroke-width 0.15s ease; }}
  .atom-node:hover circle, .atom-node.selected circle {{
    stroke: var(--ring, var(--primary));
    stroke-width: 3px;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <div class="toolbar">
      <div class="kicker">Compiled LangGraph Topology (C6H6)</div>
      <div class="btn-group">
        <button id="btn-a" class="active" type="button">Kekulé A (C1=C2)</button>
        <button id="btn-b" type="button">Kekulé B (C2=C3)</button>
      </div>
    </div>
    <svg id="mol-svg" class="svg-stage" viewBox="0 0 440 420" role="img" aria-label="Kekulé Benzene LangGraph"></svg>
    <div class="legend">
      <span class="swatch"><span class="dot" style="background: var(--chart-1, var(--primary));"></span>C1–C6: gemini-4-carbon (valency 4)</span>
      <span class="swatch"><span class="dot" style="background: var(--chart-2, var(--info));"></span>H1–H6: gemini-4-hydrogen (valency 1)</span>
      <span class="swatch"><span class="dot" style="background: var(--chart-4, var(--warning));"></span>π-bond (C=C double bond)</span>
    </div>
  </div>

  <div class="panel">
    <div class="kicker" id="sel-kicker">Selected Node · C1</div>
    <h3 class="title-sm" id="sel-title">C1 — ipso-Carbon (sp2 + 2pz)</h3>
    <p class="desc" id="sel-desc"></p>

    <div class="metrics">
      <div class="metric">
        <div class="metric-val" id="m-edges">15</div>
        <div class="metric-lbl">Covalent LangGraph Edges</div>
      </div>
      <div class="metric">
        <div class="metric-val">C=4 · H=1</div>
        <div class="metric-lbl">Atomic Valency Invariant</div>
      </div>
      <div class="metric">
        <div class="metric-val">2 : 1</div>
        <div class="metric-lbl">C=C (σ+π) : C–C (σ) Ratio</div>
      </div>
      <div class="metric">
        <div class="metric-val">4 Steps</div>
        <div class="metric-lbl">Parallel Pregel Supersteps</div>
      </div>
    </div>

    <div class="kicker">Ring Bond Orders (Preserved Double Bonding)</div>
    <div class="bond-list" id="bond-list"></div>
  </div>
</div>

<script>
const DATA = {payload_json};
let currentIsomer = "A";
let selectedAtom = "C1";

const svg = document.getElementById("mol-svg");

function render() {{
  const iso = DATA.isomers[currentIsomer];
  const coords = DATA.coords;
  const ns = "http://www.w3.org/2000/svg";
  svg.innerHTML = "";

  const defs = document.createElementNS(ns, "defs");
  const makeMarker = (id, colorVar) => {{
    const m = document.createElementNS(ns, "marker");
    m.setAttribute("id", id);
    m.setAttribute("viewBox", "0 0 10 10");
    m.setAttribute("refX", "8");
    m.setAttribute("refY", "5");
    m.setAttribute("markerWidth", "5.5");
    m.setAttribute("markerHeight", "5.5");
    m.setAttribute("orient", "auto-start-reverse");
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", "M 0 1.5 L 9 5 L 0 8.5 z");
    p.setAttribute("fill", colorVar);
    m.appendChild(p);
    return m;
  }};
  defs.appendChild(makeMarker("arr-sigma", "var(--muted-foreground, var(--foreground))"));
  defs.appendChild(makeMarker("arr-pi", "var(--chart-4, var(--warning))"));
  svg.appendChild(defs);

  // Count pair multiplicities to offset double-bond parallel edges cleanly
  const pairCounts = {{}};
  iso.edges.forEach(e => {{
    const k = [e.source, e.target].sort().join("|");
    pairCounts[k] = (pairCounts[k] || 0) + 1;
  }});

  iso.edges.forEach(e => {{
    const [x1, y1] = coords[e.source];
    const [x2, y2] = coords[e.target];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const nx = -uy;
    const ny = ux;

    const k = [e.source, e.target].sort().join("|");
    const isDouble = pairCounts[k] === 2;
    const isPi = e.label === "pi_bond";
    const offset = isDouble ? 5.0 : 0.0;

    const rStart = e.source.startsWith("H") ? 18 : 23;
    const rEnd = e.target.startsWith("H") ? 18 : 25;

    const sx = x1 + ux * rStart + nx * offset;
    const sy = y1 + uy * rStart + ny * offset;
    const ex = x2 - ux * rEnd + nx * offset;
    const ey = y2 - uy * rEnd + ny * offset;

    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", sx.toFixed(1));
    line.setAttribute("y1", sy.toFixed(1));
    line.setAttribute("x2", ex.toFixed(1));
    line.setAttribute("y2", ey.toFixed(1));
    if (isPi) {{
      line.setAttribute("stroke", "var(--chart-4, var(--warning))");
      line.setAttribute("stroke-width", "2.4");
      line.setAttribute("stroke-dasharray", "5 3");
      line.setAttribute("marker-end", "url(#arr-pi)");
    }} else {{
      line.setAttribute("stroke", isDouble ? "var(--chart-1, var(--primary))" : "var(--muted-foreground, var(--foreground))");
      line.setAttribute("stroke-width", isDouble ? "2.4" : "1.8");
      line.setAttribute("marker-end", "url(#arr-sigma)");
    }}
    svg.appendChild(line);
  }});

  Object.entries(coords).forEach(([name, [x, y]]) => {{
    const meta = DATA.atoms[name];
    const isC = meta.element === "C";
    const g = document.createElementNS(ns, "g");
    g.setAttribute("class", "atom-node" + (selectedAtom === name ? " selected" : ""));
    g.addEventListener("click", () => {{
      selectedAtom = name;
      render();
    }});

    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", x);
    c.setAttribute("cy", y);
    c.setAttribute("r", isC ? "21" : "16");
    c.setAttribute("fill", isC ? "var(--chart-1, var(--primary))" : "var(--chart-2, var(--info))");
    c.setAttribute("fill-opacity", isC ? "0.22" : "0.18");
    c.setAttribute("stroke", isC ? "var(--chart-1, var(--primary))" : "var(--chart-2, var(--info))");
    c.setAttribute("stroke-width", "2");
    g.appendChild(c);

    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", x);
    t.setAttribute("y", y + 4);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-family", "var(--font-mono, monospace)");
    t.setAttribute("font-size", isC ? "12.5" : "11");
    t.setAttribute("font-weight", "700");
    t.setAttribute("fill", "var(--foreground)");
    t.textContent = name;
    g.appendChild(t);

    svg.appendChild(g);
  }});

  // Center Kekulé label
  const centerLbl = document.createElementNS(ns, "text");
  centerLbl.setAttribute("x", "220");
  centerLbl.setAttribute("y", "214");
  centerLbl.setAttribute("text-anchor", "middle");
  centerLbl.setAttribute("font-family", "var(--font-mono, monospace)");
  centerLbl.setAttribute("font-size", "11");
  centerLbl.setAttribute("fill", "var(--muted-foreground, var(--foreground))");
  centerLbl.textContent = "Kekulé " + currentIsomer;
  svg.appendChild(centerLbl);

  // Update inspector
  const sel = DATA.atoms[selectedAtom];
  document.getElementById("sel-kicker").textContent = `${{sel.model}} · Valency ${{sel.valency}}`;
  document.getElementById("sel-title").textContent = `${{selectedAtom}} — ${{sel.role}}`;
  document.getElementById("sel-desc").textContent = sel.detail;
  document.getElementById("m-edges").textContent = String(iso.total_covalent_edges);

  const bondList = document.getElementById("bond-list");
  bondList.innerHTML = "";
  Object.entries(iso.ring_bond_orders).forEach(([bond, order]) => {{
    const div = document.createElement("div");
    div.className = "bond-pill" + (order === 2 ? " double" : "");
    div.innerHTML = `<span>${{bond}}</span><span>${{order === 2 ? "σ + π (2)" : "σ (1)"}}</span>`;
    bondList.appendChild(div);
  }});
}}

document.getElementById("btn-a").addEventListener("click", () => {{
  currentIsomer = "A";
  document.getElementById("btn-a").classList.add("active");
  document.getElementById("btn-b").classList.remove("active");
  render();
}});
document.getElementById("btn-b").addEventListener("click", () => {{
  currentIsomer = "B";
  document.getElementById("btn-b").classList.add("active");
  document.getElementById("btn-a").classList.remove("active");
  render();
}});

render();
</script>
</body>
</html>
"""
