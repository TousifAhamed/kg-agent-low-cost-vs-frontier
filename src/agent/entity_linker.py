"""Entity linker: map surface mentions in a question to concrete KG node ids.

Cheap, deterministic, and grounded — no LLM call. It recognizes:
  * well identifiers (F-12, F-15 D, ...)          -> Well node ids  (W_F-12, ...)
  * alert codes (WCT_SEVERE, WATER_BREAKTHROUGH, HIGH_WOR)
  * intervention action types (water_shutoff, ...)

The linked node ids seed k-hop retrieval (src/agent/retrieval.py) so the ReAct
loop starts from the right neighbourhood instead of blind semantic search.
Only ids that actually exist in the graph are returned, so every seed is citable.
"""
from __future__ import annotations

import re

ALERT_CODES = ("WCT_SEVERE", "WATER_BREAKTHROUGH", "HIGH_WOR")
ACTION_TYPES = ("water_shutoff", "rate_reduction", "gas_lift_opt", "polymer_injection")
WELL_RE = re.compile(r"\bF-?\s?\d+\s?[A-Z]?\b")


def _norm_well(tok: str) -> str:
    t = tok.upper().replace(" ", "")
    m = re.match(r"F-?(\d+)([A-Z]?)", t)
    return f"F-{m.group(1)}{m.group(2)}" if m else t


def link(question: str, kg) -> dict:
    """Return {wells, well_nodes, alert_codes, action_types, seed_nodes}.

    `kg` is an src.agent.retrieval.KG; only ids present in it are kept.
    """
    wells = sorted({_norm_well(t) for t in WELL_RE.findall(question)})
    well_nodes = [f"W_{w}" for w in wells if kg.has_node(f"W_{w}")]
    codes = [c for c in ALERT_CODES if c in question]
    actions = [a for a in ACTION_TYPES if a in question]
    return {"wells": wells, "well_nodes": well_nodes,
            "alert_codes": codes, "action_types": actions,
            "seed_nodes": well_nodes}
