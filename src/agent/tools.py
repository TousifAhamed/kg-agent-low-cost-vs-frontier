"""The three agent tools (roadmap M3), plus their Anthropic tool schemas.

  1. cypher_query    — structured graph retrieval (nodes by label+filter, neighbours,
                       k-hop subgraph). The KG-native counterpart to text search.
  2. semantic_search — FAISS over the dashboard corpus; the fuzzy fallback for
                       under-specified questions (same corpus S2 uses).
  3. compute_metric  — DETERMINISTIC aggregation/ranking over the graph
                       (mean/sum/min/max/count/rank, incl. derived ratios). This is
                       the capability Text-RAG lacked (M2: Recommendation <=0.18,
                       NDCG@5 ~0) — it returns exact ids + numbers, no hallucination.

Every tool result carries the contributing KG node ids in `cited_ids`, which the
citation guardrail (src/agent/cite.py) checks against the graph. Filters are
structured (never interpolated), so tool use is injection-safe.
"""
from __future__ import annotations

from typing import Any

MAX_ROWS = 40  # cap payloads so the context stays small and deterministic

SCHEMAS = [
    {
        "name": "cypher_query",
        "description": (
            "Retrieve exact records from the knowledge graph. Use for facts about specific "
            "wells, connectivity, alerts, or interventions. Returns records including their "
            "KG node id (cite these ids). Node labels: Well (attrs: short, well_type), "
            "MetricReading (metric in {water_cut,log_wor,np_cum,pressure}, value, timestamp, well), "
            "Alert (alert_code in {WCT_SEVERE,WATER_BREAKTHROUGH,HIGH_WOR}, well, raised_at), "
            "Intervention (action_type, est_cost_per_bbl, expected_uplift_bbl, well). "
            "INJECTS_INTO edges (weight, tau_days) connect injector->producer wells."),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["nodes", "neighbours", "subgraph", "latest"],
                              "description": "nodes=list by label+filter; neighbours=edges of a node id; subgraph=k-hop around seed ids; latest=most-recent MetricReading for a well+metric (use for 'most recent/latest' value or date questions)"},
                "label": {"type": "string", "enum": ["Well", "MetricReading", "Alert", "Intervention"]},
                "filters": {"type": "object", "description": "attribute->value equality filter, e.g. {\"metric\":\"water_cut\",\"well\":\"F-12\"}. 'well' matches by suffix."},
                "node_id": {"type": "string", "description": "for operation=neighbours"},
                "seed_ids": {"type": "array", "items": {"type": "string"}, "description": "for operation=subgraph"},
                "hops": {"type": "integer", "description": "k for subgraph (default 1)"},
            },
            "required": ["operation"],
        },
    },
    {
        "name": "semantic_search",
        "description": ("Fuzzy text search over serialized dashboard documents. Use only when you "
                        "cannot form a precise cypher_query. Returns doc ids + text."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "compute_metric",
        "description": (
            "Deterministically aggregate or rank over graph nodes. Use for 'how many', "
            "'mean/total cost', 'cheapest/highest', 'rank the top/bottom N'. "
            "metric=count|sum|mean|min|max return a value + the contributing node ids; "
            "metric=rank returns node ids ordered by `attribute` (respect `order` and `limit`). "
            "For best uplift-per-cost, set ratio_num=expected_uplift_bbl, ratio_den=est_cost_per_bbl "
            "and metric=rank/max."),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["count", "sum", "mean", "min", "max", "rank"]},
                "label": {"type": "string", "enum": ["Well", "MetricReading", "Alert", "Intervention"]},
                "attribute": {"type": "string", "description": "numeric attribute, e.g. est_cost_per_bbl, expected_uplift_bbl, weight"},
                "filters": {"type": "object", "description": "attribute->value equality filter ('well' matches by suffix)"},
                "order": {"type": "string", "enum": ["asc", "desc"], "description": "for rank/min/max"},
                "limit": {"type": "integer", "description": "for rank (default 5)"},
                "ratio_num": {"type": "string", "description": "numerator attribute for a derived ratio"},
                "ratio_den": {"type": "string", "description": "denominator attribute for a derived ratio"},
            },
            "required": ["metric", "label"],
        },
    },
]


def schemas_for(names: list[str]) -> list[dict]:
    """Subset of tool schemas by name — used for the -compute_metric ablation."""
    return [s for s in SCHEMAS if s["name"] in names]


def _val(rec: dict, inp: dict) -> float | None:
    num, den = inp.get("ratio_num"), inp.get("ratio_den")
    if num and den:
        try:
            return float(rec[num]) / float(rec[den])
        except (KeyError, TypeError, ZeroDivisionError):
            return None
    attr = inp.get("attribute")
    try:
        return float(rec[attr]) if attr in rec else None
    except (TypeError, ValueError):
        return None


class Tools:
    def __init__(self, kg, semantic):
        self.kg = kg
        self.semantic = semantic

    def dispatch(self, name: str, inp: dict[str, Any]) -> tuple[dict, list[str]]:
        """Return (result_payload, cited_ids)."""
        if name == "cypher_query":
            return self._cypher(inp)
        if name == "semantic_search":
            docs = self.semantic.search(inp["query"], int(inp.get("k", 5)))
            return {"docs": docs}, [d["id"] for d in docs]
        if name == "compute_metric":
            return self._compute(inp)
        return {"error": f"unknown tool {name}"}, []

    def _cypher(self, inp: dict) -> tuple[dict, list[str]]:
        op = inp.get("operation")
        if op == "nodes":
            label = inp.get("label")
            if not label:
                return {"error": "operation=nodes requires 'label' (Well|MetricReading|Alert|Intervention)"}, []
            recs = self.kg.nodes_by_label(label, inp.get("filters"))[:MAX_ROWS]
            return {"records": recs, "count": len(recs)}, [r["id"] for r in recs]
        if op == "neighbours":
            nid = inp.get("node_id", "")
            recs = self.kg.neighbours(nid)
            cited = [nid] + [r["other"] for r in recs] if self.kg.has_node(nid) else []
            return {"node_id": nid, "neighbours": recs}, cited
        if op == "latest":
            if not inp.get("filters"):
                return {"error": "operation=latest requires 'filters' like {metric, well}"}, []
            recs = self.kg.nodes_by_label("MetricReading", inp.get("filters"))
            recs = [r for r in recs if "timestamp" in r]
            if not recs:
                return {"error": "no MetricReading matched (need filters like {metric, well})"}, []
            r = max(recs, key=lambda x: str(x["timestamp"]))
            return {"id": r["id"], "metric": r.get("metric"), "value": r.get("value"),
                    "timestamp": r.get("timestamp"), "well": r.get("well")}, [r["id"]]
        if op == "subgraph":
            sub = self.kg.khop_subgraph(inp.get("seed_ids", []), int(inp.get("hops", 1)))
            sub["nodes"] = sub["nodes"][:MAX_ROWS]
            return sub, [n["id"] for n in sub["nodes"]]
        return {"error": f"unknown operation {op}"}, []

    def _compute(self, inp: dict) -> tuple[dict, list[str]]:
        label, metric = inp.get("label"), inp.get("metric")
        if not label or not metric:
            return {"error": "compute_metric requires 'label' and 'metric'"}, []
        recs = self.kg.nodes_by_label(label, inp.get("filters"))
        if metric == "count":
            return {"metric": "count", "value": len(recs)}, [r["id"] for r in recs]
        scored = [(r, _val(r, inp)) for r in recs]
        scored = [(r, v) for r, v in scored if v is not None]
        if not scored:
            return {"error": "no numeric values for that attribute/filter", "value": None}, []
        if metric == "sum":
            return {"metric": "sum", "value": round(sum(v for _, v in scored), 4)}, [r["id"] for r, _ in scored]
        if metric == "mean":
            return {"metric": "mean", "value": round(sum(v for _, v in scored) / len(scored), 4)}, [r["id"] for r, _ in scored]
        order = inp.get("order", "asc" if metric == "min" else "desc")
        scored.sort(key=lambda rv: rv[1], reverse=(order == "desc"))
        if metric in ("min", "max"):
            r, v = scored[0]
            return {"metric": metric, "value": round(v, 4), "id": r["id"], "well": r.get("well")}, [r["id"]]
        # rank
        limit = int(inp.get("limit", 5))
        top = scored[:limit]
        return {"metric": "rank", "order": order,
                "ranking": [{"id": r["id"], "value": round(v, 4)} for r, v in top]}, [r["id"] for r, _ in top]
