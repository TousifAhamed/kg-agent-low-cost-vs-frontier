"""Backend-agnostic KG query layer.

Two stores expose the SAME high-level query API so the 10 sanity queries (and the
M3 agent) run identically on:
  * NetworkXStore  — the portable pickled MultiDiGraph (default, no Docker)
  * Neo4jStore     — Neo4j Community via Docker, with PARAMETER-BOUND Cypher only
                     (no string interpolation -> injection-safe for M3 tool use)

Every query returns plain Python (ints / lists of dicts) so results are pinnable
in tests/test_sanity_queries.py against the seed-42 generator output.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import networkx as nx

PKL = Path("kg/networkx-graph.pkl")


class NetworkXStore:
    def __init__(self, path: Path = PKL):
        # SAFE: kg/networkx-graph.pkl is generated locally by src/kg_build/build_graph_nx.py
        # from local CSVs (roadmap mandates the pickled-MultiDiGraph format). Not untrusted input.
        with open(path, "rb") as fh:
            self.g: nx.MultiDiGraph = pickle.load(fh)

    def _nodes(self, label: str):
        return [(n, d) for n, d in self.g.nodes(data=True) if d.get("label") == label]

    # --- sanity / agent query API ---
    def count_wells_by_type(self) -> dict:
        out: dict[str, int] = {}
        for _, d in self._nodes("Well"):
            out[d["well_type"]] = out.get(d["well_type"], 0) + 1
        return dict(sorted(out.items()))

    def injectors_of(self, producer_short: str) -> list[str]:
        dst = f"W_{producer_short}"
        res = []
        for u, v, d in self.g.in_edges(dst, data=True):
            if d.get("label") == "INJECTS_INTO":
                res.append(self.g.nodes[u]["short"])
        return sorted(res)

    def strongest_connection(self) -> dict:
        best = None
        for u, v, d in self.g.edges(data=True):
            if d.get("label") == "INJECTS_INTO":
                if best is None or d["weight"] > best["weight"]:
                    best = {"injector": self.g.nodes[u]["short"],
                            "producer": self.g.nodes[v]["short"], "weight": d["weight"]}
        return best

    def count_alerts_by_code(self) -> dict:
        out: dict[str, int] = {}
        for _, d in self._nodes("Alert"):
            out[d["alert_code"]] = out.get(d["alert_code"], 0) + 1
        return dict(sorted(out.items()))

    def count_interventions(self) -> int:
        return len(self._nodes("Intervention"))

    def cheapest_intervention(self) -> dict:
        best = None
        for n, d in self._nodes("Intervention"):
            if best is None or d["est_cost_per_bbl"] < best["est_cost_per_bbl"]:
                best = {"id": n, "action_type": d["action_type"],
                        "est_cost_per_bbl": d["est_cost_per_bbl"]}
        return best

    def latest_np(self, producer_short: str) -> float:
        vals = [(d["timestamp"], d["value"]) for _, d in self._nodes("MetricReading")
                if d.get("metric") == "np_cum" and d.get("well", "").endswith(producer_short)]
        return round(max(vals)[1], 2) if vals else 0.0

    def alert_trace(self, alert_id: str) -> dict:
        """alert -> triggering reading -> recommended intervention (explanation backbone)."""
        reading = [v for _, v, d in self.g.out_edges(alert_id, data=True)
                   if d.get("label") == "TRIGGERED_BY"]
        interv = [u for u, _, d in self.g.in_edges(alert_id, data=True)
                  if d.get("label") == "RECOMMENDED_FOR"]
        return {"alert": alert_id, "triggered_by": reading, "interventions": interv}

    def provenance_coverage(self) -> float:
        labels = {"Well", "MetricReading", "Alert", "Intervention"}
        ns = [d for _, d in self.g.nodes(data=True) if d.get("label") in labels]
        with_prov = sum(1 for d in ns if d.get("source_record_id"))
        return round(with_prov / len(ns), 4) if ns else 0.0

    def node(self, node_id: str) -> dict | None:
        return dict(self.g.nodes[node_id]) if node_id in self.g else None


class Neo4jStore:
    """Parameter-bound Cypher wrapper. Never interpolates user text into queries."""

    def __init__(self, uri: str | None = None, user: str = "neo4j", password: str | None = None):
        from neo4j import GraphDatabase
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        password = password or os.environ.get("NEO4J_PASSWORD", "cew1password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def run(self, cypher: str, **params):
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    def count_wells_by_type(self) -> dict:
        rows = self.run("MATCH (w:Well) RETURN w.well_type AS t, count(*) AS c ORDER BY t")
        return {r["t"]: r["c"] for r in rows}

    def injectors_of(self, producer_short: str) -> list[str]:
        rows = self.run(
            "MATCH (i:Well)-[:INJECTS_INTO]->(p:Well {short:$s}) RETURN i.short AS s ORDER BY s",
            s=producer_short)
        return [r["s"] for r in rows]

    def provenance_coverage(self) -> float:
        rows = self.run(
            "MATCH (n) WHERE n:Well OR n:MetricReading OR n:Alert OR n:Intervention "
            "RETURN sum(CASE WHEN n.source_record_id IS NOT NULL THEN 1 ELSE 0 END)*1.0/count(n) AS f")
        return round(rows[0]["f"], 4) if rows else 0.0

    def close(self):
        self.driver.close()


def get_store():
    """NetworkX by default; Neo4j only if NEO4J_URI is set and reachable."""
    if os.environ.get("NEO4J_URI"):
        try:
            return Neo4jStore()
        except Exception:
            pass
    return NetworkXStore()
