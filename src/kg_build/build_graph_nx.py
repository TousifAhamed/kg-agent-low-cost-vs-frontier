"""Materialize the KG as a NetworkX MultiDiGraph (portable fallback, no Docker).

Maps the synthetic generator CSVs onto the ontology instances and attaches a PROV
back-pointer (source_record_id, timestamp, generator_seed) to every node and edge.
Persists a pickled MultiDiGraph plus a JSON-LD export for portability.

Run:  python -m src.kg_build.build_graph_nx
Out:  kg/networkx-graph.pkl , kg/graph.jsonld
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

SYN = Path("data/synthetic/v1")
OUT_PKL = Path("kg/networkx-graph.pkl")
OUT_JSONLD = Path("kg/graph.jsonld")


def _prov(row: pd.Series) -> dict:
    return {"source_record_id": row.get("source_record_id", ""),
            "generator_seed": int(row.get("generator_seed", -1))}


def build() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    wells = pd.read_csv(SYN / "wells.csv")
    conn = pd.read_csv(SYN / "connectivity.csv")
    readings = pd.read_csv(SYN / "readings.csv")
    alerts = pd.read_csv(SYN / "alerts.csv")
    interventions = pd.read_csv(SYN / "interventions.csv")

    g.add_node("F_Volve", label="Field", name="Volve")
    for _, w in wells.iterrows():
        g.add_node(w["well_id"], label="Well", name=w["well_name"], short=w["short"],
                   well_type=w["well_type"], **_prov(w))
        # Reservoir hierarchy (single reservoir for Volve v1)
        g.add_edge(w["well_id"], "F_Volve", key="IN_FIELD", label="PRODUCES_FROM")

    # Sensors + MetricReadings
    for _, r in readings.iterrows():
        sid, rid = r["sensor_id"], r["reading_id"]
        if sid not in g:
            g.add_node(sid, label="Sensor", metric_type=r["metric"], unit=r["unit"],
                       well=r["well"])
            g.add_edge(sid, f"W_{r['short']}".replace(" ", ""), key="ON_WELL", label="onWell")
        g.add_node(rid, label="MetricReading", metric=r["metric"], value=float(r["value"]),
                   unit=r["unit"], timestamp=r["date"], well=r["well"], **_prov(r))
        g.add_edge(rid, sid, key="MEASURED_BY", label="MEASURED_BY")

    # Connectivity edges (injector -> producer)
    for _, c in conn.iterrows():
        src = f"W_{c['injector'].split('-', 1)[-1]}".replace(" ", "")
        dst = f"W_{c['producer'].split('-', 1)[-1]}".replace(" ", "")
        g.add_edge(src, dst, key=c["edge_id"], label="INJECTS_INTO",
                   weight=float(c["weight"]), tau_days=float(c["tau_days"]),
                   noise_sigma=float(c["noise_sigma"]), **_prov(c))

    # Alerts, TRIGGERED_BY reading
    for _, a in alerts.iterrows():
        g.add_node(a["alert_id"], label="Alert", alert_code=a["alert_code"],
                   severity=a["severity"], well=a["well"], raised_at=a["raised_at"], **_prov(a))
        if a["triggered_by"] in g:
            g.add_edge(a["alert_id"], a["triggered_by"], key="TRIGGERED_BY", label="TRIGGERED_BY")

    # Interventions, RECOMMENDED_FOR alert, FOLLOWED_BY reading
    for _, iv in interventions.iterrows():
        g.add_node(iv["intervention_id"], label="Intervention", action_type=iv["action_type"],
                   est_cost_per_bbl=float(iv["est_cost_per_bbl"]),
                   expected_uplift_bbl=float(iv["expected_uplift_bbl"]),
                   status=iv["status"], well=iv["well"], **_prov(iv))
        if iv["recommended_for"] in g:
            g.add_edge(iv["intervention_id"], iv["recommended_for"],
                       key="RECOMMENDED_FOR", label="RECOMMENDED_FOR")
        fb = iv.get("followed_by", "")
        if isinstance(fb, str) and fb in g:
            g.add_edge(iv["intervention_id"], fb, key="FOLLOWED_BY", label="FOLLOWED_BY")
    return g


def export_jsonld(g: nx.MultiDiGraph) -> dict:
    nodes = [{"@id": n, **{k: v for k, v in d.items()}} for n, d in g.nodes(data=True)]
    edges = [{"from": u, "to": v, "rel": k, **{kk: vv for kk, vv in d.items()}}
             for u, v, k, d in g.edges(keys=True, data=True)]
    return {"@context": "https://cew1.example.org/ontology/v1", "nodes": nodes, "edges": edges}


def main() -> None:
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    g = build()
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(g, fh)
    OUT_JSONLD.write_text(json.dumps(export_jsonld(g), indent=2, default=str))
    print(f"NetworkX KG: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {OUT_PKL}")
    labels: dict[str, int] = {}
    for _, d in g.nodes(data=True):
        labels[d.get("label", "?")] = labels.get(d.get("label", "?"), 0) + 1
    print("node labels:", labels)


if __name__ == "__main__":
    main()
