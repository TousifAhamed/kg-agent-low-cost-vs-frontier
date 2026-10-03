"""Serialize the synthetic dashboard outputs into retrievable text documents.

Used by S2 (Text-RAG): the two dashboards are 'dumped as text' (roadmap RQ2) and
retrieved with FAISS + a sentence embedder. Each doc has a stable id so retrieval
is inspectable. This is the deliberately un-structured view the KG agent must beat.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SYN = Path("data/synthetic/v1")


def build_documents() -> list[dict]:
    wells = pd.read_csv(SYN / "wells.csv")
    conn = pd.read_csv(SYN / "connectivity.csv")
    readings = pd.read_csv(SYN / "readings.csv")
    alerts = pd.read_csv(SYN / "alerts.csv")
    interv = pd.read_csv(SYN / "interventions.csv")
    docs: list[dict] = []

    for _, w in wells.iterrows():
        docs.append({"id": w["well_id"],
                     "text": f"Well {w['well_name']} (short {w['short']}) is a {w['well_type']} in field Volve."})

    for _, c in conn.iterrows():
        docs.append({"id": c["edge_id"],
                     "text": f"Connectivity: injector {c['injector']} injects into producer {c['producer']} "
                             f"with CRM weight {c['weight']} (tau {c['tau_days']} days, noise {c['noise_sigma']})."})

    # Latest-per-metric reading summaries per producer (keeps the corpus focused).
    last = (readings.sort_values("date").groupby(["short", "metric"]).last().reset_index())
    for short, g in last.groupby("short"):
        kv = "; ".join(f"{r['metric']}={r['value']} {r['unit']} on {r['date']}" for _, r in g.iterrows())
        docs.append({"id": f"SUM_{short}", "text": f"Producer {short} latest dashboard readings: {kv}."})

    ac = alerts.alert_code.value_counts().to_dict()
    docs.append({"id": "ALERTS_SUMMARY",
                 "text": "Water alert counts: " + ", ".join(f"{k}={v}" for k, v in ac.items()) + "."})
    for _, a in alerts.iterrows():
        docs.append({"id": a["alert_id"],
                     "text": f"Alert {a['alert_code']} ({a['severity']}) on {a['well']} raised {a['raised_at']}, "
                             f"triggered by reading {a['triggered_by']}."})
    for _, iv in interv.iterrows():
        docs.append({"id": iv["intervention_id"],
                     "text": f"Intervention {iv['intervention_id']}: action {iv['action_type']} on {iv['well']}, "
                             f"est cost {iv['est_cost_per_bbl']} USD/bbl, expected uplift "
                             f"{iv['expected_uplift_bbl']} bbl, recommended for {iv['recommended_for']}."})
    return docs
