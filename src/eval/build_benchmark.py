"""Build a machine-gradeable benchmark of 50 questions grounded on the seed-42 Volve KG.

Every gold answer is COMPUTED from the KG (not hand-typed), so ground truth is exact
and reproducible. Emits:
  * data/benchmark/benchmark.jsonl  — {id, category, question, answer_type, gold, gold_nodes, relevance}
  * docs/questions.md               — human-readable, regenerated from the same source

Category split (roadmap): Identification 13 / Risk 12 / Recommendation 13 / Explanation 12.
answer_type drives grading (src/eval/metrics.py):
  entity_list / scalar / boolean -> exact-match ;  ranking -> NDCG@5 ;  explanation -> rubric (M3+).

Run:  python -m src.eval.build_benchmark
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.kg.query import NetworkXStore

SYN = Path("data/synthetic/v1")
OUT_JSONL = Path("data/benchmark/benchmark.jsonl")
OUT_MD = Path("docs/questions.md")

store = NetworkXStore()
readings = pd.read_csv(SYN / "readings.csv")
alerts = pd.read_csv(SYN / "alerts.csv")
conn = pd.read_csv(SYN / "connectivity.csv")
interv = pd.read_csv(SYN / "interventions.csv")

PRODUCERS = [d["short"] for _, d in store._nodes("Well") if d["well_type"] == "producer"]
INJECTORS = [d["short"] for _, d in store._nodes("Well") if d["well_type"] == "injector"]


def _latest(metric: str, short: str):
    r = readings[(readings.metric == metric) & (readings.short == short)].sort_values("date")
    return r.iloc[-1] if len(r) else None


def _norm(x):
    if isinstance(x, list):
        return sorted(str(i) for i in x)
    if isinstance(x, float):
        return round(x, 4)
    return x


Q: list[dict] = []


def add(cat, q, atype, gold, nodes, relevance=None):
    Q.append({"id": f"Q{len(Q)+1:02d}", "category": cat, "question": q,
              "answer_type": atype, "gold": _norm(gold), "gold_nodes": nodes,
              "relevance": relevance})


# ---------------- Identification (13) ----------------
add("Identification", "Which injectors are connected to producer F-12 via the CRM connectivity fit?",
    "entity_list", store.injectors_of("F-12"), [f"W_{p}" for p in ["F-12"]])
add("Identification", "What is the CRM connectivity weight from injector F-5 to producer F-11?",
    "scalar", float(conn[(conn.injector.str.endswith("F-5")) & (conn.producer.str.endswith("F-11"))].weight.iloc[0]),
    ["C_F-5_F-11"])
add("Identification", "What is the CRM time constant (tau, days) used for injector-producer pairs?",
    "scalar", float(conn.tau_days.iloc[0]), [])
add("Identification", "Which injector has the highest total outgoing connectivity weight across the field?",
    "scalar", "F-" + conn.groupby(conn.injector.str.split("F-").str[-1]).weight.sum().idxmax(), [])
add("Identification", "What is the dominant (highest-weight) supporting injector for producer F-11?",
    "scalar", "F-" + conn[conn.producer.str.endswith("F-11")].sort_values("weight").iloc[-1].injector.split("F-")[-1],
    ["W_F-11"])
add("Identification", "How many injector-producer pairs have a connectivity weight above 0.2?",
    "scalar", int((conn.weight > 0.2).sum()), [])
add("Identification", "What is the most recent log10(WOR) value for producer F-14?",
    "scalar", float(_latest("log_wor", "F-14")["value"]), [_latest("log_wor", "F-14")["reading_id"]])
add("Identification", "Which producers receive injection support from injector F-4?",
    "entity_list", sorted("F-" + p.split("F-")[-1] for p in conn[conn.injector.str.endswith("F-4")].producer),
    ["W_F-4"])
add("Identification", "What is the cumulative oil production (Np) for producer F-12 at the latest timestep?",
    "scalar", store.latest_np("F-12"), [_latest("np_cum", "F-12")["reading_id"]])
add("Identification", "What is the single strongest injector-producer connection in the field?",
    "entity_list", [store.strongest_connection()["injector"], store.strongest_connection()["producer"]], [])
add("Identification", "How many producer wells and how many injector wells are in the field?",
    "scalar", f"{len(PRODUCERS)} producers, {len(INJECTORS)} injectors", [])
add("Identification", "What is the most recent water-cut fraction for producer F-12?",
    "scalar", float(_latest("water_cut", "F-12")["value"]), [_latest("water_cut", "F-12")["reading_id"]])
add("Identification", "On what date was producer F-14's water-cut last measured?",
    "scalar", str(_latest("water_cut", "F-14")["date"]), [_latest("water_cut", "F-14")["reading_id"]])

# ---------------- Risk (12) ----------------
sev_by_well = alerts[alerts.alert_code == "WCT_SEVERE"].well.value_counts()
add("Risk", "Which producers have ever raised a WCT_SEVERE (water-cut >= 90%) alert?",
    "entity_list", sorted("F-" + w.split("F-")[-1] for w in sev_by_well.index), [])
add("Risk", "Which producers experienced a water-breakthrough alert?",
    "entity_list", sorted("F-" + w.split("F-")[-1] for w in
                          alerts[alerts.alert_code == "WATER_BREAKTHROUGH"].well.unique()), [])
add("Risk", "Which injector-producer pairs have connectivity weight above 0.3 (early-breakthrough risk)?",
    "scalar", int((conn.weight > 0.3).sum()), [])
add("Risk", "How many HIGH_WOR alerts were raised across the field?",
    "scalar", int((alerts.alert_code == "HIGH_WOR").sum()), [])
add("Risk", "Which producer has the greatest number of severe water-cut alerts?",
    "scalar", "F-" + sev_by_well.idxmax().split("F-")[-1], [])
add("Risk", "Do any producers depend on only a single injector (support single-point-of-failure)?",
    "boolean", bool(any(len(store.injectors_of(p)) == 1 for p in PRODUCERS)), [])
add("Risk", "How many total alerts were raised in the field?",
    "scalar", int(len(alerts)), [])
add("Risk", "Which producer reached water breakthrough earliest (by alert date)?",
    "scalar", "F-" + alerts[alerts.alert_code == "WATER_BREAKTHROUGH"].sort_values("raised_at").iloc[0].well.split("F-")[-1], [])
add("Risk", "What fraction of producers ever crossed the 50% water-cut breakthrough threshold?",
    "scalar", round(alerts[alerts.alert_code == "WATER_BREAKTHROUGH"].well.nunique() / len(PRODUCERS), 4), [])
add("Risk", "Is producer F-12's connectivity strong enough (>0.2 from any injector) to risk injected-water arrival?",
    "boolean", bool((conn[conn.producer.str.endswith("F-12")].weight > 0.2).any()), [])
add("Risk", "Which producer carries the highest single connectivity weight (largest injected-water exposure)?",
    "scalar", "F-" + conn.sort_values("weight").iloc[-1].producer.split("F-")[-1], [])
add("Risk", "How many distinct producers currently sit above the severe (90%) water-cut level?",
    "scalar", int(sev_by_well.shape[0]), [])

# ---------------- Recommendation (13) ----------------
by_action = interv.action_type.value_counts()
interv["ratio"] = interv.expected_uplift_bbl / interv.est_cost_per_bbl
add("Recommendation", "What is the single cheapest recommended intervention (by cost per barrel)?",
    "scalar", store.cheapest_intervention()["id"], [store.cheapest_intervention()["id"]])
add("Recommendation", "Which intervention action type is recommended most frequently?",
    "scalar", by_action.idxmax(), [])
add("Recommendation", "How many recommended interventions are there in total?",
    "scalar", int(len(interv)), [])
add("Recommendation", "Which intervention has the highest expected oil uplift (bbl)?",
    "scalar", interv.sort_values("expected_uplift_bbl").iloc[-1].intervention_id, [])
add("Recommendation", "What is the mean estimated cost-per-barrel across all interventions?",
    "scalar", round(float(interv.est_cost_per_bbl.mean()), 2), [])
add("Recommendation", "Which intervention offers the best expected-uplift-per-cost ratio?",
    "scalar", interv.sort_values("ratio").iloc[-1].intervention_id, [])
add("Recommendation", "How many interventions are recommended for producer F-14?",
    "scalar", int((interv.well.str.endswith("F-14")).sum()), [])
add("Recommendation", "Which producers have at least one recommended intervention?",
    "entity_list", sorted({"F-" + w.split("F-")[-1] for w in interv.well.unique()}), [])
add("Recommendation", "Rank the 5 cheapest interventions by cost-per-barrel (ascending).",
    "ranking", list(interv.sort_values("est_cost_per_bbl").intervention_id.head(5)),
    list(interv.sort_values("est_cost_per_bbl").intervention_id.head(5)),
    relevance={iid: 3 - min(2, i) for i, iid in
               enumerate(interv.sort_values("est_cost_per_bbl").intervention_id.head(5))})
add("Recommendation", "What action is recommended for the cheapest intervention?",
    "scalar", interv.sort_values("est_cost_per_bbl").iloc[0].action_type, [])
add("Recommendation", "How many interventions use the water_shutoff action?",
    "scalar", int((interv.action_type == "water_shutoff").sum()), [])
add("Recommendation", "For producer F-14, which recommended intervention has the lowest cost per barrel?",
    "scalar", interv[interv.well.str.endswith("F-14")].sort_values("est_cost_per_bbl").iloc[0].intervention_id, [])
add("Recommendation", "Rank the top 5 interventions by expected uplift (descending).",
    "ranking", list(interv.sort_values("expected_uplift_bbl", ascending=False).intervention_id.head(5)),
    list(interv.sort_values("expected_uplift_bbl", ascending=False).intervention_id.head(5)),
    relevance={iid: 3 - min(2, i) for i, iid in
               enumerate(interv.sort_values("expected_uplift_bbl", ascending=False).intervention_id.head(5))})

# ---------------- Explanation (12) ----------------
# These are graded by rubric in M3; gold captures the correct evidence trace (nodes).
first_bt = alerts[alerts.alert_code == "WATER_BREAKTHROUGH"].sort_values("raised_at").iloc[0]
cheap = store.cheapest_intervention()["id"]
for q, nodes in [
    ("Explain why producer F-14 raised a severe water-cut alert: trace alert -> triggering reading -> recommended intervention.",
     list(alerts[alerts.well.str.endswith("F-14")].alert_id.head(1))),
    ("Trace the earliest water-breakthrough alert: which reading triggered it, on what date, and what intervention was recommended?",
     [first_bt["alert_id"], first_bt["triggered_by"]]),
    ("Why is the F-5 -> F-11 pair flagged as an early-breakthrough risk? Explain using its connectivity weight.",
     ["C_F-5_F-11"]),
    ("Explain the provenance chain of the cheapest recommended intervention back to its source record.",
     [cheap]),
    ("What evidence supports a HIGH_WOR alert (which reading and threshold)?",
     list(alerts[alerts.alert_code == "HIGH_WOR"].alert_id.head(1))),
    ("Explain the difference in water-cut trajectory between producers F-12 and F-15 D.",
     ["W_F-12", "W_F-15D"]),
    ("Why does producer F-11 receive more support from injector F-5 than from F-4? Explain using the weights.",
     ["C_F-5_F-11", "C_F-4_F-11"]),
    ("Explain, end to end, the risk-to-recommendation chain for the field's highest-risk producer.",
     [sev_by_well.idxmax()]),
    ("Explain why every KG node carries a source_record_id and why that matters for citation validation.",
     []),
    ("What causes the recommended water_shutoff interventions, in terms of the water diagnostics?",
     list(interv[interv.action_type == "water_shutoff"].intervention_id.head(1))),
    ("Describe the connectivity structure of injector F-4: which producers it supports and the relative weights.",
     ["W_F-4"]),
    ("Summarize how an alert, its triggering reading, and its recommended intervention link together in the ontology.",
     []),
]:
    add("Explanation", q, "explanation", "rubric", nodes)


def main() -> None:
    counts: dict[str, int] = {}
    for q in Q:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    assert counts == {"Identification": 13, "Risk": 12, "Recommendation": 13, "Explanation": 12}, counts
    assert len(Q) == 50

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as fh:
        for q in Q:
            fh.write(json.dumps(q, default=str) + "\n")

    tag = {"Identification": "ID", "Risk": "RI", "Recommendation": "RE", "Explanation": "EX"}
    lines = ["# Benchmark Questions (50) — grounded on the seed-42 Volve KG", "",
             "Machine-gradeable: every gold answer is computed from the KG "
             "(`data/benchmark/benchmark.jsonl`). Regenerate with "
             "`python -m src.eval.build_benchmark`.", "",
             f"Distribution: Identification {counts['Identification']}, Risk {counts['Risk']}, "
             f"Recommendation {counts['Recommendation']}, Explanation {counts['Explanation']}.",
             "Grading: entity_list/scalar/boolean -> exact-match; ranking -> NDCG@5; explanation -> rubric (M3+).", ""]
    cur = None
    for q in Q:
        if q["category"] != cur:
            cur = q["category"]
            lines += ["", f"## {cur}", ""]
        lines.append(f"- `[{tag[cur]}]` **{q['id']}** {q['question']}  ")
        lines.append(f"  <sub>type={q['answer_type']} · gold=`{q['gold']}`</sub>")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(Q)} questions -> {OUT_JSONL} and {OUT_MD}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
