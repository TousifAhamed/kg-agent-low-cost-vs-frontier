"""S3 baseline: KG-RAG retrieval-only (no agent, no tools, no compute).

Isolates the value of *structured retrieval* from the value of *agentic reasoning*:
we entity-link the question, extract a k-hop subgraph, serialize it to text, and make
ONE LLM call to answer. Unlike S4 there is no tool loop, no deterministic aggregation,
and no citation guardrail. The S4 - S3 gap is the roadmap's success-band signal for
"the agent adds value beyond retrieval".

Run:  python -m src.baselines.kg_rag           # all 50
Out:  results/baselines/S3_kg_rag.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.entity_linker import link
from src.agent.retrieval import KG
from src.baselines.llm_only import load_bench
from src.eval.metrics import grade
from src.llm import LLM

OUT = Path("results/baselines/S3_kg_rag.jsonl")
MAX_NODES = 60

SYSTEM = ("You are a petroleum production analyst. Use ONLY the knowledge-graph facts provided "
          "to answer. Be concise: give well ids, the exact number, or the ordered id list as asked. "
          "If the facts are insufficient, say so. End with a line 'ANSWER: <concise answer>'.")

_KEEP = {"label", "short", "well_type", "metric", "value", "timestamp", "well", "alert_code",
         "raised_at", "action_type", "est_cost_per_bbl", "expected_uplift_bbl"}


def serialize(kg: KG, seeds: list[str], hops: int = 2) -> str:
    """Serialize the k-hop neighbourhood. MetricReading time-series are summarized to
    the latest 2 per (well, metric) — realistic retrieval keeps the freshest readings,
    but S3 still has no mechanism to aggregate/rank (that is S4's compute_metric)."""
    if not seeds:  # field-wide question: seed from all wells
        seeds = [n for n, d in kg.g.nodes(data=True) if d.get("label") == "Well"]
    sub = kg.khop_subgraph(seeds, hops)
    readings, others = [], []
    for n in sub["nodes"]:
        (readings if n.get("label") == "MetricReading" else others).append(n)
    # keep the 2 most-recent readings per (well, metric)
    readings.sort(key=lambda n: str(n.get("timestamp", "")), reverse=True)
    kept, seen = [], {}
    for n in readings:
        key = (n.get("well"), n.get("metric"))
        if seen.get(key, 0) < 2:
            seen[key] = seen.get(key, 0) + 1
            kept.append(n)
    lines = []
    for n in (others[:MAX_NODES] + kept):
        attrs = ", ".join(f"{k}={n[k]}" for k in n if k in _KEEP and k != "label")
        lines.append(f"({n['id']} :{n.get('label','?')}) {attrs}")
    for e in sub["edges"]:
        if e["rel"] in ("MEASURED_BY", "onWell"):
            continue  # skip the dense sensor plumbing
        w = f" weight={e['weight']}" if "weight" in e else ""
        lines.append(f"({e['from']}) -[{e['rel']}]-> ({e['to']}){w}")
    return "\n".join(lines)


def run(hops: int = 1) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kg = KG()
    llm = LLM()
    rows = []
    for q in load_bench():
        ctx = serialize(kg, link(q["question"], kg)["seed_nodes"], hops)
        pred = llm.ask(SYSTEM, f"KG facts:\n{ctx}\n\nQuestion: {q['question']}")
        ans = pred.split("ANSWER:")[-1].strip() if "ANSWER:" in pred else pred
        g = grade(q, ans) if q["answer_type"] != "explanation" else {"exact_match": None, "ndcg_at_5": None}
        rows.append({"question_id": q["id"], "system": "S3", "category": q["category"],
                     "answer": ans, **g})
        print(f"{q['id']} {q['category'][:4]} em={g['exact_match']} ndcg={g['ndcg_at_5']}")
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"S3 done: {len(rows)} answers, {llm.calls} calls -> {OUT}")
    return OUT


if __name__ == "__main__":
    run()
