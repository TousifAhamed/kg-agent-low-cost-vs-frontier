"""Run S4 (the KG agent) over the 50-question benchmark.

For each question the agent produces an answer + validated citations; we grade with
the same graders as the baselines (EM / NDCG@5), and Explanation questions with the
0-3 rubric judge (src/eval/rubric.py). We also record provenance signals unique to
S4: citation validity and #tool calls. Output is the roadmap's agent-run log.

  python -m src.agent.run_agent            # all 50
  python -m src.agent.run_agent --n 3      # smoke test (first 3)
Out: results/agent-runs-v1.jsonl  (+ prints an S4-vs-S2 summary)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.loop import Agent
from src.baselines.llm_only import load_bench
from src.eval.metrics import grade
from src.eval.rubric import grade_explanation

OUT = Path("results/agent-runs-v1.jsonl")


def run(n: int | None = None) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agent = Agent()
    from src.llm import LLM
    judge = LLM(max_tokens=64)  # reuse one judge client for all explanation grades
    bench = load_bench()
    if n:
        bench = bench[:n]
    rows = []
    for q in bench:
        res = agent.answer(q["question"])
        ans = res["answer"]
        if q["answer_type"] == "explanation":
            score = grade_explanation(q["question"], q.get("gold_nodes", []), res["raw"], judge)
            g = {"exact_match": None, "ndcg_at_5": None, "rubric": score}
        else:
            g = {**grade(q, ans), "rubric": None}
        chk = res["citation_check"]
        rows.append({"question_id": q["id"], "system": "S4", "category": q["category"],
                     "answer": ans, "citations": res["citations"],
                     "citation_ok": chk["ok"], "citation_invalid": chk["not_in_kg"],
                     "n_tool_calls": res["n_tool_calls"], **g})
        tag = g.get("exact_match")
        tag = f"em={tag}" if tag is not None else (f"ndcg={g['ndcg_at_5']}" if g["ndcg_at_5"] is not None else f"rubric={g['rubric']}")
        print(f"{q['id']} {q['category'][:4]} {tag} cites={res['citations'][:3]} ok={chk['ok']} tools={res['n_tool_calls']}")
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    _summary(rows)
    print(f"\nS4 done: {len(rows)} answers -> {OUT}")


def _summary(rows: list[dict]) -> None:
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    em = avg([r["exact_match"] for r in rows])
    ndcg = avg([r["ndcg_at_5"] for r in rows])
    rub = avg([r["rubric"] for r in rows])
    cite_ok = avg([1.0 if r["citation_ok"] else 0.0 for r in rows])
    print("\n=== S4 summary ===")
    print(f"EM={em}  NDCG@5={ndcg}  Explanation rubric(0-3)={rub}  citation-valid={cite_ok}")
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    for cat, rs in by.items():
        cem = avg([r["exact_match"] for r in rs])
        print(f"  {cat:15s} EM={cem}  n={len(rs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="run only first N questions (smoke test)")
    args = ap.parse_args()
    run(args.n)


if __name__ == "__main__":
    main()
