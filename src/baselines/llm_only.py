"""S1 baseline: LLM-only (no context).

The backbone answers each benchmark question with NO access to the KG or dashboards.
Establishes the floor: how far pure parametric knowledge gets on field-specific,
grounded questions. Expected to be weak on Identification (it cannot know Volve's
seed-42 numbers) — that gap is the whole point of RQ2.

Run:  python -m src.baselines.llm_only
Out:  results/baselines/S1_llm_only.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.metrics import grade
from src.llm import LLM

BENCH = Path("data/benchmark/benchmark.jsonl")
OUT = Path("results/baselines/S1_llm_only.jsonl")

SYSTEM = ("You are a petroleum production analyst. Answer the question as concisely as possible. "
          "If the question asks for wells, list their identifiers. If it asks for a number, give the number. "
          "If asked to rank, list the identifiers in order. If you do not have the data, give your best estimate. "
          "End with a line 'ANSWER: <concise answer>'.")


def load_bench() -> list[dict]:
    return [json.loads(l) for l in open(BENCH)]


def run() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM()
    rows = []
    for q in load_bench():
        pred = llm.ask(SYSTEM, q["question"])
        ans = pred.split("ANSWER:")[-1].strip() if "ANSWER:" in pred else pred
        g = grade(q, ans)
        rows.append({"question_id": q["id"], "system": "S1", "category": q["category"],
                     "answer": ans, **g})
        print(f"{q['id']} {q['category'][:4]} em={g['exact_match']} ndcg={g['ndcg_at_5']}")
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"S1 done: {len(rows)} answers, {llm.calls} LLM calls -> {OUT}")


if __name__ == "__main__":
    run()
