"""Run the M2 baselines and aggregate results into the roadmap's CSV.

  python -m src.eval.run_all --run              # run S1 + S2(minilm,k=5), then aggregate
  python -m src.eval.run_all --run --sweep      # also sweep S2 over {minilm,bge} x {3,5,10}
  python -m src.eval.run_all                    # aggregate whatever jsonl already exist

Aggregates every results/baselines/*.jsonl into:
  results/baselines/baseline_results.csv   cols {question_id, system, answer, exact_match, ndcg_at_5}
  results/baselines/summary.csv            EM / NDCG@5 by system (+ best S2 config)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

OUTDIR = Path("results/baselines")


def _run_baselines(sweep: bool) -> None:
    from src.baselines import llm_only, text_rag
    llm_only.run()
    if sweep:
        for emb in ("minilm", "bge"):
            for k in (3, 5, 10):
                text_rag.run(emb, k)
    else:
        text_rag.run("minilm", 5)


def aggregate() -> pd.DataFrame:
    rows = []
    for fp in glob.glob(str(OUTDIR / "*.jsonl")):
        tag = Path(fp).stem
        for line in open(fp):
            r = json.loads(line)
            sysname = r["system"] if r["system"] == "S1" else f"S2:{r.get('embedder','?')}k{r.get('topk','?')}"
            rows.append({"question_id": r["question_id"], "system": sysname,
                         "category": r["category"], "answer": r["answer"],
                         "exact_match": r.get("exact_match"), "ndcg_at_5": r.get("ndcg_at_5"),
                         "source_file": tag})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No result jsonl found yet. Run with --run once API credits are available.")
        return df
    df.to_csv(OUTDIR / "baseline_results.csv", index=False)

    summ = (df.groupby("system")
              .agg(EM=("exact_match", lambda s: round(s.dropna().mean(), 4) if s.notna().any() else None),
                   NDCG_at_5=("ndcg_at_5", lambda s: round(s.dropna().mean(), 4) if s.notna().any() else None),
                   n=("question_id", "count"))
              .reset_index())
    summ.to_csv(OUTDIR / "summary.csv", index=False)
    print(summ.to_string(index=False))
    # RQ2 note: best S2 config is the one to report against the future S4 agent.
    s2 = summ[summ.system.str.startswith("S2")]
    if not s2.empty and s2["EM"].notna().any():
        best = s2.loc[s2["EM"].idxmax()]
        print(f"\nBest S2 config by EM: {best['system']} (EM={best['EM']}, NDCG@5={best['NDCG_at_5']})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute baselines (needs ANTHROPIC credits)")
    ap.add_argument("--sweep", action="store_true", help="sweep S2 embedder x top_k")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.run:
        _run_baselines(args.sweep)
    aggregate()


if __name__ == "__main__":
    main()
