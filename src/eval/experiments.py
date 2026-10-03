"""M4 experiment matrix: ablations, S3, RQ3 robustness, and 3-seed statistics.

Resumable: each system writes results/experiments/<tag>.jsonl and is SKIPPED if that
file already exists (protects the API budget; delete a file to re-run it).

Systems
  S4_r1..r3         full agent, 3 repeats (mean +- SE)         [RQ2 headline]
  S4_no_compute     agent WITHOUT compute_metric              [RQ2 ablation]
  S4_no_guardrail   agent WITHOUT citation guardrail          [RQ2 ablation]
  RQ3_dropoutXX     full agent on edge/reading-dropout KG     [RQ3 robustness]
  RQ3_noiseXX       full agent on weight-noise KG             [RQ3 robustness]
S2 (3 repeats) and S3 are run by their own modules / here.

  python -m src.eval.experiments --run       # run everything missing, then aggregate
  python -m src.eval.experiments             # aggregate only
Aggregates -> tables/m4_summary.csv  (EM/NDCG/rubric/citation, with SE across repeats).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd

from src.agent.loop import Agent
from src.agent.retrieval import KG, SemanticIndex
from src.baselines.llm_only import load_bench
from src.eval.metrics import grade
from src.eval.rubric import grade_explanation

EXP = Path("results/experiments")


def run_agent_system(tag: str, agent_factory, judge, bench) -> None:
    out = EXP / f"{tag}.jsonl"
    if out.exists():
        print(f"[skip] {tag} (exists)")
        return
    print(f"[run ] {tag}")
    agent = agent_factory()
    rows = []
    for q in bench:
        res = agent.answer(q["question"])
        if q["answer_type"] == "explanation":
            g = {"exact_match": None, "ndcg_at_5": None,
                 "rubric": grade_explanation(q["question"], q.get("gold_nodes", []), res["raw"], judge)}
        else:
            g = {**grade(q, res["answer"]), "rubric": None}
        rows.append({"question_id": q["id"], "system": tag, "category": q["category"],
                     "answer": res["answer"], "citation_ok": res["citation_check"]["ok"],
                     "citation_invalid": res["citation_check"]["not_in_kg"],
                     "n_tool_calls": res["n_tool_calls"], **g})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"       -> {out} ({len(rows)} rows)")


def run_all() -> None:
    from src.llm import LLM
    bench = load_bench()
    judge = LLM(max_tokens=64)
    kg = KG()
    sem = SemanticIndex("minilm")           # build once, share across every agent
    full = dict(kg=kg, semantic=sem)

    # RQ2 headline: 3 repeats of the full agent (r1 == the M3 run, copied in from agent-runs-v1)
    _seed_r1_from_m3()
    for r in (2, 3):
        run_agent_system(f"S4_r{r}", lambda: Agent(**full), judge, bench)

    # RQ2 ablations
    run_agent_system("S4_no_compute",
                     lambda: Agent(tools=["cypher_query", "semantic_search"], **full), judge, bench)
    run_agent_system("S4_no_guardrail", lambda: Agent(use_guardrail=False, **full), judge, bench)

    # RQ3 robustness
    from src.agent.perturb import edge_dropout, weight_noise
    for p in (0.3, 0.6):
        run_agent_system(f"RQ3_dropout{int(p*100)}",
                         lambda p=p: Agent(kg=edge_dropout(kg, p, seed=42), semantic=sem), judge, bench)
    for s in (0.3, 0.6):
        run_agent_system(f"RQ3_noise{int(s*100)}",
                         lambda s=s: Agent(kg=weight_noise(kg, s, seed=42), semantic=sem), judge, bench)


def _seed_r1_from_m3() -> None:
    """Reuse the M3 full-agent run as repeat #1 (identical config) to save 50 calls."""
    dst = EXP / "S4_r1.jsonl"
    src = Path("results/agent-runs-v1.jsonl")
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    for r in rows:
        r["system"] = "S4_r1"
    with open(dst, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print("[seed] S4_r1 <- results/agent-runs-v1.jsonl")


# ---------------- aggregation ----------------
def _load() -> pd.DataFrame:
    rows = []
    for fp in glob.glob(str(EXP / "*.jsonl")):
        for l in open(fp, encoding="utf-8"):
            rows.append(json.loads(l))
    # include the M2 baselines for context
    for fp in ["results/baselines/S1_llm_only.jsonl",
               "results/baselines/S2_text_rag_minilm_k10.jsonl",
               "results/baselines/S3_kg_rag.jsonl"]:
        if Path(fp).exists():
            for l in open(fp, encoding="utf-8"):
                r = json.loads(l)
                r["system"] = {"S1": "S1", "S2": "S2_minilm_k10", "S3": "S3_kg_rag"}.get(r["system"], r["system"])
                rows.append(r)
    return pd.DataFrame(rows)


def _agg_one(df: pd.DataFrame) -> dict:
    def m(col):
        s = df[col].dropna() if col in df else pd.Series(dtype=float)
        return round(s.mean(), 4) if len(s) else None
    return {"EM": m("exact_match"), "NDCG@5": m("ndcg_at_5"), "rubric": m("rubric"),
            "citation_valid": round(df["citation_ok"].mean(), 4) if "citation_ok" in df and df["citation_ok"].notna().any() else None,
            "n": df["question_id"].nunique()}


def aggregate() -> None:
    df = _load()
    if df.empty:
        print("no results yet"); return
    per = {sys: _agg_one(g) for sys, g in df.groupby("system")}

    # collapse S4_r1..r3 into a mean +- SE row
    reps = {k: v for k, v in per.items() if k.startswith("S4_r")}
    if reps:
        for metric in ("EM", "NDCG@5"):
            vals = [v[metric] for v in reps.values() if v[metric] is not None]
            if vals:
                mean = sum(vals) / len(vals)
                se = (statistics_pstdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                print(f"S4 (full, {len(vals)} repeats) {metric}: mean={mean:.4f} +- {se:.4f} SE  (runs={[round(x,3) for x in vals]})")

    out = pd.DataFrame(per).T
    order = ["S1", "S2_minilm_k10", "S3_kg_rag", "S4_r1", "S4_r2", "S4_r3",
             "S4_no_compute", "S4_no_guardrail",
             "RQ3_dropout30", "RQ3_dropout60", "RQ3_noise30", "RQ3_noise60"]
    out = out.reindex([o for o in order if o in out.index])
    Path("tables").mkdir(exist_ok=True)
    out.to_csv("tables/m4_summary.csv")
    print("\n" + out.to_string())
    print("\n-> tables/m4_summary.csv")


def statistics_pstdev(vals):
    n = len(vals); mu = sum(vals) / n
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        run_all()
    aggregate()


if __name__ == "__main__":
    main()
