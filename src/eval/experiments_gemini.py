"""Gemini-backbone re-run of the full experiment matrix (controlled comparison).

Every generative system (S1, S2, S3, S4 + ablations + RQ3) is re-answered with the
Gemini backbone so all reported numbers share ONE backbone. Results go to results/gemini/
(the Claude-backbone files under results/experiments and results/baselines are left intact
and become a portability footnote).

Robust to free-tier quotas:
  - PER-QUESTION checkpointing: each answer is appended immediately, so a 429/503 loses at
    most one question. Re-running resumes from where it stopped (skips answered ids).
  - Systems run in priority order, so if the daily quota is exhausted the RQ2 headline
    (S1/S2/S3/S4_r1) is completed first.

  python -m src.eval.experiments_gemini --run     # run everything missing, then aggregate
  python -m src.eval.experiments_gemini           # aggregate only
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd

from src.agent.entity_linker import link
from src.agent.loop import Agent
from src.agent.retrieval import KG, SemanticIndex
from src.baselines import llm_only, text_rag, kg_rag
from src.baselines.corpus import build_documents
from src.baselines.llm_only import load_bench
from src.eval.metrics import grade
from src.eval.rubric import grade_explanation

OUT = Path("results/gemini")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["question_id"] for l in open(path, encoding="utf-8") if l.strip()}


def run_system(tag: str, answer_fn, bench, judge=None, grade_rubric: bool = False) -> None:
    """Answer every not-yet-done question and append its graded row immediately.

    answer_fn(q) -> dict with at least {"answer"}; agents also return
    {"raw","citation_ok","citation_invalid","n_tool_calls"}.
    """
    out = OUT / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = [q for q in bench if q["id"] not in done]
    if not todo:
        print(f"[skip] {tag} (all {len(bench)} done)")
        return
    print(f"[run ] {tag} ({len(todo)} of {len(bench)} remaining)")
    with open(out, "a", encoding="utf-8") as fh:
        for q in todo:
            res = answer_fn(q)
            if q["answer_type"] == "explanation":
                rub = (grade_explanation(q["question"], q.get("gold_nodes", []),
                                         res.get("raw", res["answer"]), judge)
                       if grade_rubric and judge else None)
                g = {"exact_match": None, "ndcg_at_5": None, "rubric": rub}
            else:
                g = {**grade(q, res["answer"]), "rubric": None}
            row = {"question_id": q["id"], "system": tag, "category": q["category"],
                   "answer": res["answer"], "citation_ok": res.get("citation_ok"),
                   "citation_invalid": res.get("citation_invalid"),
                   "n_tool_calls": res.get("n_tool_calls"), **g}
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"  {q['id']} em={g['exact_match']} ndcg={g['ndcg_at_5']} "
                  f"cite_ok={row['citation_ok']}")


def run_all() -> None:
    from src.llm import LLM
    bench = load_bench()
    judge = LLM(max_tokens=64)
    kg = KG()
    sem = SemanticIndex("minilm")            # local embeddings; shared across agents
    full = dict(kg=kg, semantic=sem)

    # ---- baseline answer functions (reuse the existing prompts/retrieval) ----
    llm = LLM()

    def s1(q):
        pred = llm.ask(llm_only.SYSTEM, q["question"])
        return {"answer": pred.split("ANSWER:")[-1].strip() if "ANSWER:" in pred else pred}

    docs = build_documents()
    model = text_rag.SentenceTransformer(text_rag.EMBEDDERS["minilm"])
    index = text_rag.build_index(model, docs)

    def s2(q):
        qv = model.encode([q["question"]], normalize_embeddings=True).astype("float32")
        _, idx = index.search(qv, 10)
        ctx = "\n".join(f"[{docs[i]['id']}] {docs[i]['text']}" for i in idx[0])
        pred = llm.ask(text_rag.SYSTEM, f"Context:\n{ctx}\n\nQuestion: {q['question']}")
        return {"answer": pred.split("ANSWER:")[-1].strip() if "ANSWER:" in pred else pred}

    def s3(q):
        ctx = kg_rag.serialize(kg, link(q["question"], kg)["seed_nodes"], hops=1)
        pred = llm.ask(kg_rag.SYSTEM, f"KG facts:\n{ctx}\n\nQuestion: {q['question']}")
        return {"answer": pred.split("ANSWER:")[-1].strip() if "ANSWER:" in pred else pred}

    def agent_fn(agent):
        def f(q):
            r = agent.answer(q["question"])
            return {"answer": r["answer"], "raw": r["raw"],
                    "citation_ok": r["citation_check"]["ok"],
                    "citation_invalid": r["citation_check"]["not_in_kg"],
                    "n_tool_calls": r["n_tool_calls"]}
        return f

    # ---- priority order: RQ2 headline first, then ablations, then RQ3 ----
    run_system("S1", s1, bench)
    run_system("S2_minilm_k10", s2, bench)
    run_system("S3_kg_rag", s3, bench)

    for r in (1, 2, 3):
        run_system(f"S4_r{r}", agent_fn(Agent(**full)), bench, judge, grade_rubric=True)

    run_system("S4_no_compute",
               agent_fn(Agent(tools=["cypher_query", "semantic_search"], **full)),
               bench, judge, grade_rubric=True)
    run_system("S4_no_guardrail", agent_fn(Agent(use_guardrail=False, **full)),
               bench, judge, grade_rubric=True)

    from src.agent.perturb import edge_dropout, weight_noise
    for p in (0.3, 0.6):
        run_system(f"RQ3_dropout{int(p*100)}",
                   agent_fn(Agent(kg=edge_dropout(kg, p, seed=42), semantic=sem)), bench)
    for s in (0.3, 0.6):
        run_system(f"RQ3_noise{int(s*100)}",
                   agent_fn(Agent(kg=weight_noise(kg, s, seed=42), semantic=sem)), bench)


# ---------------- aggregation ----------------
def _agg_one(df: pd.DataFrame) -> dict:
    def m(col):
        s = df[col].dropna() if col in df else pd.Series(dtype=float)
        return round(s.mean(), 4) if len(s) else None
    cit = df["citation_ok"].dropna() if "citation_ok" in df else pd.Series(dtype=float)
    return {"EM": m("exact_match"), "NDCG@5": m("ndcg_at_5"), "rubric": m("rubric"),
            "citation_valid": round(cit.mean(), 4) if len(cit) else None,
            "n": df["question_id"].nunique()}


def aggregate() -> None:
    rows = []
    for fp in glob.glob(str(OUT / "*.jsonl")):
        for l in open(fp, encoding="utf-8"):
            if l.strip():
                rows.append(json.loads(l))
    if not rows:
        print("no gemini results yet"); return
    df = pd.DataFrame(rows)
    per = {sys: _agg_one(g) for sys, g in df.groupby("system")}

    reps = {k: v for k, v in per.items() if k.startswith("S4_r")}
    for metric in ("EM", "NDCG@5"):
        vals = [v[metric] for v in reps.values() if v[metric] is not None]
        if vals:
            mean = sum(vals) / len(vals)
            se = (_pstdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            print(f"S4 (full, {len(vals)} repeats) {metric}: mean={mean:.4f} +- {se:.4f} SE  "
                  f"(runs={[round(x,3) for x in vals]})")

    order = ["S1", "S2_minilm_k10", "S3_kg_rag", "S4_r1", "S4_r2", "S4_r3",
             "S4_no_compute", "S4_no_guardrail",
             "RQ3_dropout30", "RQ3_dropout60", "RQ3_noise30", "RQ3_noise60"]
    out = pd.DataFrame(per).T.reindex([o for o in order if o in per])
    Path("tables").mkdir(exist_ok=True)
    out.to_csv("tables/m4_summary_gemini.csv")
    print("\n" + out.to_string())
    print("\n-> tables/m4_summary_gemini.csv")


def _pstdev(vals):
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
