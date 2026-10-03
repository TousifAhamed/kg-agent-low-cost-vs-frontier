"""Offline EM re-grade of every stored run under strict AND normalized graders.

M5 failure analysis found ~53% of S4(Gemini) failure instances were correct answers
failed by strict-EM parsing (prose answers), not model errors. This driver re-scores
every results/**/*.jsonl from the stored `answer` text with both graders, applied
uniformly to every system, so the comparison stays controlled. No API calls.

  python -m src.eval.regrade          # writes tables/em_regrade.csv + prints summary

Sanity: the strict re-score is checked against the stored `exact_match` per row;
mismatches are reported (they would mean the stored numbers are not reproducible).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from src.eval.metrics import exact_match, exact_match_norm

BENCH = Path("data/benchmark/benchmark.jsonl")
RESULT_GLOBS = ["results/baselines/*.jsonl", "results/experiments/*.jsonl",
                "results/gemini/*.jsonl"]
OUT_CSV = Path("tables/em_regrade.csv")


def _bench() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}


def regrade() -> list[dict]:
    bench = _bench()
    rows, mismatches = [], 0
    for pattern in RESULT_GLOBS:
        for fp in sorted(glob.glob(pattern)):
            n = strict = norm = 0
            for line in open(fp, encoding="utf-8"):
                r = json.loads(line)
                q = bench.get(r.get("question_id"))
                if q is None or q["answer_type"] in ("ranking", "explanation"):
                    continue
                s = exact_match(q["answer_type"], q["gold"], r.get("answer") or "")
                m = exact_match_norm(q["answer_type"], q["gold"], r.get("answer") or "",
                                     q["question"])
                if r.get("exact_match") is not None and s != r["exact_match"]:
                    mismatches += 1
                n += 1
                strict += s
                norm += m
            if n:
                rows.append({"file": fp.replace("\\", "/"), "n_em_questions": n,
                             "em_strict": round(strict / n, 4),
                             "em_norm": round(norm / n, 4),
                             "delta": round((norm - strict) / n, 4)})
    if mismatches:
        print(f"WARNING: {mismatches} rows where strict re-score != stored exact_match")
    return rows


def main() -> None:
    rows = regrade()
    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8") as fh:
        fh.write("file,n_em_questions,em_strict,em_norm,delta\n")
        for r in rows:
            fh.write(f"{r['file']},{r['n_em_questions']},{r['em_strict']},"
                     f"{r['em_norm']},{r['delta']}\n")
    w = max(len(r["file"]) for r in rows)
    print(f"{'file':<{w}}  n   strict  norm    delta")
    for r in rows:
        print(f"{r['file']:<{w}}  {r['n_em_questions']:<3} {r['em_strict']:<7} "
              f"{r['em_norm']:<7} {r['delta']:+.4f}")
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
