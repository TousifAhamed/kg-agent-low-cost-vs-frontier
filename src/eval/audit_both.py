"""Symmetric instance-level audit of the full agent's EM answers, both backbones.

Camera-ready response to reviewers: the original failure audit covered only the small
backbone's strict-EM failures, and the normalized grader's four rules were written after
that audit. Here every one of the 216 EM-graded instances (36 items x 3 repeats x 2
backbones) gets a verdict judged against the question *as worded*, starting from the
normalized grader and overriding it only where inspection of the stored answer and the
KG shows the grader is wrong. Every override is listed below with its reason.

Run from the repo root (no API calls):
    python -m src.eval.audit_both
Writes results/audit/em_audit_both_backbones.csv and prints scores plus a paired
item-level bootstrap of the backbone difference.
"""
from __future__ import annotations

import csv
import glob
import json
import random
from pathlib import Path

from src.eval.metrics import exact_match, exact_match_norm

BENCH = Path("data/benchmark/benchmark.jsonl")
OUT = Path("results/audit/em_audit_both_backbones.csv")
RUNS = {"small": "results/gemini/S4_r*.jsonl", "large": "results/experiments/S4_r*.jsonl"}

# Items whose gold answer or wording is defective (verified against the KG):
#   Q16  asks "which pairs" but gold is the count 2.
#   Q19  gold False comes from a generator bug: injectors_of('F-1 C') and
#        injectors_of('F-15 D') return [] (space in the well name), so the
#        single-injector check never fired. In the KG F-1 C is fed only by F-4 and
#        F-15 D only by F-5, so the correct answer is True.
#   Q21  "earliest breakthrough" is a tie (F-12 and F-14, both 2010-06-01); gold keeps F-12.
#   Q25  asks how many producers "currently" exceed 90% water cut, but gold (2) counts
#        producers that ever raised a WCT_SEVERE alert; at the latest reading only
#        F-14 (0.9662) is above 0.90 (F-12 is 0.8835), so the answer as worded is 1.
DEFECTIVE_EM = {"Q16", "Q19", "Q21", "Q25"}

# Reasons for instances the normalized grader accepts but strict rejects (all verified correct).
FLIP_REASON = {
    "Q01": "artifact: well id from the question echoed in prose",
    "Q02": "artifact: strict first-number picked a digit from a well id",
    "Q07": "artifact: strict first-number picked a digit from 'log10'/'F-14'",
    "Q09": "artifact: strict first-number picked '12' (F-12) or the date",
    "Q12": "artifact: strict first-number picked '12' from 'F-12'",
    "Q13": "artifact: prose date ('July 1, 2016')",
    "Q22": "artifact: strict first-number picked '5' before the stated fraction 1.0",
    "Q35": "artifact: 'water shutoff' vs gold token 'water_shutoff'",
}


def verdict(backbone: str, qid: str, answer: str, norm: int) -> tuple[int, str]:
    """Audited correctness against the question as worded, with a reason."""
    a = answer.replace(" ", "")
    if qid == "Q16":   # both pairs above 0.3 are F-4->F-11 and F-5->F-11
        ok = "F-11" in answer and "F-4" in answer and "F-5" in answer
        return int(ok), "gold defect: lists the two correct pairs; gold is a bare count"
    if qid == "Q19":   # correct answer is True
        ok = answer.strip().lower().startswith("yes")
        return int(ok), "gold defect (generator bug): correct answer is True"
    if qid == "Q25":   # as worded ("currently"), the answer is 1
        ok = answer.strip().startswith("1")
        return int(ok), ("question defect: 'currently' -> 1; " +
                         ("answered 1 (correct as worded)" if ok else "answered 2 (matches gold, not the question)"))
    if qid == "Q22" and not norm and ("100%" in a or "5/5" in a):
        return 1, "artifact NOT covered by the normalized grader: fraction given as 5/5 = 100%"
    if qid == "Q33":
        return norm, "genuine: lists 4 of 5 producers, omits F-15 D"
    if qid == "Q04" and not norm:
        return 0, "genuine: aggregation error, ranks F-5 (0.8821) above F-4 (0.9186)"
    if qid == "Q22" and not norm:
        return 0, "genuine: counts 4 of 5 producers"
    if qid == "Q21":
        return norm, "gold defect (tie): answer correctly reports the F-12/F-14 tie"
    return norm, ""


def load() -> list[dict]:
    bench = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}
    rows = []
    for backbone, pat in RUNS.items():
        for fp in sorted(glob.glob(pat)):
            rep = Path(fp).stem.split("_r")[-1]
            for line in open(fp, encoding="utf-8"):
                r = json.loads(line)
                q = bench.get(r.get("question_id"))
                if q is None or q["answer_type"] in ("ranking", "explanation"):
                    continue
                ans = r.get("answer") or ""
                s = exact_match(q["answer_type"], q["gold"], ans)
                n = exact_match_norm(q["answer_type"], q["gold"], ans, q["question"])
                v, why = verdict(backbone, q["id"], ans, n)
                if not why and n and not s:
                    why = FLIP_REASON.get(q["id"], "artifact")
                rows.append({"backbone": backbone, "repeat": rep, "qid": q["id"],
                             "category": q["category"], "strict": s, "norm": n,
                             "audited": v, "reason": why, "gold": json.dumps(q["gold"]),
                             "answer": ans.replace("\n", " ")})
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def item_means(rows, backbone, col, items):
    return {q: mean([r[col] for r in rows if r["backbone"] == backbone and r["qid"] == q])
            for q in items}


def paired_bootstrap(rows, col, items, n_boot=10000, seed=0):
    s = item_means(rows, "small", col, items)
    l = item_means(rows, "large", col, items)
    d = [s[q] - l[q] for q in items]
    rng = random.Random(seed)
    boots = sorted(mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    better_s = sum(x > 0 for x in d)
    better_l = sum(x < 0 for x in d)
    return mean(d), lo, hi, better_s, better_l


def main() -> None:
    rows = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    all_items = sorted({r["qid"] for r in rows})
    clean = [q for q in all_items if q not in DEFECTIVE_EM]
    for label, items in (("all 36 EM items", all_items), (f"{len(clean)} clean items", clean)):
        print(f"\n== {label} ==")
        for b in ("small", "large"):
            sub = [r for r in rows if r["backbone"] == b and r["qid"] in items]
            reps = " / ".join(f"{mean([r['audited'] for r in sub if r['repeat']==k]):.3f}"
                              for k in ("1", "2", "3"))
            print(f"  {b:5s}: strict {mean([r['strict'] for r in sub]):.3f}  "
                  f"norm {mean([r['norm'] for r in sub]):.3f}  "
                  f"audited {mean([r['audited'] for r in sub]):.3f}  (audited per repeat {reps})")
        for col in ("norm", "audited"):
            md, lo, hi, bs, bl = paired_bootstrap(rows, col, items)
            print(f"  paired small-large [{col}]: {md:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"items small>large {bs}, large>small {bl}")

    print("\n== strict-EM failures, by audited cause ==")
    for b in ("small", "large"):
        fails = [r for r in rows if r["backbone"] == b and not r["strict"]]
        buckets: dict[str, int] = {}
        for r in fails:
            key = r["reason"].split(":")[0] if r["reason"] else "?"
            buckets[key] = buckets.get(key, 0) + 1
        correct = sum(r["audited"] for r in fails)
        print(f"  {b}: {len(fails)} strict failures, {correct} correct on audit; {buckets}")
    false_credit = [r for r in rows if r["norm"] and not r["audited"]]
    print("\n== graded correct by normalized EM but wrong on audit ==")
    for r in false_credit:
        print(f"  {r['backbone']} r{r['repeat']} {r['qid']}: {r['reason']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
