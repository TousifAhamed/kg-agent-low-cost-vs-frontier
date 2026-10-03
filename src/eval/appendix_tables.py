"""Generate the camera-ready appendix tables straight from the data (no hand-typed numbers):
  A  the full 50-item benchmark (question, gold, grader, evidence class, defect flag)
  C  item-level audit for every item whose audited verdict differs from 'all correct'
  D  explanation-rubric scores per item, both backbones

    python -m src.eval.appendix_tables   # writes manuscript/generated/*.tex
Requires results/audit/em_audit_both_backbones.csv (python -m src.eval.audit_both).
"""
from __future__ import annotations

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

BENCH = Path("data/benchmark/benchmark.jsonl")
AUDIT = Path("results/audit/em_audit_both_backbones.csv")
OUT = Path("manuscript/generated")

# Evidence each EM item's gold rests on (see src/agent/perturb.py for what is perturbed).
WEIGHT = {"Q02", "Q04", "Q05", "Q06", "Q10", "Q16", "Q23", "Q24"}   # INJECTS_INTO weights
EDGE = {"Q01", "Q03", "Q08", "Q19"}                                  # INJECTS_INTO edges
READ = {"Q07", "Q09", "Q12", "Q13"}                                  # MetricReading nodes
DEFECT = {"Q16": "a", "Q19": "b", "Q21": "c", "Q25": "d", "Q39": "e", "Q40": "c", "Q43": "e"}
# Short reasons for the item-level audit table (Appendix C).
TABLE_REASON = {
    "Q01": "question-echo artifact", "Q02": "first-number artifact", "Q07": "first-number artifact",
    "Q09": "first-number artifact", "Q12": "first-number artifact", "Q13": "prose-date artifact",
    "Q35": "underscore artifact", "Q04": "genuine aggregation error (small r2)",
    "Q16": "gold defect (a)", "Q19": "gold defect (b)", "Q25": "question defect (d)",
    "Q22": "first-number artifact (large r1); 5/5 = 100% missed by normalized grader (large r2, r3); "
           "genuine 4-of-5 count (small r2)",
    "Q33": "genuine: omits F-15 D (both backbones)",
}
CAT = {"Identification": "Id", "Risk": "Risk", "Recommendation": "Rec", "Explanation": "Expl"}
GRADER = {"ranking": "NDCG", "explanation": "Rubric"}


def tex(s: str) -> str:
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    s = s.replace("->", r"$\rightarrow$").replace(">=", r"$\geq$")
    s = s.replace(">", r"$>$").replace("<", r"$<$")
    s = s.replace("_", r"\_\allowbreak{}")
    return s


def ident(s: str) -> str:
    """Monospace identifier that may break after underscores."""
    return r"\texttt{" + tex(s) + "}"


def gold_cell(q: dict) -> str:
    g, t = q["gold"], q["answer_type"]
    if t == "explanation":
        nodes = ", ".join(ident(n) for n in q.get("gold_nodes") or [])
        return "rubric" + (f"; evidence {nodes}" if nodes else "")
    if isinstance(g, list):
        if any("_" in str(x) for x in g):          # ranked id lists: one id per line
            return r"\newline ".join(f"{i}. " + ident(x) for i, x in enumerate(g, 1))
        return ", ".join(tex(x) for x in g)
    if isinstance(g, bool):
        return "true" if g else "false"
    return ident(g) if "_" in str(g) else tex(g)


def evidence(qid: str, t: str) -> str:
    if t in ("ranking", "explanation"):
        return "--"
    if qid in WEIGHT:
        return "W"
    if qid in EDGE:
        return "E"
    if qid in READ:
        return "R"
    return "A"


def table_a(bench: list[dict]) -> str:
    rows = []
    for q in bench:
        flag = f"$^{{{DEFECT[q['id']]}}}$" if q["id"] in DEFECT else ""
        rows.append(" & ".join([
            q["id"] + flag, CAT[q["category"]], tex(q["question"]), gold_cell(q),
            GRADER.get(q["answer_type"], "EM"), evidence(q["id"], q["answer_type"])]) + r" \\")
    return "\n".join(rows) + "\n"


def table_c(rows: list[dict]) -> str:
    by = defaultdict(dict)
    why = defaultdict(set)
    for r in rows:
        by[r["qid"]][(r["backbone"], r["repeat"])] = (int(r["strict"]), int(r["norm"]), int(r["audited"]))
        if r["reason"]:
            why[r["qid"]].add(r["reason"].split(";")[0])
    out = []
    for qid in sorted(by):
        cells = by[qid]
        if all(v == (1, 1, 1) for v in cells.values()):
            continue
        def triple(b):
            return " \\,|\\, ".join("".join("\\checkmark" if x else "$\\times$" for x in cells[(b, k)])
                                    for k in ("1", "2", "3"))
        out.append(f"{qid} & {triple('small')} & {triple('large')} & {tex(TABLE_REASON.get(qid, '; '.join(sorted(why[qid]))))} \\\\")
    return "\n".join(out) + "\n"


def table_d(bench: list[dict]) -> str:
    exp = [q["id"] for q in bench if q["answer_type"] == "explanation"]
    scores = {}
    for name, pat in (("small", "results/gemini/S4_r*.jsonl"), ("large", "results/experiments/S4_r*.jsonl")):
        for fp in sorted(glob.glob(pat)):
            for line in open(fp, encoding="utf-8"):
                r = json.loads(line)
                if r["question_id"] in exp:
                    scores.setdefault((name, r["question_id"]), []).append(r.get("rubric"))
    out = []
    for qid in exp:
        s, l = scores[("small", qid)], scores[("large", qid)]
        out.append(f"{qid} & {' / '.join(map(str, s))} & {' / '.join(map(str, l))} \\\\")
    def m(name):
        v = [x for (n, _), xs in scores.items() if n == name for x in xs if x is not None]
        return sum(v) / len(v)
    out.append(r"\midrule")
    out.append(f"Mean & {m('small'):.2f} & {m('large'):.2f} \\\\")
    return "\n".join(out) + "\n"


def main() -> None:
    bench = [json.loads(l) for l in open(BENCH, encoding="utf-8")]
    rows = list(csv.DictReader(open(AUDIT, encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark_rows.tex").write_text(table_a(bench), encoding="utf-8")
    (OUT / "audit_rows.tex").write_text(table_c(rows), encoding="utf-8")
    (OUT / "explanation_rows.tex").write_text(table_d(bench), encoding="utf-8")
    print("wrote", ", ".join(str(p) for p in sorted(OUT.glob("*.tex"))))


if __name__ == "__main__":
    main()
