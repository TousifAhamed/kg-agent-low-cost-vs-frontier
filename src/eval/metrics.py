"""Grading for the benchmark (roadmap Appendix B).

Exact-Match for entity_list / scalar / boolean; NDCG@5 for ranking; explanation is
rubric-graded later (M3) and returns None here. Graders parse a model's free-text
answer robustly (regex token/number extraction) so baselines are scored fairly.
"""
from __future__ import annotations

import math
import re

# Suffix letters (F-15D, F-1C) are adjacent to the number in the id. A well id in
# free text may be written spaced ("F-15 D"); collapse that ONLY when the suffix is a
# lone capital letter, so "F-4 and ..." does NOT absorb the 'a' of "and".
SPACED_SUFFIX_RE = re.compile(r"\b([FIP]-?\d+)\s+([A-Z])\b")
WELL_RE = re.compile(r"[FIP]-?\d+[A-Z]?", re.I)
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_well(tok: str) -> str:
    t = tok.upper().replace(" ", "")
    m = re.match(r"([FIP])-?(\d+)([A-Z]?)", t)
    return f"{m.group(1)}-{m.group(2)}{m.group(3)}" if m else t


def _wells(text: str) -> set[str]:
    text = SPACED_SUFFIX_RE.sub(r"\1\2", text)
    return {_norm_well(t) for t in WELL_RE.findall(text)}


def _first_num(text: str) -> float | None:
    m = NUM_RE.search(text.replace(",", ""))
    return float(m.group()) if m else None


def exact_match(answer_type: str, gold, pred: str) -> int:
    pred = (pred or "").strip()
    if answer_type == "entity_list":
        gold_set = {_norm_well(str(g)) for g in (gold if isinstance(gold, list) else [gold])}
        return int(_wells(pred) == gold_set and len(gold_set) > 0)
    if answer_type == "boolean":
        pl = pred.lower()
        pos = any(w in pl for w in ["yes", "true", "correct", "affirmative"])
        neg = any(w in pl for w in ["no", "false", "not "])
        pv = True if (pos and not neg) else (False if neg else None)
        return int(pv is not None and pv == bool(gold))
    # scalar
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        pv = _first_num(pred)
        return int(pv is not None and abs(pv - float(gold)) <= max(0.02 * abs(float(gold)), 1e-4))
    # scalar string. Golds with >=2 numbers (dates "2016-07-01", counts
    # "5 producers, 2 injectors") are matched by number sequence so a correct answer
    # phrased differently still passes; single-token golds (ids, action types, "F-5")
    # use a normalized substring match.
    gnums = _nums(str(gold))
    if len(gnums) >= 2:
        return int(_greedy_subseq(gnums, _nums(pred)))
    g = str(gold).lower().replace(" ", "")
    return int(g in pred.lower().replace(" ", ""))


def _nums(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM_RE.findall(text)]


# ------------------------------------------------------------------ normalized EM
# Prose-tolerant variant (SQuAD-style normalization), applied UNIFORMLY to every
# system. Motivation (M5 failure analysis): weaker backbones answer in prose, and
# strict EM then fails answers that are verbatim-correct — `_first_num` picks digits
# out of well ids ("...from injector F-5 ... is 0.4368" -> 5), entity-set equality
# breaks on wells echoed from the question, and "July 1, 2016" != "2016-07-01".
# Golds are unchanged; only the parsing of the model's free text is relaxed.

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_DATE_RE = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")


def _norm_dates(text: str) -> str:
    """Rewrite 'July 1, 2016' as '2016-07-01' so number-sequence matching works."""
    def sub(m):
        mon = _MONTHS.get(m.group(1)[:3].lower())
        return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}" if mon else m.group(0)
    return _DATE_RE.sub(sub, text)


def _squash(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())


def exact_match_norm(answer_type: str, gold, pred: str, question: str = "") -> int:
    pred = (pred or "").strip()
    if answer_type == "entity_list":
        gold_set = {_norm_well(str(g)) for g in (gold if isinstance(gold, list) else [gold])}
        pw = _wells(pred)
        if not gold_set:
            return 0
        if pw == gold_set:
            return 1
        # prose answers echo wells named in the question; discount those (never golds)
        echo = _wells(question) - gold_set
        return int(pw - echo == gold_set)
    if answer_type == "boolean":
        return exact_match(answer_type, gold, pred)
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        tol = max(0.02 * abs(float(gold)), 1e-4)
        return int(any(abs(v - float(gold)) <= tol for v in _nums(pred)))
    # scalar string: date-normalize, then number-sequence or squashed substring match
    pred = _norm_dates(pred)
    gnums = _nums(_norm_dates(str(gold)))
    if len(gnums) >= 2:
        return int(_greedy_subseq(gnums, _nums(pred)))
    return int(_squash(str(gold)) in _squash(pred))


def _greedy_subseq(sub: list[float], seq: list[float]) -> bool:
    """True if `sub` appears in `seq` in order (extra numbers in seq allowed)."""
    i = 0
    for x in seq:
        if i < len(sub) and abs(x - sub[i]) < 1e-9:
            i += 1
    return i == len(sub)


def ndcg_at_5(gold_ranking: list[str], relevance: dict, pred: str) -> float:
    ids = re.findall(r"I_[A-Za-z0-9_\-]+", pred) or re.findall(r"[A-Za-z0-9_\-]{4,}", pred)
    seen, order = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); order.append(i)
    rel = relevance or {iid: 3 - min(2, r) for r, iid in enumerate(gold_ranking)}
    dcg = sum((2 ** rel.get(i, 0) - 1) / math.log2(rank + 2) for rank, i in enumerate(order[:5]))
    ideal = sorted(rel.values(), reverse=True)[:5]
    idcg = sum((2 ** r - 1) / math.log2(rank + 2) for rank, r in enumerate(ideal))
    return round(dcg / idcg, 4) if idcg else 0.0


def grade(q: dict, pred: str) -> dict:
    at = q["answer_type"]
    if at == "ranking":
        return {"ndcg_at_5": ndcg_at_5(q["gold"], q.get("relevance"), pred), "exact_match": None}
    if at == "explanation":
        return {"exact_match": None, "ndcg_at_5": None}  # rubric in M3
    return {"exact_match": exact_match(at, q["gold"], pred), "ndcg_at_5": None}
