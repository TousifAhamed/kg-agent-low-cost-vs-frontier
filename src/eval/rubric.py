"""LLM-as-judge rubric grader for Explanation questions (roadmap Appendix B, 0-3).

Explanation answers can't be exact-matched, so a judge scores them against the gold
evidence trace (the KG nodes that SHOULD appear in a correct explanation):

  0  wrong / no real evidence
  1  partially correct, missing key steps or evidence
  2  correct reasoning, most evidence present
  3  fully correct end-to-end trace with the right evidence nodes

The judge is the same backbone but a separate call with a fixed rubric; it sees the
question, the gold evidence node ids, and the candidate answer. Returns an int 0-3.
"""
from __future__ import annotations

import re

from src.llm import LLM

JUDGE_SYSTEM = (
    "You are a strict grader for petroleum-analytics explanations. Score the candidate "
    "answer from 0 to 3 against the rubric. Reward correct causal chains "
    "(alert -> triggering reading -> recommended intervention), correct use of "
    "connectivity weights, and reference to the listed gold evidence nodes. Penalize "
    "fabricated ids/numbers and missing steps. Reply with exactly one line 'SCORE: <0|1|2|3>'.")

RUBRIC = ("0 = wrong or unsupported; 1 = partially correct, key steps/evidence missing; "
          "2 = correct reasoning with most evidence; 3 = fully correct end-to-end trace "
          "citing the right evidence nodes.")


def grade_explanation(question: str, gold_nodes: list, answer: str, judge: LLM | None = None) -> int:
    judge = judge or LLM(max_tokens=64)
    user = (f"Rubric: {RUBRIC}\n\nQuestion: {question}\n"
            f"Gold evidence node ids (should be reflected): {gold_nodes}\n\n"
            f"Candidate answer:\n{answer}\n\nScore now.")
    out = judge.ask(JUDGE_SYSTEM, user)
    m = re.search(r"SCORE:\s*([0-3])", out)
    return int(m.group(1)) if m else 0
