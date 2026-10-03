"""Citation-validation guardrail (roadmap M3).

The agent must ground every answer in KG node ids that (a) actually exist in the
graph and (b) were actually returned by a tool during this run. This is the
anti-hallucination mechanism: in M2, Text-RAG invented plausible-but-fake
intervention ids (e.g. 'F-12_2014-01-01'); here such ids fail validation and the
loop is asked to retry (<=2 times) using only real evidence.
"""
from __future__ import annotations


def parse_citations(text: str) -> list[str]:
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("CITATIONS:"):
            raw = line.split(":", 1)[1]
            return [c.strip() for c in raw.replace(";", ",").split(",") if c.strip()]
    return []


def parse_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def validate(citations: list[str], kg, retrieved_ids: set[str]) -> dict:
    """Split citations into valid / not-in-kg / not-retrieved.

    valid = exists in KG AND was surfaced by a tool this run.
    """
    in_kg = [c for c in citations if kg.has_node(c)]
    not_in_kg = [c for c in citations if not kg.has_node(c)]
    not_retrieved = [c for c in in_kg if c not in retrieved_ids]
    valid = [c for c in in_kg if c in retrieved_ids]
    return {"valid": valid, "not_in_kg": not_in_kg, "not_retrieved": not_retrieved,
            "ok": bool(valid) and not not_in_kg}
