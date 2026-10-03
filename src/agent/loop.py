"""S4: the LLM-over-KG ReAct agent (roadmap M3).

A tool-use loop over the three grounded tools (src/agent/tools.py). The backbone
(Claude Opus 4.8) plans, calls tools, and must finish with an ANSWER + CITATIONS
block. The citation guardrail (src/agent/cite.py) then checks the cited ids against
the graph AND against what tools actually returned this run; on failure the agent is
asked to fix it (retry <= 2). Entity linking (src/agent/entity_linker.py) seeds the
first turn so the model starts from the right neighbourhood.

  from src.agent.loop import Agent
  Agent().answer("Which injectors are connected to producer F-12?")
"""
from __future__ import annotations

from pathlib import Path

from src.agent import cite
from src.agent.entity_linker import link
from src.agent.retrieval import KG, SemanticIndex
from src.agent.tools import SCHEMAS, Tools, schemas_for
from src.llm import LLM

SYSTEM = Path("prompts/react_template.txt").read_text(encoding="utf-8")
MAX_TOOL_ROUNDS = 5
MAX_CITE_RETRIES = 2
ALL_TOOLS = [s["name"] for s in SCHEMAS]


class Agent:
    """Configurable KG agent. Defaults = the full S4 system.

    tools:         subset of {cypher_query, semantic_search, compute_metric} (ablations)
    use_guardrail: if False, accept the first answer (no citation-retry) — guardrail ablation
    kg / semantic: inject a shared/perturbed KG or a pre-built index (avoids re-encoding)
    """

    def __init__(self, embedder: str = "minilm", tools: list[str] | None = None,
                 use_guardrail: bool = True, kg: KG | None = None, semantic: SemanticIndex | None = None):
        self.kg = kg or KG()
        self.schemas = schemas_for(tools or ALL_TOOLS)
        self.tools = Tools(self.kg, semantic or SemanticIndex(embedder))
        self.use_guardrail = use_guardrail
        self.max_retries = MAX_CITE_RETRIES if use_guardrail else 0
        self.llm = LLM(max_tokens=1024)

    def answer(self, question: str) -> dict:
        hint = link(question, self.kg)
        seed = f"\n\n[linked KG seed nodes: {hint['seed_nodes']}]" if hint["seed_nodes"] else ""
        history = [{"role": "user", "text": question + seed}]
        retrieved: set[str] = set()
        trace: list[dict] = []
        cite_retries = 0

        for _round in range(MAX_TOOL_ROUNDS + self.max_retries + 1):
            resp = self.llm.tool_step(SYSTEM, history, self.schemas)

            if resp["stop"] == "tool_use":
                history.append({"role": "tool_calls", "calls": resp["tool_calls"],
                                "raw": resp["raw"]})
                results = []
                for call in resp["tool_calls"]:
                    payload, cited = self.tools.dispatch(call["name"], call["input"])
                    retrieved.update(cited)
                    trace.append({"tool": call["name"], "input": call["input"],
                                  "cited": cited})
                    results.append({"id": call["id"], "name": call["name"],
                                    "content": _json(payload)})
                history.append({"role": "tool_results", "results": results})
                continue

            # end: model produced a textual answer
            text = resp["text"]
            citations = cite.parse_citations(text)
            check = cite.validate(citations, self.kg, retrieved)
            if not self.use_guardrail or check["ok"] or cite_retries >= self.max_retries:
                return {"answer": cite.parse_answer(text), "raw": text,
                        "citations": check["valid"], "citation_check": check,
                        "n_tool_calls": len(trace), "llm_calls": self.llm.calls,
                        "trace": trace}
            # guardrail retry
            cite_retries += 1
            history.append({"role": "assistant", "text": text})
            history.append({"role": "user", "text": _fix_msg(check)})

        # Ran out of tool rounds without a final answer. Weaker backbones can loop on
        # tool calls; force a no-tool response so we always return a graded answer.
        history.append({"role": "user", "text":
                        "Stop calling tools. Using only the evidence already gathered, give "
                        "your final answer now as an ANSWER line followed by a CITATIONS line."})
        final = self.llm.tool_step(SYSTEM, history, [])
        text = final["text"]
        check = cite.validate(cite.parse_citations(text), self.kg, retrieved)
        return {"answer": cite.parse_answer(text), "raw": text, "citations": check["valid"],
                "citation_check": check, "n_tool_calls": len(trace),
                "llm_calls": self.llm.calls, "trace": trace}


def _json(payload) -> str:
    import json
    return json.dumps(payload, default=str)


def _fix_msg(check: dict) -> str:
    parts = []
    if check["not_in_kg"]:
        parts.append(f"these ids do not exist in the KG: {check['not_in_kg']}")
    if check["not_retrieved"]:
        parts.append(f"these ids were never returned by a tool this run: {check['not_retrieved']}")
    if not check["valid"]:
        parts.append("you cited no valid evidence")
    return ("Citation check failed: " + "; ".join(parts) +
            ". Call tools to obtain real node ids, then re-answer with a corrected "
            "ANSWER/CITATIONS block citing only ids returned by tools.")
