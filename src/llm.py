"""Provider-agnostic LLM backbone for the KG-RAG pipeline.

Two backends behind one interface so the project can run on Gemini's free tier while
keeping Claude selectable for a backbone-portability comparison:

  - GeminiLLM (default) — google-genai, model gemini-3.1-flash-lite.
  - ClaudeLLM            — anthropic, model claude-opus-4-8 (original behaviour).

Selected by env LLM_BACKEND in {gemini, claude} (default gemini). Both expose:
  - ask(system, user) -> str                      : single-shot (S1/S2/S3, rubric judge)
  - tool_step(system, history, schemas) -> dict   : one ReAct turn, provider-neutral shape
        {stop: "tool_use"|"end", text: str, tool_calls: [{id,name,input}], raw: any}

`history` is a provider-NEUTRAL list the agent loop owns; each backend rebuilds its own
message format from it every call. Neutral items:
  {"role":"user","text": str}
  {"role":"assistant","text": str}
  {"role":"tool_calls","calls":[{"id","name","input"}]}
  {"role":"tool_results","results":[{"id","name","content": str}]}

Tool schemas stay in the single neutral form defined in src/agent/tools.py (Anthropic-style
{name, description, input_schema}); each backend adapts them internally.

Key resolution order (per provider, value loaded into this process only, never printed):
  1. os.environ  2. .env (python-dotenv)  3. Windows HKCU\\Environment (setx values).
"""
from __future__ import annotations

import json
import os
import time

GEMINI_MODEL = "gemini-3.1-flash-lite"
CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_MODEL = GEMINI_MODEL

_MAX_API_RETRIES = 5


def _load_key(*names: str) -> str:
    """Resolve the first of `names` from env -> .env -> HKCU. Never printed."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                for name in names:
                    try:
                        val, _ = winreg.QueryValueEx(k, name)
                        if val:
                            os.environ[name] = val
                            return val
                    except OSError:
                        continue
        except OSError:
            pass
    raise RuntimeError(f"None of {names} found (env, .env, or HKCU\\Environment).")


def _is_daily_quota(exc: Exception) -> bool:
    """Free-tier PER-DAY cap — retrying today is futile, so fail fast."""
    s = str(exc).lower()
    return "perday" in s or "per day" in s or "requestsperday" in s


def _is_transient(exc: Exception) -> bool:
    """Retryable: short-window rate limits or transient server errors (not a daily cap)."""
    if _is_daily_quota(exc):
        return False
    s = str(exc).lower()
    return any(t in s for t in ("429", "resource_exhausted", "rate limit", "quota",
                                "503", "unavailable", "overloaded", "500", "internal"))


def _with_retry(fn):
    """Retry with exponential backoff on transient rate-limit errors (free-tier 429s)."""
    last = None
    for attempt in range(_MAX_API_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
            last = exc
            if not _is_transient(exc) or attempt == _MAX_API_RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt * 5, 60))
    raise last  # unreachable


# --------------------------------------------------------------------------- Gemini

class GeminiLLM:
    def __init__(self, model: str = GEMINI_MODEL, max_tokens: int = 1024):
        from google import genai
        self._genai = genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=_load_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.calls = 0

    def ask(self, system: str, user: str) -> str:
        t = self._types
        self.calls += 1
        resp = _with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=user,
            config=t.GenerateContentConfig(
                system_instruction=system, max_output_tokens=self.max_tokens)))
        return (resp.text or "").strip()

    def _tools(self, schemas: list[dict]):
        t = self._types
        decls = [t.FunctionDeclaration(name=s["name"], description=s["description"],
                                       parameters_json_schema=s["input_schema"])
                 for s in schemas]
        return [t.Tool(function_declarations=decls)]

    def _contents(self, history: list[dict]):
        t = self._types
        out = []
        for m in history:
            role = m["role"]
            if role == "user":
                out.append(t.Content(role="user", parts=[t.Part.from_text(text=m["text"])]))
            elif role == "assistant":
                out.append(t.Content(role="model", parts=[t.Part.from_text(text=m["text"])]))
            elif role == "tool_calls":
                # Replay the model's actual content: Gemini 3.x requires the original
                # thought_signature on functionCall parts to be echoed back verbatim.
                raw = m.get("raw")
                if raw is not None and getattr(raw, "candidates", None):
                    out.append(raw.candidates[0].content)
                else:
                    parts = [t.Part(function_call=t.FunctionCall(name=c["name"], args=c["input"]))
                             for c in m["calls"]]
                    out.append(t.Content(role="model", parts=parts))
            elif role == "tool_results":
                parts = [t.Part.from_function_response(name=r["name"], response=_as_dict(r["content"]))
                         for r in m["results"]]
                out.append(t.Content(role="tool", parts=parts))
        return out

    def tool_step(self, system: str, history: list[dict], schemas: list[dict]) -> dict:
        t = self._types
        self.calls += 1
        resp = _with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=self._contents(history),
            config=t.GenerateContentConfig(
                system_instruction=system, max_output_tokens=self.max_tokens,
                tools=self._tools(schemas) if schemas else None,
                automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))))
        calls = resp.function_calls or []
        if calls:
            tool_calls = [{"id": getattr(c, "id", None) or f"{c.name}_{i}",
                           "name": c.name, "input": dict(c.args or {})}
                          for i, c in enumerate(calls)]
            return {"stop": "tool_use", "text": "", "tool_calls": tool_calls, "raw": resp}
        return {"stop": "end", "text": (resp.text or "").strip(), "tool_calls": [], "raw": resp}


def _as_dict(content: str) -> dict:
    """Gemini function_response wants a dict; wrap non-dict JSON payloads."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"result": content}
    return obj if isinstance(obj, dict) else {"result": obj}


# --------------------------------------------------------------------------- Claude

class ClaudeLLM:
    def __init__(self, model: str = CLAUDE_MODEL, max_tokens: int = 1024):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=_load_key("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.calls = 0

    def ask(self, system: str, user: str) -> str:
        self.calls += 1
        resp = _with_retry(lambda: self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}]))
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    @staticmethod
    def _schemas(schemas: list[dict]) -> list[dict]:
        return [{"name": s["name"], "description": s["description"],
                 "input_schema": s["input_schema"]} for s in schemas]

    def _messages(self, history: list[dict]) -> list[dict]:
        out = []
        for m in history:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["text"]})
            elif role == "assistant":
                out.append({"role": "assistant", "content": m["text"]})
            elif role == "tool_calls":
                out.append({"role": "assistant",
                            "content": [{"type": "tool_use", "id": c["id"],
                                         "name": c["name"], "input": c["input"]}
                                        for c in m["calls"]]})
            elif role == "tool_results":
                out.append({"role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": r["id"],
                                         "content": r["content"]} for r in m["results"]]})
        return out

    def tool_step(self, system: str, history: list[dict], schemas: list[dict]) -> dict:
        self.calls += 1
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=self._messages(history))
        if schemas:
            kwargs["tools"] = self._schemas(schemas)
        resp = _with_retry(lambda: self.client.messages.create(**kwargs))
        if resp.stop_reason == "tool_use":
            tool_calls = [{"id": b.id, "name": b.name, "input": dict(b.input)}
                          for b in resp.content if getattr(b, "type", "") == "tool_use"]
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {"stop": "tool_use", "text": text, "tool_calls": tool_calls, "raw": resp}
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return {"stop": "end", "text": text.strip(), "tool_calls": [], "raw": resp}


# --------------------------------------------------------------------------- facade

def LLM(model: str | None = None, max_tokens: int = 1024):
    """Return the configured backend (env LLM_BACKEND, default 'gemini')."""
    backend = os.environ.get("LLM_BACKEND", "gemini").strip().lower()
    if backend == "claude":
        return ClaudeLLM(model or CLAUDE_MODEL, max_tokens)
    return GeminiLLM(model or GEMINI_MODEL, max_tokens)
