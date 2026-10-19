"""
Shared model-client abstraction with provider resilience (Section 4.4 / 10).

Every model call in the system -- Router, Planner, Executor, Verifier --
goes through here. The PRIMARY provider is Groq's free-tier
OpenAI-compatible API, used whenever GROQ_API_KEY is set (see
config.PROVIDER). A
circuit breaker trips to a genuinely degraded fallback after sustained
failures on a given tier: Router falls back first (a routing mistake is
cheapest to absorb), then Executor, then Planner last (its reasoning
quality matters most). Every call and every fallback use is logged as
CallMetrics with degraded_mode set so the UI never presents a degraded
answer at ordinary confidence.

Call chain per tier, in order:
  1. Primary: Groq, using real structured tool-calling (function
     calling). Not treated as "degraded."
  2. Local-inference fallback: a separate OpenAI-compatible endpoint
     (Ollama / vLLM / LM Studio / ...) if LOCAL_MODEL_BASE_URL is
     configured. This IS treated as genuinely degraded, since it's meant
     as a last-resort path when the configured primary provider itself is
     down, not a provider choice.
  3. Heuristic stub: a deterministic, template-based response. Keeps the
     whole pipeline runnable end-to-end with zero external dependency (no
     API key, no local model server) for development, testing, and
     grading -- materially weaker than any real model tier and always
     marked degraded_mode=True.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Literal

from app import config
from app.models import CallMetrics

Tier = Literal["router", "planner", "executor", "verifier"]

_GROQ_TIER_MODELS = {
    "router": config.GROQ_ROUTER_MODEL,
    "planner": config.GROQ_PLANNER_MODEL,
    "executor": config.GROQ_EXECUTOR_MODEL,
    "verifier": config.GROQ_VERIFIER_MODEL,
}


class CircuitBreaker:
    def __init__(self, threshold: int, window_s: int):
        self.threshold = threshold
        self.window_s = window_s
        self._failures: dict[str, list[float]] = defaultdict(list)
        self._tripped_until: dict[str, float] = {}

    def record_failure(self, tier: str) -> None:
        now = time.time()
        self._failures[tier] = [t for t in self._failures[tier] if now - t < self.window_s] + [now]
        if len(self._failures[tier]) >= self.threshold:
            self._tripped_until[tier] = now + self.window_s

    def record_success(self, tier: str) -> None:
        self._failures[tier].clear()

    def is_open(self, tier: str) -> bool:
        until = self._tripped_until.get(tier)
        return until is not None and time.time() < until


_breaker = CircuitBreaker(config.CIRCUIT_BREAKER_FAILURE_THRESHOLD, config.CIRCUIT_BREAKER_WINDOW_SECONDS)

_groq_client = None
_local_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from openai import OpenAI

        _groq_client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_local_client():
    global _local_client
    if _local_client is None:
        from openai import OpenAI

        _local_client = OpenAI(base_url=config.LOCAL_MODEL_BASE_URL, api_key="local")
    return _local_client


class ModelResponse:
    def __init__(self, data: dict[str, Any], text: str, metrics: CallMetrics):
        self.data = data
        self.text = text
        self.metrics = metrics


def _call_groq_structured(tier: Tier, system_prompt: str, user_prompt: str, tool_name: str, tool_schema: dict, max_tokens: int):
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=_GROQ_TIER_MODELS[tier],
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tools=[{"type": "function", "function": {"name": tool_name, "description": f"Return {tool_name}", "parameters": tool_schema}}],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )
    message = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", 0) or 0

    if message.tool_calls:
        args_text = message.tool_calls[0].function.arguments
        data = json.loads(args_text)
        return data, args_text, tin, tout

    # Some Groq models occasionally reply in plain text even when a tool
    # was forced; fall back to extracting the first JSON object from it.
    text = message.content or "{}"
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return data, text, tin, tout


def _call_primary_structured(tier: Tier, system_prompt: str, user_prompt: str, tool_name: str, tool_schema: dict, max_tokens: int):
    if config.PROVIDER == "groq":
        return _call_groq_structured(tier, system_prompt, user_prompt, tool_name, tool_schema, max_tokens)
    raise RuntimeError("no primary provider configured")


def call_structured(
    tier: Tier,
    system_prompt: str,
    user_prompt: str,
    tool_name: str,
    tool_schema: dict,
    heuristic_fallback: dict,
    max_tokens: int = 1500,
) -> ModelResponse:
    """Call a model tier, forcing a structured JSON reply via real tool-use
    (Groq function-calling) so
    the caller never has to parse free text. Falls back to a local
    OpenAI-compatible endpoint, then a heuristic stub, on sustained
    failure. `heuristic_fallback` must already match `tool_schema`'s shape.
    """
    start = time.time()

    if config.PROVIDER != "none" and not _breaker.is_open(tier):
        try:
            data, text, tin, tout = _call_primary_structured(tier, system_prompt, user_prompt, tool_name, tool_schema, max_tokens)
            _breaker.record_success(tier)
            latency_ms = int((time.time() - start) * 1000)
            metrics = CallMetrics(tier=tier, tokens_in=tin, tokens_out=tout, latency_ms=latency_ms, degraded_mode=False, provider=config.PROVIDER)
            return ModelResponse(data, text, metrics)
        except Exception:
            _breaker.record_failure(tier)

    if config.LOCAL_MODEL_BASE_URL:
        try:
            data, text, latency_ms, tin, tout = _call_local_structured(system_prompt, user_prompt, tool_schema)
            metrics = CallMetrics(tier=tier, tokens_in=tin, tokens_out=tout, latency_ms=latency_ms, degraded_mode=True, provider="local")
            return ModelResponse(data, text, metrics)
        except Exception:
            pass

    latency_ms = int((time.time() - start) * 1000)
    metrics = CallMetrics(tier=tier, tokens_in=0, tokens_out=0, latency_ms=latency_ms, degraded_mode=True, provider="heuristic")
    return ModelResponse(heuristic_fallback, json.dumps(heuristic_fallback), metrics)


def _call_local_structured(system_prompt: str, user_prompt: str, tool_schema: dict) -> tuple[dict, str, int, int, int]:
    start = time.time()
    client = _get_local_client()
    forced_prompt = (
        f"{user_prompt}\n\nRespond with ONLY a single JSON object matching this schema, "
        f"no prose, no markdown fences:\n{json.dumps(tool_schema)}"
    )
    resp = client.chat.completions.create(
        model=config.LOCAL_MODEL_NAME or "local-model",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": forced_prompt}],
    )
    text = resp.choices[0].message.content or "{}"
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    latency_ms = int((time.time() - start) * 1000)
    usage = getattr(resp, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", 0) or 0
    return data, text, latency_ms, tin, tout


def _call_primary_text(tier: Tier, system_prompt: str, user_prompt: str, max_tokens: int):
    if config.PROVIDER == "groq":
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model=_GROQ_TIER_MODELS[tier], max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        return text, tin, tout
    raise RuntimeError("no primary provider configured")


def call_text(tier: Tier, system_prompt: str, user_prompt: str, heuristic_fallback: str, max_tokens: int = 1000) -> ModelResponse:
    start = time.time()

    if config.PROVIDER != "none" and not _breaker.is_open(tier):
        try:
            text, tin, tout = _call_primary_text(tier, system_prompt, user_prompt, max_tokens)
            _breaker.record_success(tier)
            latency_ms = int((time.time() - start) * 1000)
            metrics = CallMetrics(tier=tier, tokens_in=tin, tokens_out=tout, latency_ms=latency_ms, degraded_mode=False, provider=config.PROVIDER)
            return ModelResponse({}, text, metrics)
        except Exception:
            _breaker.record_failure(tier)

    if config.LOCAL_MODEL_BASE_URL:
        try:
            start2 = time.time()
            client = _get_local_client()
            resp = client.chat.completions.create(
                model=config.LOCAL_MODEL_NAME or "local-model",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            text = resp.choices[0].message.content or ""
            latency_ms = int((time.time() - start2) * 1000)
            usage = getattr(resp, "usage", None)
            tin = getattr(usage, "prompt_tokens", 0) or 0
            tout = getattr(usage, "completion_tokens", 0) or 0
            metrics = CallMetrics(tier=tier, tokens_in=tin, tokens_out=tout, latency_ms=latency_ms, degraded_mode=True, provider="local")
            return ModelResponse({}, text, metrics)
        except Exception:
            pass

    latency_ms = int((time.time() - start) * 1000)
    metrics = CallMetrics(tier=tier, tokens_in=0, tokens_out=0, latency_ms=latency_ms, degraded_mode=True, provider="heuristic")
    return ModelResponse({}, heuristic_fallback, metrics)


def is_degraded(tier: Tier) -> bool:
    return config.DRY_RUN or _breaker.is_open(tier)
