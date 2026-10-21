"""Model Adapter: one function per provider, common signature.
Every call is stateless (no history). SDK-level retries are off; the runner retries
and logs every attempt. Keys come only from environment variables."""
import json
import os
import random
import time
from dataclasses import dataclass

from .cases import Prompt
from .schema import ModelConfig

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class Raw:
    text: str
    model_returned: str | None
    tokens_in: int
    tokens_out: int
    stop_reason: str   # normalised: end | max_tokens | refusal | <provider value>
    request_id: str | None
    latency_ms: int


class ProviderError(Exception):
    def __init__(self, msg, status=None, retryable=False):
        super().__init__(msg)
        self.status, self.retryable = status, retryable


def call_model(cfg: ModelConfig, prompt: Prompt, schema: dict) -> Raw:
    t0 = time.monotonic()
    try:
        raw = _CALL[cfg.provider](cfg, prompt, schema)
    except ProviderError:
        raise
    except Exception as e:  # classify SDK errors once, for all providers
        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        status = status if isinstance(status, int) else None
        conn = type(e).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"}
        raise ProviderError(f"{type(e).__name__}: {e}", status, conn or status in RETRY_STATUS) from e
    raw.latency_ms = int((time.monotonic() - t0) * 1000)
    return raw


def _anthropic(cfg, prompt, schema):
    import anthropic
    client = anthropic.Anthropic(max_retries=0)  # resolves ANTHROPIC_API_KEY / ant profile
    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if cfg.effort:
        output_config["effort"] = cfg.effort
    extra = {"temperature": cfg.temperature} if cfg.temperature is not None else None  # removed on current models
    kw = {"thinking": {"type": "disabled"}} if cfg.thinking is False else {}
    r = client.messages.create(model=cfg.model, max_tokens=cfg.max_tokens, system=prompt.system,
                               messages=[{"role": "user", "content": prompt.user}],
                               output_config=output_config, extra_body=extra, **kw)
    text = "".join(b.text for b in r.content if b.type == "text")
    stop = {"end_turn": "end", "max_tokens": "max_tokens", "refusal": "refusal"}.get(r.stop_reason, r.stop_reason)
    return Raw(text, r.model, r.usage.input_tokens, r.usage.output_tokens, stop, r.id, 0)


def _openai(cfg, prompt, schema):
    import openai
    client = openai.OpenAI(max_retries=0)  # OPENAI_API_KEY
    kw = {}
    if cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if cfg.effort:
        kw["reasoning"] = {"effort": cfg.effort}
    r = client.responses.create(model=cfg.model, instructions=prompt.system, input=prompt.user,
                                max_output_tokens=cfg.max_tokens, store=False,
                                text={"format": {"type": "json_schema", "name": "judgment",
                                                 "schema": schema, "strict": True}}, **kw)
    refused = any(getattr(c, "type", None) == "refusal"
                  for o in r.output for c in (getattr(o, "content", None) or []))
    reason = getattr(r.incomplete_details, "reason", None) if r.incomplete_details else None
    stop = "refusal" if refused or reason == "content_filter" else \
        "max_tokens" if reason == "max_output_tokens" else "end" if r.status == "completed" else str(r.status)
    return Raw(r.output_text or "", r.model, r.usage.input_tokens, r.usage.output_tokens, stop, r.id, 0)


def _gemini(cfg, prompt, schema):
    from google import genai
    from google.genai import types
    client = genai.Client()  # GEMINI_API_KEY / GOOGLE_API_KEY
    thinking = (types.ThinkingConfig(thinking_level=cfg.effort.upper()) if cfg.effort
                else types.ThinkingConfig(thinking_budget=0) if cfg.thinking is False else None)
    r = client.models.generate_content(
        model=cfg.model, contents=prompt.user,
        config=types.GenerateContentConfig(
            system_instruction=prompt.system, temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens, response_mime_type="application/json",
            response_json_schema=schema, thinking_config=thinking))
    u = r.usage_metadata
    out_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)  # thinking is billed as output
    blocked = r.prompt_feedback and r.prompt_feedback.block_reason
    fr = str(r.candidates[0].finish_reason.name) if r.candidates and r.candidates[0].finish_reason else None
    stop = "refusal" if blocked or fr in {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST"} else \
        "max_tokens" if fr == "MAX_TOKENS" else "end" if fr == "STOP" else str(fr)
    return Raw(r.text or "", r.model_version, u.prompt_token_count or 0, out_tokens, stop, r.response_id, 0)


def _fake(cfg, prompt, schema):
    """Offline stand-in for --dry-run: plausible JSON that moves with the evidence."""
    rng = random.Random(hash((prompt.user, cfg.key, os.urandom(4))))
    pros = sum(side == "prosecution" for side, _ in prompt.blocks)
    defs = len(prompt.blocks) - pros
    p = max(0, min(100, round(rng.gauss(35 + 8 * pros - 10 * defs, 8))))
    labels = list(prompt.label_map)[:3]
    out = {"reasoning": "Synthetic dry-run response.", "legal_standard_applied": "Beyond reasonable doubt.",
           "key_evidence": [{"label": lab, "bearing": "supports_conviction"} for lab in labels],
           "missing_information": [], "probability_committed": p,
           "verdict": "convict" if p >= 75 else "acquit", "confidence": rng.randint(3, 6)}
    return Raw(json.dumps(out), "fake", len(prompt.user) // 4, 120, "end", None, 0)


_CALL = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini, "fake": _fake}
