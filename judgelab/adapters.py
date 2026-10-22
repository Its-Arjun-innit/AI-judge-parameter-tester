"""Model Adapter: one function per provider, common signature.
Every call is stateless (no history). SDK-level retries are off; the runner retries
and logs every attempt. Cloud keys come only from environment variables; the local
Ollama provider needs none and never falls back to a cloud provider."""
import json
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .cases import Prompt
from .schema import ModelConfig

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class Raw:
    text: str
    model_returned: str | None
    tokens_in: int
    tokens_out: int
    stop_reason: str   # normalised: end | max_tokens | refusal | context_limit | <provider value>
    request_id: str | None
    latency_ms: int
    request: dict = field(default_factory=dict)  # settings actually sent (not the prompt text)
    meta: dict = field(default_factory=dict)     # provider-reported metrics, e.g. Ollama timings


def _req(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None}


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
    return Raw(text, r.model, r.usage.input_tokens, r.usage.output_tokens, stop, r.id, 0,
               _req(model=cfg.model, max_tokens=cfg.max_tokens, effort=cfg.effort, thinking=kw.get("thinking"),
                    temperature=cfg.temperature, format="response_schema"))


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
    return Raw(r.output_text or "", r.model, r.usage.input_tokens, r.usage.output_tokens, stop, r.id, 0,
               _req(model=cfg.model, max_output_tokens=cfg.max_tokens, format="response_schema", **kw))


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
    return Raw(r.text or "", r.model_version, u.prompt_token_count or 0, out_tokens, stop, r.response_id, 0,
               _req(model=cfg.model, max_output_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    thinking=thinking.model_dump(exclude_none=True) if thinking else None, format="response_schema"))


# ---- Ollama (local, no key) ----------------------------------------------------------
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local traffic never goes via a proxy
OLLAMA_INFO = {"version": None, "digests": {}}  # filled by ollama_preflight(); recorded with every attempt


def ollama_url() -> str:
    """Ollama address from OLLAMA_HOST (default localhost). Only this computer, unless
    JUDGELAB_ALLOW_REMOTE_OLLAMA=1 - the web interface can never point it elsewhere."""
    url = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
    if "://" not in url:
        url = "http://" + url
    p = urllib.parse.urlparse(url)
    host = "127.0.0.1" if p.hostname in (None, "0.0.0.0", "::") else p.hostname  # server bind-all -> connect locally
    if host not in LOOPBACK and os.environ.get("JUDGELAB_ALLOW_REMOTE_OLLAMA") != "1":
        raise ProviderError(f"The Ollama address {url} is not on this computer. "
                            "Set JUDGELAB_ALLOW_REMOTE_OLLAMA=1 to allow it.", None, False)
    return f"{p.scheme or 'http'}://{'[' + host + ']' if ':' in host else host}:{p.port or 11434}"


def _ollama_http(path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    url = ollama_url() + path
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    try:
        with _NO_PROXY.open(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code == 404 and body and "model" in body:
            raise ProviderError(f"Model '{body['model']}' is not installed in Ollama "
                                f"(install it with: ollama pull {body['model']}). {detail}", 404, False) from e
        raise ProviderError(f"Ollama returned HTTP {e.code}: {detail}", e.code, e.code in RETRY_STATUS) from e
    except (TimeoutError, socket.timeout) as e:
        raise ProviderError(f"Ollama did not answer within {timeout:.0f} s", None, True) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise ProviderError(f"Ollama did not answer within {timeout:.0f} s", None, True) from e
        raise ProviderError(f"Ollama is not running at {url.rsplit('/api', 1)[0]}. Start the Ollama app, "
                            f"then try again. ({e.reason})", None, False) from e


def ollama_status() -> tuple[bool, str]:
    """(running, version or a plain-language problem)."""
    try:
        return True, _ollama_http("/api/version", timeout=3).get("version", "unknown")
    except ProviderError as e:
        return False, str(e)


def ollama_models() -> list[dict]:
    """Installed models (never downloads anything)."""
    out = []
    for m in _ollama_http("/api/tags", timeout=5).get("models", []):
        d = m.get("details") or {}
        out.append({"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 1),
                    "parameters": d.get("parameter_size"), "quantization": d.get("quantization_level"),
                    "family": d.get("family"), "digest": m.get("digest")})
    return out


def ollama_show(name: str) -> dict:
    """Context length, capabilities and thinking options reported by Ollama for one model."""
    r = _ollama_http("/api/show", {"model": name}, timeout=10)
    ctx = next((v for k, v in (r.get("model_info") or {}).items() if k.endswith(".context_length")), None)
    return {"context_length": ctx, "capabilities": r.get("capabilities") or [],
            "thinking": (r.get("thinking") or {}).get("values")}


def ollama_preflight(names: list[str]) -> dict:
    """Before any analysis: Ollama must be running and every model installed. Records digests."""
    ok, info = ollama_status()
    if not ok:
        raise ProviderError(info)
    installed = {m["name"]: m["digest"] for m in ollama_models()}
    missing = [n for n in names if n not in installed and f"{n}:latest" not in installed]
    if missing:
        raise ProviderError(f"Not installed in Ollama: {', '.join(missing)}. Install with `ollama pull <name>`; "
                            "models are never downloaded automatically.")
    OLLAMA_INFO.update(version=info, digests=installed)
    return OLLAMA_INFO


def _ollama(cfg, prompt, schema):
    options = _req(num_predict=cfg.max_tokens, num_ctx=cfg.num_ctx, seed=cfg.seed, temperature=cfg.temperature)
    body = {"model": cfg.model, "stream": False, "keep_alive": "10m", "options": options, "format": schema,
            "messages": [{"role": "system", "content": prompt.system},   # exactly two messages:
                         {"role": "user", "content": prompt.user}]}      # no history between analyses
    if cfg.thinking is not None:
        body["think"] = cfg.thinking
    r = _ollama_http("/api/chat", body, timeout=cfg.timeout_s)
    tin, tout = r.get("prompt_eval_count") or 0, r.get("eval_count") or 0
    stop = {"stop": "end", "length": "max_tokens"}.get(r.get("done_reason"), r.get("done_reason") or "unknown")
    if cfg.num_ctx and tin + (r.get("prompt_eval_cached_count") or 0) + tout >= cfg.num_ctx:
        stop = "context_limit"  # Ollama silently drops the start of an over-long prompt
    digest = OLLAMA_INFO["digests"].get(cfg.model) or OLLAMA_INFO["digests"].get(f"{cfg.model}:latest")
    returned = r.get("model") or cfg.model
    if digest:
        returned += f"@{digest[:12]} (ollama {OLLAMA_INFO['version']})"
    meta = {k: r.get(k) for k in ("done_reason", "total_duration", "load_duration", "prompt_eval_count",
                                  "prompt_eval_cached_count", "eval_count", "created_at")}
    request = {k: v for k, v in body.items() if k not in ("messages", "format")} | {"format": "response_schema"}
    return Raw((r.get("message") or {}).get("content") or "", returned, tin, tout, stop, None, 0, request, meta)


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
    return Raw(json.dumps(out), "fake", len(prompt.user) // 4, 120, "end", None, 0, {"model": "fake"})


_CALL = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini, "ollama": _ollama, "fake": _fake}
