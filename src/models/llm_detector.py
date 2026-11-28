"""Local zero-shot LLM phishing detector for CYB-5 (Ollama, loopback only).

Scope: *detection* of phishing by a generative model. This module never asks
the model to write phishing text, never sends a message to a remote service and
never receives the true label. The prompt is frozen in ``configs/llm_prompt_v1.txt``
and its SHA-256 is pinned in ``configs/llm_config_v1.json``; a mismatch aborts.

Error policy (explicit, frozen in the config): any output that is not a valid
label (invalid JSON, refusal, transport error, timeout) is counted as an error
and mapped to ``invalid_policy`` (default: LEGITIMATE, i.e. "not flagged").
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error, parse, request

LABEL_TO_INT = {"PHISHING": 1, "LEGITIMATE": 0}
SYSTEM_MARK = "=====SYSTEM====="
USER_MARK = "=====USER====="
EMAIL_PLACEHOLDER = "{EMAIL}"

_LABEL_JSON_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["PHISHING", "LEGITIMATE"]}},
    "required": ["label"],
    "additionalProperties": False,
}
_REFUSAL_RE = re.compile(
    r"(i can(?:'|no)t|i am unable|i'm unable|i won't|sorry|as an ai)", re.IGNORECASE
)
_LABEL_RE = re.compile(
    r"""["']?label["']?\s*:\s*["']?(PHISHING|LEGITIMATE)\b""", re.IGNORECASE
)
_EMAIL_TAG_RE = re.compile(r"<\s*/?\s*email\s*>", re.IGNORECASE)

# Conservative upper bound used only to *flag* possible context overflow.
_MIN_CHARS_PER_TOKEN = 2.5

OUTPUT_CLASSES = ("strict", "lenient", "invalid", "error")


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class LLMConfig:
    model: str
    ollama_url: str
    prompt_file: str
    prompt_sha256: str
    temperature: float
    seed: int
    num_ctx: int
    num_predict: int
    constrained_decoding: bool
    invalid_policy: str
    timeout_s: float
    keep_alive: str
    max_email_chars: Optional[int]
    model_license_note: str = ""

    @staticmethod
    def load(path: Path) -> "LLMConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in LLMConfig.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = LLMConfig(**raw)
        if cfg.invalid_policy not in ("legitimate", "phishing"):
            raise ValueError("invalid_policy must be 'legitimate' or 'phishing'")
        if cfg.temperature != 0:
            raise ValueError("Protocol requires temperature 0")
        assert_loopback(cfg.ollama_url)
        return cfg


def assert_loopback(url: str) -> None:
    """Refuse any non-local endpoint: emails must never leave the machine."""
    host = parse.urlparse(url).hostname or ""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(f"Refusing non-loopback endpoint (privacy): {url}")


# --------------------------------------------------------------------------- prompt
def load_prompt(path: Path, expected_sha256: Optional[str] = None) -> tuple[str, str, str]:
    """Return (system, user_template, sha256). Aborts if the frozen hash differs."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise ValueError(
            "Prompt differs from the frozen version "
            f"(expected {expected_sha256[:12]}..., got {sha[:12]}...). "
            "Create llm_prompt_v2.txt and a new config instead of editing v1."
        )
    text = raw.decode("utf-8")
    if text.count(SYSTEM_MARK) != 1 or text.count(USER_MARK) != 1:
        raise ValueError("Prompt file needs exactly one SYSTEM and one USER section")
    head, user = text.split(USER_MARK)
    system = head.split(SYSTEM_MARK)[1].strip()
    user = user.strip()
    if user.count(EMAIL_PLACEHOLDER) != 1:
        raise ValueError("User template needs exactly one {EMAIL} placeholder")
    return system, user, sha


# --------------------------------------------------------------------------- parsing
def parse_output(raw: str) -> tuple[Optional[str], str]:
    """Parse a model reply into (label | None, output_class).

    strict  : the whole reply is exactly {"label": "PHISHING"|"LEGITIMATE"}
    lenient : exactly one valid label can be recovered (fences, extra prose)
    invalid : anything else, including replies naming both labels
    """
    s = (raw or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and set(obj) == {"label"} and obj["label"] in LABEL_TO_INT:
            return obj["label"], "strict"
    except (ValueError, TypeError):
        pass
    found = {m.upper() for m in _LABEL_RE.findall(s)}
    if len(found) == 1:
        return next(iter(found)), "lenient"
    return None, "invalid"


def to_pred(label: Optional[str], policy: str) -> int:
    if label is not None:
        return LABEL_TO_INT[label]
    return 0 if policy == "legitimate" else 1


def sanitize_email(text: str, max_chars: Optional[int]) -> tuple[str, bool]:
    """Neutralise our own delimiter tags; optionally cap the length (default: no cap)."""
    body = "" if text is None else str(text)
    capped = False
    if max_chars is not None and len(body) > max_chars:
        body, capped = body[:max_chars], True
    return _EMAIL_TAG_RE.sub("[email-tag]", body), capped


# --------------------------------------------------------------------------- result
@dataclass
class Prediction:
    label: Optional[str]
    pred: int
    output_class: str
    refusal: bool
    error: str
    latency_s: float
    prompt_tokens: Optional[int]
    gen_tokens: Optional[int]
    raw_chars: int
    context_overflow: bool
    capped: bool


# --------------------------------------------------------------------------- detector
class LLMDetector:
    def __init__(self, config: LLMConfig, root: Path = Path(".")):
        self.cfg = config
        prompt_path = Path(root) / config.prompt_file
        self.system, self.user_tpl, self.prompt_sha256 = load_prompt(
            prompt_path, config.prompt_sha256
        )
        # Bypass any HTTP(S)_PROXY: loopback traffic must stay local.
        self._opener = request.build_opener(request.ProxyHandler({}))

    # -- HTTP helpers
    def _call(self, path: str, payload: Optional[dict], timeout: float) -> dict:
        url = self.cfg.ollama_url.rstrip("/") + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with self._opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def describe_model(self) -> dict:
        """Version, digest and details of the served model (no email content)."""
        info = {"ollama_version": self._call("/api/version", None, 10).get("version")}
        tags = self._call("/api/tags", None, 30).get("models", [])
        entry = next((m for m in tags if m.get("name") in (self.cfg.model, self.cfg.model + ":latest")), None)
        if entry is None:
            raise RuntimeError(f"Model not pulled locally: {self.cfg.model}")
        show = self._call("/api/show", {"model": self.cfg.model}, 30)
        info.update(
            model=self.cfg.model, digest=entry.get("digest"), size_bytes=entry.get("size"),
            details=show.get("details", {}),
        )
        return info

    def build_messages(self, text: str) -> tuple[list[dict], bool]:
        body, capped = sanitize_email(text, self.cfg.max_email_chars)
        user = self.user_tpl.replace(EMAIL_PLACEHOLDER, body, 1)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}], capped

    def _payload(self, messages: list[dict]) -> dict:
        c = self.cfg
        payload = {
            "model": c.model, "messages": messages, "stream": False,
            "keep_alive": c.keep_alive,
            "options": {"temperature": c.temperature, "seed": c.seed,
                        "num_ctx": c.num_ctx, "num_predict": c.num_predict},
        }
        if c.constrained_decoding:
            payload["format"] = _LABEL_JSON_SCHEMA
        return payload

    def warm_up(self) -> float:
        """Load the model once; returns load seconds (kept out of latency stats)."""
        msgs, _ = self.build_messages("Warm-up message about a meeting schedule.")
        data = self._call("/api/chat", self._payload(msgs), self.cfg.timeout_s)
        return float(data.get("load_duration") or 0) / 1e9

    def classify(self, text: str) -> Prediction:
        messages, capped = self.build_messages(text)
        est_tokens = sum(len(m["content"]) for m in messages) / _MIN_CHARS_PER_TOKEN
        overflow = est_tokens > (self.cfg.num_ctx - self.cfg.num_predict)
        t0 = time.perf_counter()
        try:
            data = self._call("/api/chat", self._payload(messages), self.cfg.timeout_s)
        except (error.URLError, TimeoutError, OSError, ValueError) as exc:
            return Prediction(None, to_pred(None, self.cfg.invalid_policy), "error", False,
                              type(exc).__name__, time.perf_counter() - t0, None, None, 0,
                              overflow, capped)
        latency = time.perf_counter() - t0
        content = (data.get("message") or {}).get("content", "") or ""
        label, klass = parse_output(content)
        prompt_tokens = data.get("prompt_eval_count")
        if prompt_tokens is not None and prompt_tokens >= self.cfg.num_ctx - self.cfg.num_predict:
            overflow = True
        return Prediction(
            label, to_pred(label, self.cfg.invalid_policy), klass,
            bool(label is None and _REFUSAL_RE.search(content)), "", latency,
            prompt_tokens, data.get("eval_count"), len(content), overflow, capped,
        )
