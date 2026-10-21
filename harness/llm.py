from __future__ import annotations

import contextvars
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import model_cfg, sampling_cfg

Message = dict[str, str]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    latency: float = 0.0
    finish_reason: str = ""
    logprobs: list[dict] | None = None


@dataclass
class Budget:
    limit: int | None
    used: int = 0

    def charge(self, n: int) -> None:
        if self.limit is not None and self.used + n > self.limit:
            raise BudgetExceeded(f"{self.used}+{n} > {self.limit}")
        self.used += n

    @property
    def left(self) -> int:
        return 10**9 if self.limit is None else self.limit - self.used


@dataclass
class Scope:
    task_id: str = ""
    technique: str = ""
    seed: int = 0
    budget: Budget = field(default_factory=lambda: Budget(None))


_scope: contextvars.ContextVar[Scope] = contextvars.ContextVar("scope", default=Scope())


class scope:
    """with scope(task_id, technique, budget=B, seed=s): every call inside is charged and logged."""

    def __init__(self, task_id: str, technique: str, budget: int | None = None, seed: int = 0):
        self.s = Scope(task_id, technique, seed, Budget(budget))

    def __enter__(self) -> Scope:
        self.tok = _scope.set(self.s)
        return self.s

    def __exit__(self, *a):
        _scope.reset(self.tok)


def current_scope() -> Scope:
    return _scope.get()


class Ledger:
    def __init__(self, path: Path | None):
        self.path, self.lock = path, threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        if not self.path:
            return
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class Backend:
    supports_temperature = True
    supports_n = False

    def complete(self, messages: list[Message], *, n: int, temperature: float | None, seed: int | None,
                 max_tokens: int, stop: list[str] | None, effort: str | None, logprobs: int | None) -> list[Completion]:
        raise NotImplementedError


def _retry(fn, tries: int = 8):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if i == tries - 1 or (status is not None and status < 500 and status != 429):
                raise
            time.sleep(min(60, 2**i + random.random()))


class AnthropicBackend(Backend):
    supports_temperature = False

    def __init__(self, model: str):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def complete(self, messages, *, n, temperature, seed, max_tokens, stop, effort, logprobs):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        out = []
        for _ in range(n):
            t0 = time.time()
            kw: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, messages=msgs,
                                      thinking={"type": "adaptive"})
            if system:
                kw["system"] = system
            if stop:
                kw["stop_sequences"] = stop
            if effort:
                kw["extra_body"] = {"output_config": {"effort": effort}}
            r = _retry(lambda: self.client.messages.create(**kw))
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            det = getattr(r.usage, "output_tokens_details", None) or {}
            rt = det.get("thinking_tokens") if isinstance(det, dict) else getattr(det, "thinking_tokens", None)
            out.append(Completion(text, r.usage.input_tokens, r.usage.output_tokens, rt, time.time() - t0, r.stop_reason or ""))
        return out


class OpenAICompatBackend(Backend):
    supports_n = True

    def __init__(self, model: str, base_url: str, key_env: str, provider: str | None = None, closed: bool = False):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ[key_env])
        self.model, self.provider = model, provider
        self.supports_temperature = not closed
        self.supports_n = not provider

    def complete(self, messages, *, n, temperature, seed, max_tokens, stop, effort, logprobs):
        extra: dict[str, Any] = {}
        if self.provider:
            extra["providerOptions"] = {"gateway": {"only": [self.provider], "order": [self.provider]}}
        kw: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=max_tokens, n=n if self.supports_n else 1)
        if self.supports_temperature and temperature is not None:
            kw["temperature"] = temperature
        if seed is not None and self.supports_temperature:
            kw["seed"] = seed
        if stop:
            kw["stop"] = stop
        if effort:
            kw["reasoning_effort"] = effort
        if logprobs:
            kw["logprobs"], kw["top_logprobs"] = True, logprobs
        if extra:
            kw["extra_body"] = extra
        out = []
        for _ in range(1 if self.supports_n else n):
            t0 = time.time()
            r = _retry(lambda: self.client.chat.completions.create(**kw))
            u = r.usage
            det = getattr(u, "completion_tokens_details", None)
            rt = getattr(det, "reasoning_tokens", None) if det else None
            for c in r.choices:
                lp = [{"token": t.token, "logprob": t.logprob, "top": {x.token: x.logprob for x in (t.top_logprobs or [])}}
                      for t in (c.logprobs.content or [])] if logprobs and c.logprobs else None
                out.append(Completion(c.message.content or "", u.prompt_tokens // max(1, len(r.choices)),
                                      u.completion_tokens // max(1, len(r.choices)), rt, time.time() - t0,
                                      c.finish_reason or "", lp))
        return out


class FakeBackend(Backend):
    """Deterministic backend for tests: replies with fn(messages, i)."""
    supports_n = True

    def __init__(self, fn):
        self.fn = fn

    def complete(self, messages, *, n, temperature, seed, max_tokens, stop, effort, logprobs):
        return [Completion(self.fn(messages, i), 10, 10) for i in range(n)]


def make_backend(cfg: dict) -> Backend:
    b = cfg["backend"]
    if b == "anthropic":
        return AnthropicBackend(cfg["model"])
    if b == "fireworks":
        return OpenAICompatBackend(cfg["model"], "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY")
    if b == "vercel":
        return OpenAICompatBackend(cfg["model"], "https://ai-gateway.vercel.sh/v1", "AI_GATEWAY_API_KEY",
                                   provider=cfg.get("provider"), closed=True)
    if b == "local":
        from .whitebox import LocalBackend
        return LocalBackend(cfg["hf_id"], adapter=cfg.get("adapter"))
    raise ValueError(b)


class LLM:
    """A model plus its sampling defaults; every call is charged to the current scope and logged."""

    def __init__(self, key: str, backend: Backend | None = None, ledger: Ledger | None = None, cfg: dict | None = None):
        self.key = key
        self.cfg = cfg or model_cfg(key)
        self.backend = backend or make_backend(self.cfg)
        self.ledger = ledger or Ledger(None)
        s = sampling_cfg()
        self.temperature = s["open"]["temperature"]
        self.bon_temperature = s["open"]["best_of_n_temperature"]
        self.max_tokens = s["max_tokens"]
        self.price = self.cfg.get("price", {"input": 0, "output": 0})

    @property
    def closed(self) -> bool:
        return not self.backend.supports_temperature

    def with_backend(self, backend: Backend, key: str | None = None) -> "LLM":
        return LLM(key or self.key, backend, self.ledger, self.cfg)

    def chat(self, messages: list[Message] | str, *, n: int = 1, temperature: float | None = None, role: str = "gen",
             max_tokens: int | None = None, stop: list[str] | None = None, effort: str | None = None,
             logprobs: int | None = None, seed: int | None = None) -> list[Completion]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        sc = current_scope()
        sc.budget.charge(n)
        temp = self.temperature if temperature is None else temperature
        seed = sc.seed if seed is None else seed
        outs = self.backend.complete(messages, n=n, temperature=temp, seed=seed, max_tokens=max_tokens or self.max_tokens,
                                     stop=stop, effort=effort or self.cfg.get("effort"), logprobs=logprobs)
        for c in outs:
            cost = (c.input_tokens * self.price["input"] + c.output_tokens * self.price["output"]) / 1e6
            self.ledger.write({"t": time.time(), "model": self.key, "task": sc.task_id, "technique": sc.technique,
                               "seed": sc.seed, "role": role, "in": c.input_tokens, "out": c.output_tokens,
                               "reasoning": c.reasoning_tokens, "latency": round(c.latency, 3), "cost": cost,
                               "finish": c.finish_reason})
        return outs

    def ask(self, messages: list[Message] | str, **kw) -> str:
        return self.chat(messages, **kw)[0].text

    def sample(self, messages: list[Message] | str, n: int, **kw) -> list[str]:
        kw.setdefault("temperature", self.bon_temperature)
        return [c.text for c in self.chat(messages, n=n, **kw)]


_FENCE = re.compile(r"```[ \t]*([\w+#-]*)[^\n]*\n(.*?)```", re.S)


def extract_code(text: str, lang: str = "python") -> str:
    blocks = _FENCE.findall(text or "")
    if not blocks:
        return (text or "").strip()
    same = [b for l, b in blocks if l.lower() in {lang, lang[:2], ""} or lang.startswith(l.lower())]
    return max(same or [b for _, b in blocks], key=len).strip()
