"""E12 head-to-head: one model (Gemma 4 31B IT), each technique in its published configuration, at most 16 calls per task."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from ..config import DATA, model_cfg
from ..llm import LLM, current_scope
from ..tasks import Output, Task

CEILING = 16
H2H: dict[str, type["H2HTechnique"]] = {}


def h2h(cls):
    H2H[cls.hid] = cls
    return cls


@dataclass
class Ctx:
    """What a technique may use. llm: the E12 model behind the API. lm(): the same weights loaded locally (white-box access).
    tuned(name): the LLM serving a LoRA adapter trained by prepare(). Components that call another LLM use `llm`."""
    llm: LLM
    model_key: str = "g31"
    artifacts: Path = DATA / "h2h"
    train_tasks: list[Task] = field(default_factory=list)
    tuned_models: dict[str, str] = field(default_factory=dict)
    last_code: str = ""

    @cached_property
    def hf_id(self) -> str:
        return model_cfg(self.model_key)["hf_id"]

    def lm(self, adapter: str | None = None):
        from ..whitebox import local_lm
        return local_lm(self.hf_id, adapter)

    def charge(self, n: int = 1, role: str = "local") -> None:
        """Counts n locally generated sequences (white-box decoding) as n calls."""
        sc = current_scope()
        sc.budget.charge(n)
        self.llm.ledger.write({"t": time.time(), "model": self.model_key, "task": sc.task_id, "technique": sc.technique,
                               "seed": sc.seed, "role": role, "calls": n})

    def keep(self, code: str) -> str:
        """Records the best code so far; if the ceiling is hit, the runner reports it."""
        self.last_code = code
        return code

    def tuned(self, name: str) -> LLM:
        from ..llm import make_backend
        cfg = dict(self.llm.cfg, model=self.tuned_models[name])
        return LLM(f"{self.model_key}:{name}", make_backend(cfg), self.llm.ledger, cfg)

    def dir(self, hid: str) -> Path:
        p = self.artifacts / hid
        p.mkdir(parents=True, exist_ok=True)
        return p


class H2HTechnique:
    hid = ""
    name = ""
    family = ""                   # prompting | feedback | refinement | multiagent | retrieval | decoding | training
    target: tuple[str, ...] = ()  # dimensions it targets: RM, EFF, SEC
    code_url = ""                 # public code adapted, if any
    ported = False                # defined only for another language and ported to Python
    whitebox = False              # needs logits or activations
    trains = False                # prepare() trains a component or the model

    def prepare(self, ctx: Ctx) -> None:
        """Offline step (retrieval index, trained detector, reward model, LoRA adapter). Not counted in the call budget."""

    def solve(self, task: Task, ctx: Ctx) -> str:
        raise NotImplementedError

    def run(self, task: Task, ctx: Ctx) -> Output:
        code = self.solve(task, ctx)
        return Output(task.id, self.hid, ctx.model_key, code, calls=current_scope().budget.used, budget=CEILING)
