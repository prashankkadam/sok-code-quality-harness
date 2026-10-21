from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Test:
    """kind: 'assert' (Python code run after the solution), 'io' (stdin -> expected stdout), or 'call' (args -> expected)."""
    kind: str
    code: str = ""
    stdin: str = ""
    expected: Any = None


@dataclass
class Task:
    id: str
    benchmark: str
    prompt: str
    language: str = "python"
    mode: str = "function"          # function | stdio | edit | repo
    entry_point: str | None = None
    public_tests: list[Test] = field(default_factory=list)
    hidden_tests: list[Test] = field(default_factory=list)
    reference: str | None = None
    source: str | None = None       # code to edit (edit tasks)
    date: str | None = None
    time_limit: float = 10.0
    memory_mb: int = 1024
    meta: dict = field(default_factory=dict)

    @property
    def target(self) -> str:
        return self.meta.get("target_dim", "")


@dataclass
class Output:
    task_id: str
    technique: str
    model: str
    code: str
    raw: str = ""
    calls: int = 0
    seed: int = 0
    budget: int = 1
    trace: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
