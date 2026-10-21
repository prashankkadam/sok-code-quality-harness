from __future__ import annotations

from abc import ABC, abstractmethod

from ..tasks import Output, Task

REGISTRY: dict[str, type["Benchmark"]] = {}


def register(cls: type["Benchmark"]) -> type["Benchmark"]:
    REGISTRY[cls.name] = cls
    return cls


def get(name: str, **kw) -> "Benchmark":
    from . import agent, correctness, efficiency, quality, security  # noqa: F401  (populate REGISTRY)
    return REGISTRY[name](**kw)


class Benchmark(ABC):
    """A benchmark loads tasks, scores outputs, and aggregates per-output scores into its reported metrics.

    score_batch returns one dict per output (raw measurements, e.g. {"passed": True, "instructions": 123});
    aggregate turns the rows of one (technique, model, seed) cell into {metric: value}."""

    name: str = ""
    dims: tuple[str, ...] = ()          # subset of ("COR", "RM", "EFF", "SEC")
    metrics: dict[str, bool] = {}       # metric -> higher is better
    language: str = "python"
    mode: str = "function"

    def __init__(self, limit: int | None = None, after: str | None = None, before: str | None = None):
        self.limit, self.after, self.before = limit, after, before

    @abstractmethod
    def load(self) -> list[Task]: ...

    def tasks(self) -> list[Task]:
        ts = self.load()
        if self.after:
            ts = [t for t in ts if t.date and t.date > self.after]
        if self.before:
            ts = [t for t in ts if t.date and t.date <= self.before]
        return ts[: self.limit] if self.limit else ts

    def score(self, task: Task, out: Output) -> dict:
        raise NotImplementedError

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        return [self.score(tasks[o.task_id], o) for o in outs]

    @abstractmethod
    def aggregate(self, rows: list[dict]) -> dict[str, float]: ...
