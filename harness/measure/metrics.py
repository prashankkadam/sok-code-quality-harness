"""Metric definitions used in Tables E2-E9."""
from __future__ import annotations

from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int = 1) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(results: list[tuple[int, int]], k: int = 1) -> float:
    """results: (samples, correct) per task -> percent."""
    return 100 * float(np.mean([pass_at_k(n, c, k) for n, c in results])) if results else float("nan")


def dps(count: float | None, refs: list[tuple[float, float]]) -> tuple[float, float]:
    """EvalPerf differential performance score (evalplus.evalperf.perf_worker). refs: one (instruction count, cumulative
    score) per reference cluster in dataset order, slowest first. The sample takes the score of the fastest cluster whose
    representative it matches or beats; DPS_norm = 100 * (cluster rank) / n clusters. A correct sample whose profiling
    failed (count None) or that is slower than every cluster scores 0."""
    if count is None:
        return 0.0, 0.0
    for j in range(len(refs) - 1, -1, -1):
        if refs[j][0] is not None and count <= refs[j][0]:
            return float(refs[j][1]), 100 * (j + 1) / len(refs)
    return 0.0, 0.0


def mercury_beyond(runtime: float | None, ref_runtimes: list[float]) -> float:
    """Mercury Beyond: position of the sample's runtime within the range of submitted solutions' runtimes (0 if incorrect)."""
    if runtime is None or not ref_runtimes:
        return 0.0
    lo, hi = min(ref_runtimes), max(ref_runtimes)
    if hi <= lo:
        return 100.0 if runtime <= lo else 0.0
    return 100 * float(np.clip((hi - runtime) / (hi - lo), 0, 1))


def effibenchx_et(runtime: float | None, human: float) -> float:
    """EffiBench-X execution-time score relative to the human-expert solution (0 if incorrect), percent."""
    if runtime is None or runtime <= 0:
        return 0.0
    return 100 * min(1.0, human / runtime)


def effibench_net(runtime: float | None, canonical: float) -> float | None:
    """EffiBench normalized execution time (lower is better), defined on correct outputs only."""
    return None if runtime is None or canonical <= 0 else runtime / canonical


def per_kloc(issues: int, ncloc: float) -> float | None:
    return None if ncloc <= 0 else 1000 * issues / ncloc


def share(flags: list[bool]) -> float:
    return 100 * float(np.mean(flags)) if flags else float("nan")


def quadratic_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    a, b = np.asarray(a) - 1, np.asarray(b) - 1
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()
