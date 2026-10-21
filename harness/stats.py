"""Cell metrics, paired per-task effects with bootstrap half-widths, the trade-off matrix, budget curves, E7 counts and
the agreement of the manual ratings."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as st

from . import store
from .benchmarks.base import get
from .measure.metrics import per_kloc, quadratic_kappa, share


@dataclass(frozen=True)
class Metric:
    label: str
    dim: str
    benches: tuple[str, ...]
    key: str
    higher: bool
    kind: str = "bench"   # bench: Benchmark.aggregate()[key]; quality: computed from measurement-stack rows


MAIN = [
    Metric("HE+", "COR", ("humaneval_plus",), "pass@1", True),
    Metric("MBPP+", "COR", ("mbpp_plus",), "pass@1", True),
    Metric("LCB", "COR", ("lcb_v6",), "pass@1", True),
    Metric("LCBP", "COR", ("lcb_pro",), "pass@1", True),
    Metric("DS", "COR", ("deepswe",), "pass@1", True),
    Metric("C%", "RM", ("pythonsecurityeval",), "maint_share", False, "quality"),
    Metric("Lint", "RM", ("race_r", "race_m", "nofuneval"), "maint_per_kloc", False, "quality"),
    Metric("CC", "RM", ("race_r", "race_m", "nofuneval"), "cc", False, "quality"),
    Metric("J", "RM", ("race_r", "race_m", "nofuneval"), "judge", True, "quality"),
    Metric("DPS", "EFF", ("evalperf",), "dps_norm", True),
    Metric("ET", "EFF", ("effibench_x",), "et", True),
    Metric("Bey", "EFF", ("mercury",), "beyond", True),
    Metric("FS", "SEC", ("cweval",), "func_sec@1", True),
    Metric("SP", "SEC", ("baxbench",), "sec_pass@1", True),
    Metric("Ins", "SEC", ("pythonsecurityeval",), "sec_share", False, "quality"),
    Metric("B/k", "SEC", ("pythonsecurityeval",), "sec_per_kloc", False, "quality"),
]

H2H = [
    Metric("EP", "COR", ("humaneval_plus", "mbpp_plus"), "pass@1", True),
    Metric("LCB", "COR", ("lcb_v6",), "pass@1", True),
    Metric("LCBP", "COR", ("lcb_pro",), "pass@1", True),
    Metric("CWE", "SEC", ("cweval",), "func_sec@1", True),
    Metric("CSE", "SEC", ("cyberseceval",), "insecure_rate", False),
    Metric("EB", "EFF", ("effibench",), "net", False),
    Metric("DPS", "EFF", ("evalperf",), "dps_norm", True),
    Metric("RM", "RM", ("humaneval_plus", "mbpp_plus", "lcb_v6"), "pylint_score", True, "quality"),
]


def quality_value(key: str, rows: list[dict]) -> float:
    q = [r["quality"] for r in rows if r.get("quality")]
    if not q:
        return float("nan")
    if key == "maint_share":
        return share([x.get("sonar_maint", 0) > 0 for x in q])
    if key == "sec_share":
        return share([x.get("sonar_sec", 0) > 0 for x in q])
    if key == "maint_per_kloc":
        return per_kloc(sum(x.get("sonar_maint", 0) for x in q), sum(x.get("ncloc", 0) for x in q)) or float("nan")
    if key == "sec_per_kloc":
        return per_kloc(sum(x.get("sonar_sec", 0) for x in q), sum(x.get("ncloc", 0) for x in q)) or float("nan")
    vals = [x[key] for x in q if x.get(key) is not None and not np.isnan(x[key])]
    return float(np.mean(vals)) if vals else float("nan")


def value(m: Metric, rows_by_bench: dict[str, list[dict]]) -> float:
    rows = [r for b in m.benches for r in rows_by_bench.get(b, [])]
    if not rows:
        return float("nan")
    if m.kind == "quality":
        return quality_value(m.key, rows)
    vals, w = [], []
    for b in m.benches:
        rb = rows_by_bench.get(b, [])
        if rb:
            vals.append(get(b).aggregate([r["score"] for r in rb]).get(m.key, float("nan")))
            w.append(len({r["task_id"] for r in rb}))
    return float(np.average(vals, weights=w)) if vals else float("nan")


def per_task(m: Metric, rows_by_bench: dict[str, list[dict]]) -> dict[str, float]:
    by = defaultdict(lambda: defaultdict(list))
    for b in m.benches:
        for r in rows_by_bench.get(b, []):
            by[f"{b}/{r['task_id']}"][b].append(r)
    return {t: value(m, rb) for t, rb in by.items()}


class Results:
    """All cells of one experiment; values are means over seeds."""

    def __init__(self, exp: str):
        self.exp = exp
        self.rows: dict[tuple, list[dict]] = defaultdict(list)
        for model, tech, B, seed, bench, p in store.cells(exp):
            self.rows[(model, tech, B, seed, bench)] = store.read(p)
        self.seeds = sorted({k[3] for k in self.rows})

    def bench_rows(self, model: str, tech: str, B: int, seed: int) -> dict[str, list[dict]]:
        return {k[4]: v for k, v in self.rows.items() if k[:4] == (model, tech, B, seed)}

    def cell(self, m: Metric, model: str, tech: str, B: int) -> float:
        vals = [value(m, self.bench_rows(model, tech, B, s)) for s in self.seeds]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def tasks(self, m: Metric, model: str, tech: str, B: int) -> dict[str, float]:
        acc = defaultdict(list)
        for s in self.seeds:
            for t, v in per_task(m, self.bench_rows(model, tech, B, s)).items():
                if not np.isnan(v):
                    acc[t].append(v)
        return {t: float(np.mean(v)) for t, v in acc.items()}

    def calls(self, model: str, tech: str, B: int) -> float:
        c = [r["calls"] for k, rows in self.rows.items() if k[:3] == (model, tech, B) for r in rows]
        return float(np.mean(c)) if c else float("nan")


def bootstrap_halfwidth(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> float:
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float((hi - lo) / 2)


def paired(a: dict[str, float], b: dict[str, float]) -> list[float]:
    return [a[t] - b[t] for t in a.keys() & b.keys()]


def halfwidths(R: Results, models: list[str], techs: list[str], B: int = 4, metrics=MAIN) -> dict[tuple[str, str], float]:
    """Table E8: 95% half-width of the paired per-task effect against T7 at B, pooled over techniques, per metric and model."""
    out = {}
    for m in metrics:
        for model in models:
            ref = R.tasks(m, model, "T7", B)
            hws = [bootstrap_halfwidth(paired(R.tasks(m, model, t, B), ref)) for t in techs if t != "T7"]
            hws = [h for h in hws if not np.isnan(h)]
            out[(m.label, model)] = float(np.median(hws)) if hws else float("nan")
    return out


def tradeoff(R: Results, models: list[str], techs: list[str], hw: dict, ref: str, targets: dict[str, set[str]],
             metrics=MAIN) -> dict[tuple[str, str], tuple[int, int, int]]:
    """Table E6: per technique and dimension, (better, worse, compared) metric-model cells beyond the half-width.
    ref='T0' compares B=4 techniques with the single shot at B=1; ref='T7' with best-of-4."""
    out = {}
    for t in techs:
        for dim in ("COR", "RM", "EFF", "SEC"):
            better = worse = n = 0
            for m in [m for m in metrics if m.dim == dim]:
                for model in models:
                    a, b = R.cell(m, model, t, 4), R.cell(m, model, ref, 1 if ref == "T0" else 4)
                    h = hw.get((m.label, model), float("nan"))
                    if any(np.isnan(x) for x in (a, b, h)):
                        continue
                    n += 1
                    d = (a - b) if m.higher else (b - a)
                    better += d > h
                    worse += d < -h
            out[(t, dim)] = (better, worse, n)
    return out


def budget_curves(R: Results, models: list[str], techs: list[str], metrics=MAIN) -> dict[tuple[str, str, str, int], float]:
    """Table E9: primary metric per dimension at B in {1,4,8,16}; T3, T4, T5 and T7 at B=1 equal T0 by construction."""
    primary = {"COR": "HE+", "RM": "Lint", "EFF": "DPS", "SEC": "FS"}
    out = {}
    for m in [m for m in metrics if primary.get(m.dim) == m.label]:
        for model in models:
            for t in techs:
                for B in (1, 4, 8, 16):
                    src = "T0" if B == 1 and t in {"T0", "T3", "T4", "T5", "T7"} else t
                    out[(m.label, model, t, B)] = R.cell(m, model, src, 1 if src == "T0" else B)
    return out


def h2h_summary(R: Results, techs: dict[str, tuple[str, tuple[str, ...]]], model: str = "g31", B: int = 16) -> dict:
    """E7 counts: techniques that improve every target metric against T0 and against best-of-n, per family.
    techs: hid -> (family, target dims)."""
    col = {m.label: m for m in H2H}
    tgt = {"RM": ["RM"], "EFF": ["EB", "DPS"], "SEC": ["CWE", "CSE"]}
    res = {"T0": defaultdict(lambda: [0, 0]), "BoN": defaultdict(lambda: [0, 0])}
    for hid, (fam, dims) in techs.items():
        labels = [l for d in dims for l in tgt.get(d, [])]
        for ref in ("T0", "BoN"):
            ok = []
            for l in labels:
                m = col[l]
                a, b = R.cell(m, model, hid, B), R.cell(m, model, ref, B)
                ok.append((a > b) if m.higher else (a < b))
            res[ref][fam][1] += 1
            res[ref][fam][0] += bool(ok) and all(ok)
    return {k: dict(v) for k, v in res.items()}


def correlation(xs: list[float], ys: list[float]) -> float:
    ok = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
    return float(st.pearsonr(*zip(*ok))[0]) if len(ok) > 2 else float("nan")


def draw_manual_sample(exp: str, models: list[str], per_model: int = 110, seed: int = 0, out: Path | None = None) -> Path:
    """Stratified over techniques and benchmarks, drawn from the generated code only (before any score is inspected)."""
    rng = random.Random(seed)
    out = out or store.RUNS / exp / "manual_sample.csv"
    rows = []
    for model in models:
        strata = defaultdict(list)
        for m, tech, B, s, bench, p in store.cells(exp):
            if m == model and B in (1, 4) and bench not in {"deepswe", "lcb_pro"}:
                for r in store.read(p):
                    if r["code"].strip():
                        strata[(tech, bench)].append((m, tech, B, s, bench, r["task_id"], r["code"]))
        keys, i = sorted(strata), 0
        while len(rows) < per_model * (models.index(model) + 1) and any(strata.values()):
            k = keys[i % len(keys)]
            i += 1
            if strata[k]:
                rows.append(strata[k].pop(rng.randrange(len(strata[k]))))
    rng.shuffle(rows)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "model", "technique", "B", "seed", "benchmark", "task_id", "code", "rating"])
        for i, r in enumerate(rows):
            w.writerow([i, *r, ""])
    return out


def manual_agreement(exp: str, rated: Path) -> dict[str, float]:
    """Quadratic-weighted kappa and Spearman rho of the 1-5 manual ratings with the judge, and rho with the maintainability issue count."""
    idx = {}
    for model, tech, B, s, bench, p in store.cells(exp):
        for r in store.read(p):
            idx[(model, tech, str(B), str(s), bench, r["task_id"])] = r["quality"]
    man, jud, iss = [], [], []
    for r in csv.DictReader(open(rated)):
        q = idx.get((r["model"], r["technique"], r["B"], r["seed"], r["benchmark"], r["task_id"]))
        if q and r["rating"] and q.get("judge") is not None:
            man.append(int(r["rating"]))
            jud.append(q["judge"])
            iss.append(q.get("sonar_maint", 0))
    return {"n": len(man), "kappa_qw": quadratic_kappa(man, [int(round(j)) for j in jud]),
            "rho_judge": float(st.spearmanr(man, jud)[0]), "rho_issues": float(st.spearmanr(man, iss)[0])}
