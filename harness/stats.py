"""Cell metrics, paired per-task effects with bootstrap half-widths (at every budget), p-values and a Benjamini-Hochberg
correction, per-seed dispersion, the trade-off matrix, budget curves, E12 counts and intervals, the agreement of the two
judges and of the manual ratings, and the authors' own numbers for the E12 fidelity check."""
from __future__ import annotations

import csv
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import stats as st

from . import store
from .benchmarks.base import get
from .config import CONFIGS, ROOT
from .measure.metrics import per_kloc, quadratic_kappa, share

LIT_H2H = ROOT / "review" / "sok" / "slr" / "top" / "lit_h2h.csv"


@dataclass(frozen=True)
class Metric:
    label: str
    dim: str
    benches: tuple[str, ...]
    key: str
    higher: bool
    kind: str = "bench"   # bench: Benchmark.aggregate()[key]; quality: computed from measurement-stack rows
    secondary: bool = False  # a robustness view of another metric (the non-Anthropic judge); not counted in E6 or the FDR


MAIN = [
    Metric("HE+", "COR", ("humaneval_plus",), "pass@1", True),
    Metric("LCB", "COR", ("lcb_v6",), "pass@1", True),
    Metric("LCBP", "COR", ("lcb_pro",), "pass@1", True),
    Metric("DS", "COR", ("deepswe",), "pass@1", True),
    Metric("C%", "RM", ("pythonsecurityeval",), "maint_share", False, "quality"),
    Metric("Lint", "RM", ("race_r", "race_m", "nofuneval"), "maint_per_kloc", False, "quality"),
    Metric("CC", "RM", ("race_r", "race_m", "nofuneval"), "cc", False, "quality"),
    Metric("J", "RM", ("race_r", "race_m", "nofuneval"), "judge", True, "quality"),
    Metric("Jg", "RM", ("race_r", "race_m", "nofuneval"), "judge2", True, "quality", True),
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

    def seed_values(self, m: Metric, model: str, tech: str, B: int) -> list[float]:
        vals = [value(m, self.bench_rows(model, tech, B, s)) for s in self.seeds]
        return [v for v in vals if not np.isnan(v)]

    def cell(self, m: Metric, model: str, tech: str, B: int) -> float:
        vals = self.seed_values(m, model, tech, B)
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


def _boot_means(diffs: list[float], n_boot: int, seed: int) -> np.ndarray:
    d = np.asarray(diffs, float)
    return d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(_boot_means(diffs, n_boot, seed), [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_halfwidth(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> float:
    lo, hi = bootstrap_ci(diffs, n_boot, seed)
    return (hi - lo) / 2


def bootstrap_p(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for a zero mean difference: twice the smaller share of bootstrap means beyond 0."""
    if len(diffs) < 2:
        return float("nan")
    m = _boot_means(diffs, n_boot, seed)
    return float(min(1.0, 2 * (min((m <= 0).sum(), (m >= 0).sum()) + 1) / (n_boot + 1)))


def paired(a: dict[str, float], b: dict[str, float]) -> list[float]:
    return [a[t] - b[t] for t in a.keys() & b.keys()]


def halfwidths(R: Results, models: list[str], techs: list[str], B: int = 4, metrics=MAIN) -> dict[tuple[str, str], float]:
    """Table E7: 95% half-width of the paired per-task effect against T7 at B, pooled over techniques, per metric and model."""
    out = {}
    for m in metrics:
        for model in models:
            ref = R.tasks(m, model, "T7", B)
            hws = [bootstrap_halfwidth(paired(R.tasks(m, model, t, B), ref)) for t in techs if t != "T7"]
            hws = [h for h in hws if not np.isnan(h)]
            out[(m.label, model)] = float(np.median(hws)) if hws else float("nan")
    return out


def halfwidths_by_budget(R: Results, models: list[str], techs: list[str], budgets=(4, 8, 16), metrics=MAIN) -> dict[int, dict]:
    """Table E7 at every budget: B -> {(metric, model): half-width against T7 at that B}."""
    return {B: halfwidths(R, models, techs, B, metrics) for B in budgets}


def seed_dispersion(R: Results, models: list[str], techs: list[str], budgets=(1, 4, 8, 16), metrics=MAIN
                    ) -> dict[tuple[str, str], tuple[float, float, int]]:
    """Standard deviation across the seeds of every (technique, B) cell, per metric and model as (median, max, cells)."""
    out = {}
    for m in metrics:
        for model in models:
            sds = [float(np.std(v, ddof=1)) for t in techs for B in ([1] if t == "T0" else budgets)
                   if len(v := R.seed_values(m, model, t, B)) > 1]
            out[(m.label, model)] = (float(np.median(sds)), float(np.max(sds)), len(sds)) if sds else (float("nan"),) * 2 + (0,)
    return out


def tradeoff(R: Results, models: list[str], techs: list[str], hw: dict, ref: str, targets: dict[str, set[str]],
             metrics=MAIN) -> dict[tuple[str, str], tuple[int, int, int]]:
    """Table E6: per technique and dimension, (better, worse, compared) metric-model cells beyond the half-width.
    ref='T0' compares B=4 techniques with the single shot at B=1; ref='T7' with best-of-4."""
    out = {}
    for t in techs:
        for dim in ("COR", "RM", "EFF", "SEC"):
            better = worse = n = 0
            for m in [m for m in metrics if m.dim == dim and not m.secondary]:
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


def benjamini_hochberg(ps: list[float], q: float = 0.05) -> list[bool]:
    p = np.nan_to_num(np.asarray(ps, float), nan=1.0)
    if not len(p):
        return []
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    k = int(np.max(np.nonzero(ok)[0])) + 1 if ok.any() else 0
    keep = np.zeros(len(p), bool)
    keep[order[:k]] = True
    return keep.tolist()


def fdr_cells(R: Results, models: list[str], techs: list[str], refs=("T0", "T7"), B: int = 4, q: float = 0.05,
              metrics=MAIN) -> list[dict]:
    """Every metric-model-technique cell of Table E6 (B=4 against T0 at B=1 and against T7 at B=4) with its two-sided paired
    bootstrap p-value; Benjamini-Hochberg at q over all cells jointly marks the differences that survive."""
    cells = []
    for ref in refs:
        for t in techs:
            if t == ref:
                continue
            for m in [m for m in metrics if not m.secondary]:
                for model in models:
                    d = paired(R.tasks(m, model, t, B), R.tasks(m, model, ref, 1 if ref == "T0" else B))
                    if len(d) < 2:
                        continue
                    mean = float(np.mean(d))
                    cells.append({"ref": ref, "technique": t, "metric": m.label, "dim": m.dim, "model": model, "n": len(d),
                                  "diff": mean, "better": mean > 0 if m.higher else mean < 0, "p": bootstrap_p(d)})
    for c, keep in zip(cells, benjamini_hochberg([c["p"] for c in cells], q)):
        c["survives"] = keep
    return cells


def budget_curves(R: Results, models: list[str], techs: list[str], metrics=MAIN) -> dict[tuple[str, str, str, int], float]:
    """Table E8: primary metric per dimension at B in {1,4,8,16}; T3, T4, T5, T7 and T7lo at B=1 equal T0 by construction."""
    primary = {"COR": "HE+", "RM": "Lint", "EFF": "DPS", "SEC": "FS"}
    out = {}
    for m in [m for m in metrics if primary.get(m.dim) == m.label]:
        for model in models:
            for t in techs:
                for B in (1, 4, 8, 16):
                    src = "T0" if B == 1 and t in {"T0", "T3", "T4", "T5", "T7", "T7lo"} else t
                    out[(m.label, model, t, B)] = R.cell(m, model, src, 1 if src == "T0" else B)
    return out


def h2h_summary(R: Results, techs: dict[str, tuple[str, tuple[str, ...]]], model: str = "g31", B: int = 16,
                matched: dict[str, str] | None = None) -> dict:
    """E12 counts: techniques that improve every target metric against T0, best-of-n and the family's budget-matched
    reference (matched: family -> BoN@k or BoN, from run.matched_budgets), per family. techs: hid -> (family, target
    dims). A reference without results is skipped."""
    col = {m.label: m for m in H2H}
    tgt = {"RM": ["RM"], "EFF": ["EB", "DPS"], "SEC": ["CWE", "CSE"]}
    res = {r: defaultdict(lambda: [0, 0]) for r in ("T0", "BoN", "BoN-matched")}
    for hid, (fam, dims) in techs.items():
        labels = [l for d in dims for l in tgt.get(d, [])]
        for ref in res:
            rid = (matched or {}).get(fam, "") if ref == "BoN-matched" else ref
            if not rid:
                continue
            pairs = [(R.cell(col[l], model, hid, B), R.cell(col[l], model, rid, B), col[l].higher) for l in labels]
            if any(np.isnan(b) for _, b, _ in pairs):
                continue
            res[ref][fam][1] += 1
            res[ref][fam][0] += bool(pairs) and all((a > b) if hi else (a < b) for a, b, hi in pairs)
    return {k: dict(v) for k, v in res.items()}


def h2h_intervals(R: Results, hids: list[str], model: str = "g31", B: int = 16, refs=("T0", "BoN"), metrics=H2H
                  ) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """E12: per technique, reference and benchmark column, the mean paired per-task difference (technique minus reference)
    and its 95% bootstrap interval."""
    out = {}
    for ref in refs:
        rt = {m.label: R.tasks(m, model, ref, B) for m in metrics}
        for hid in hids:
            for m in metrics:
                d = paired(R.tasks(m, model, hid, B), rt[m.label])
                out[(hid, ref, m.label)] = (float(np.mean(d)) if d else float("nan"), *bootstrap_ci(d))
    return out


def correlation(xs: list[float], ys: list[float]) -> float:
    ok = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
    return float(st.pearsonr(*zip(*ok))[0]) if len(ok) > 2 else float("nan")


def _num(x) -> float | None:
    return None if x is None or np.isnan(x) else float(x)


def _grade(x: float) -> int:
    return int(min(5, max(1, math.floor(x + 0.5))))


def _agree(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Quadratic-weighted kappa (on 1-5 grades, halves rounded up) and Spearman rho of two paired score lists."""
    if len(xs) < 3:
        return {"n": len(xs), "kappa_qw": float("nan"), "rho": float("nan")}
    return {"n": len(xs), "kappa_qw": float(quadratic_kappa([_grade(x) for x in xs], [_grade(y) for y in ys])),
            "rho": float(st.spearmanr(xs, ys)[0])}


def judge_agreement(exps=("main", "extended", "h2h")) -> dict[str, dict[str, float]]:
    """Agreement of the Anthropic and the non-Anthropic judge over every output both scored: on the raw 1-5 scores and on
    the length-controlled scores."""
    raw, adj = ([], []), ([], [])
    for exp in exps:
        for *_, p in store.cells(exp):
            for r in store.read(p):
                q = r.get("quality") or {}
                for (a, b), suffix in ((raw, "_raw"), (adj, "")):
                    x, y = _num(q.get(f"judge{suffix}")), _num(q.get(f"judge2{suffix}"))
                    if x is not None and y is not None:
                        a.append(x)
                        b.append(y)
    return {"raw": _agree(*raw), "length_controlled": _agree(*adj)}


def draw_manual_sample(exp: str, models: list[str], per_model: int = 110, seed: int = 0, raters=("A", "B"),
                       out_dir: Path | None = None) -> list[Path]:
    """Stratified over techniques and benchmarks, drawn from the generated code only (before any score is inspected).
    One CSV per rater with the same items, each file in its own random order; `item` identifies an output across files."""
    rng = random.Random(seed)
    out_dir = out_dir or store.RUNS / exp
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
    paths = []
    for rater in raters:
        order = list(range(len(rows)))
        random.Random(f"{seed}-{rater}").shuffle(order)
        path = out_dir / f"manual_sample_{rater}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item", "model", "technique", "B", "seed", "benchmark", "task_id", "code", "rating"])
            for i in order:
                w.writerow([i, *rows[i], ""])
        paths.append(path)
    return paths


def manual_agreement(exp: str, rated: list[Path]) -> dict:
    """Inter-rater quadratic-weighted kappa and Spearman rho of the 1-5 manual ratings (every pair of raters), and each
    rater's and the mean rating's agreement with both judges and rho with the maintainability issue count. Only items
    rated by every rater count."""
    idx = {}
    for model, tech, B, s, bench, p in store.cells(exp):
        for r in store.read(p):
            idx[(model, tech, str(B), str(s), bench, r["task_id"])] = r.get("quality") or {}
    names = [Path(p).stem.rsplit("_", 1)[-1] for p in rated]
    ratings, quality = [], {}
    for p in rated:
        got = {}
        for r in csv.DictReader(open(p)):
            if r["rating"].strip():
                got[r["item"]] = int(r["rating"])
                quality[r["item"]] = idx.get((r["model"], r["technique"], r["B"], r["seed"], r["benchmark"], r["task_id"]))
        ratings.append(got)
    items = sorted(set.intersection(*(set(g) for g in ratings)), key=int)
    series = {n: [g[i] for i in items] for n, g in zip(names, ratings)}
    if len(ratings) > 1:
        series["mean"] = [float(np.mean([g[i] for g in ratings])) for i in items]
    out: dict = {"n": len(items), "raters": names}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            out[f"{names[a]}~{names[b]}"] = _agree(series[names[a]], series[names[b]])
    for name, xs in series.items():
        for field, judge in (("judge", "opus"), ("judge2", "gpt61")):
            pairs = [(x, _num((quality[i] or {}).get(field))) for x, i in zip(xs, items)]
            pairs = [(x, j) for x, j in pairs if j is not None]
            out[f"{name}~{judge}"] = _agree(*map(list, zip(*pairs))) if pairs else _agree([], [])
        iss = [(x, (quality[i] or {}).get("sonar_maint")) for x, i in zip(xs, items)]
        iss = [(x, y) for x, y in iss if y is not None]
        out[f"{name}~issues"] = {"n": len(iss), "rho": float(st.spearmanr(*zip(*iss))[0]) if len(iss) > 2 else float("nan")}
    return out


FIDELITY_BENCH = [
    (r"HumanEval\+", "humaneval_plus", "pass@1"),
    (r"HumanEval \(base tests\)", "humaneval_plus", "base@1"),
    (r"MBPP\+", "mbpp_plus", "pass@1"),
    (r"MBPP \(base tests\)", "mbpp_plus", "base@1"),
    (r"LiveCodeBench Pro", "lcb_pro", "pass@1"),
    (r"LiveCodeBench", "lcb_v6", "pass@1"),
    (r"EvalPerf", "evalperf", "dps_norm"),
    (r"EffiBench-X", "effibench_x", "et"),
    (r"EffiBench", "effibench", "net"),
    (r"CWEval", "cweval", "func_sec@1"),
    (r"CyberSecEval", "cyberseceval", "secure_rate"),
    (r"Mercury", "mercury", "beyond"),
    (r"BaxBench", "baxbench", "sec_pass@1"),
]


def lit_value(r: dict) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", r["value"].replace(",", ""))
    if not m:
        return None
    return 100 * float(m.group()) if "fraction" in r["setting"].lower() else float(m.group())


def lit_rows(hid: str | None = None) -> list[dict]:
    """lit_h2h.csv (each technique's number in its authors' own setting) with its old E12 ids mapped to the current ones
    through configs/h2h_renumber.csv; rows of removed techniques drop out."""
    new = {r["old_id"]: r["h2h_id"] for r in csv.DictReader(open(CONFIGS / "h2h_renumber.csv")) if r["old_id"]}
    rows = [r | {"old_id": r["h2h_id"], "h2h_id": new[r["h2h_id"]], "lit": lit_value(r)}
            for r in csv.DictReader(open(LIT_H2H)) if r["h2h_id"] in new]
    return [r for r in rows if hid is None or r["h2h_id"] == hid]


def setting_spec(setting: str, column: str) -> tuple[str, str, dict] | None:
    """(benchmark, metric, benchmark arguments) of an original setting when its benchmark is one of ours, else None."""
    for pat, bench, metric in FIDELITY_BENCH:
        if re.search(pat, setting):
            kw = {}
            d = re.search(r"(\d{4}-\d{2}-\d{2}) to present", setting)
            if d:
                kw["after"] = (date.fromisoformat(d.group(1)) - timedelta(days=1)).isoformat()
            v = re.search(r"\((Instruct|Autocomplete)\)", setting)
            if bench == "cyberseceval" and v:
                kw["variants"] = (v.group(1).lower(),)
            return bench, "calls" if column == "Calls" else metric, kw
    return None
