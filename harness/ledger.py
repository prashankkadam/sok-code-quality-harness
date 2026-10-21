"""Compute ledger: calls, tokens and dollars per experiment and provider from runs/<exp>/calls.jsonl, plus fine-tuning
dollars and GPU-hours from runs/train/*/jobs.jsonl. Judge calls (role "judge") are counted under the judge row whatever
experiment made them; E12 training jobs (names starting with "e7-" or adapters under the h2h data) under h2h."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import RUNS, load

EXPS = ["main", "extended", "agent", "h2h", "train", "judge", "validation"]
PROVIDERS = {"anthropic": "Claude API", "vercel": "Vercel AI Gateway", "fireworks": "Fireworks inference"}
COLUMNS = ["calls", "in", "out", "reasoning", *PROVIDERS.values(), "ft_usd", "gpu_hours"]


def _provider(model: str) -> str | None:
    cfg = load("models")["models"].get(model.split(":")[0])
    return PROVIDERS.get(cfg["backend"]) if cfg else None


def _lines(path: Path):
    if path.exists():
        for line in open(path):
            if line.strip():
                yield json.loads(line)


def _job_exp(job: dict) -> str:
    name = f"{job.get('output', '')} {job.get('job', '')}"
    return "h2h" if "/e7-" in name or name.startswith("e7-") or "/h2h/" in name else "train"


def aggregate(runs: Path = RUNS) -> dict[str, dict[str, float]]:
    out = {e: defaultdict(float) for e in EXPS}
    for exp in EXPS:
        for r in _lines(runs / exp / "calls.jsonl"):
            row = out["judge" if r.get("role") == "judge" else exp]
            if "calls" in r:
                row["calls"] += r["calls"]
                continue
            row["calls"] += 1
            for k in ("in", "out", "reasoning"):
                row[k] += r.get(k) or 0
            p = _provider(r.get("model", ""))
            if p:
                row[p] += r.get("cost") or 0.0
    for p in sorted(runs.glob("train/*/jobs.jsonl")):
        for j in _lines(p):
            row = out[_job_exp(j)]
            row["ft_usd"] += j.get("cost") or 0.0
            row["gpu_hours"] += j.get("gpu_hours") or 0.0
    return {e: dict(v) for e, v in out.items()}


def table(agg: dict[str, dict[str, float]]) -> str:
    """LaTeX body: one row per experiment and a total row; tokens in millions, dollars and GPU-hours rounded."""
    def cells(v: dict) -> str:
        c = [f"{v.get('calls', 0):,.0f}"] + [f"{v.get(k, 0) / 1e6:,.1f}" for k in ("in", "out", "reasoning")]
        c += [f"{v.get(k, 0):,.0f}" for k in (*PROVIDERS.values(), "ft_usd")] + [f"{v.get('gpu_hours', 0):,.1f}"]
        return " & ".join(c)
    total = {k: sum(v.get(k, 0) for v in agg.values()) for k in COLUMNS}
    rows = [f"{e} & {cells(v)}" + r"\\" for e, v in agg.items() if v]
    return "\n".join(rows + [r"\midrule", rf"Total & {cells(total)}" + r"\\"])


def write(out: Path | None = None) -> Path:
    out = out or RUNS / "tables" / "tab_ledger.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    agg = aggregate()
    out.write_text(table(agg))
    out.with_suffix(".json").write_text(json.dumps(agg, indent=1))
    return out
