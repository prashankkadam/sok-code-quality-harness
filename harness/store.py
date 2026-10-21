"""Results layout: runs/<experiment>/<model>/<technique>/B<b>/seed<s>/<benchmark>.jsonl, one line per output with its scores."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import RUNS
from .tasks import Output


def cell_path(exp: str, model: str, tech: str, B: int, seed: int, bench: str) -> Path:
    return RUNS / exp / model / tech / f"B{B}" / f"seed{seed}" / f"{bench}.jsonl"


def done_ids(path: Path) -> set[str]:
    return {json.loads(l)["task_id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()


def append(path: Path, out: Output, score: dict | None = None, quality: dict | None = None, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = asdict(out) | {"score": score or {}, "quality": quality or {}}
    if private:
        rec.update(code="", raw="", trace=[])
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def rewrite(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def cells(exp: str) -> list[tuple[str, str, int, int, str, Path]]:
    out = []
    for p in sorted((RUNS / exp).glob("*/*/B*/seed*/*.jsonl")):
        seed, B, tech, model = p.parent.name, p.parent.parent.name, p.parent.parent.parent.name, p.parents[3].name
        out.append((model, tech, int(B[1:]), int(seed[4:]), p.stem, p))
    return out
