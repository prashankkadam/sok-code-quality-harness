from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = Path(os.environ.get("HARNESS_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("HARNESS_RUNS", ROOT / "runs"))


@lru_cache
def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def model_cfg(key: str) -> dict:
    return load("models")["models"][key]


def sampling_cfg() -> dict:
    return load("models")["sampling"]


def grid(name: str) -> list[str]:
    return load("models")[name]
