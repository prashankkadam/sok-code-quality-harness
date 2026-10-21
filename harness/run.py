"""Experiment drivers: main grid (E2-E6, E8, E9), extended grid (E10 and its technique and agent tables), and E7."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import store
from .benchmarks.base import Benchmark, get
from .config import CONFIGS, RUNS, grid, load, model_cfg, sampling_cfg
from .llm import LLM, BudgetExceeded, Ledger, make_backend, scope
from .measure.stack import quality
from .tasks import Output, Task
from .techniques.main import TECHNIQUES, TUNED

log = logging.getLogger("harness")
BUDGETS = [1, 4, 8, 16]
INFERENCE = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
LCB_V6_END = "2025-04-30"


def ledger(exp: str) -> Ledger:
    return Ledger(RUNS / exp / "calls.jsonl")


def model(key: str, exp: str, tuned: str | None = None) -> LLM:
    cfg = model_cfg(key)
    if tuned:
        cfg = dict(cfg, model=tuned_ids()[key][tuned])
    return LLM(f"{key}:{tuned}" if tuned else key, make_backend(cfg), ledger(exp), cfg)


def tuned_ids() -> dict:
    p = CONFIGS / "tuned.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def tasks_for(bench: str, key: str) -> list[Task] | None:
    """Clean splits: LiveCodeBench v6 only for models whose cutoff precedes its last problems, restricted to problems after
    the cutoff; LiveCodeBench Pro on contests after 2026-07-01 for every model."""
    cut = model_cfg(key).get("cutoff")
    if bench == "lcb_v6":
        if not cut or cut >= LCB_V6_END[:7]:
            return None
        return get(bench, after=f"{cut}-31").tasks()
    if bench == "lcb_pro":
        return get(bench, after=load("benchmarks")["lcb_pro"]["contests_after"]).tasks()
    return get(bench).tasks()


def generate(exp: str, llm: LLM, tech: str, bench: str, tasks: list[Task], B: int, seed: int, dims: tuple[str, ...],
             workers: int = 16) -> Path:
    path = store.cell_path(exp, llm.key.split(":")[0], tech, B, seed, bench)
    todo = [t for t in tasks if t.id not in store.done_ids(path)]
    impl = TECHNIQUES[tech](dims)

    def one(t: Task) -> Output:
        with scope(t.id, tech, budget=B, seed=seed) as sc:
            out = impl.run(t, llm, B)
            out.calls, out.seed = sc.budget.used, seed
        return out

    with ThreadPoolExecutor(workers) as ex:
        for out in ex.map(one, todo):
            store.append(path, out)
    return path


def score(path: Path, bench: Benchmark, tasks: dict[str, Task], judge: LLM | None, sonar: bool = True) -> None:
    rows = store.read(path)
    if not rows or all(r.get("score") or r.get("quality") for r in rows):
        return
    outs = [Output(**{k: r[k] for k in Output.__dataclass_fields__}) for r in rows]
    scores = bench.score_batch(tasks, outs)
    qual = quality(tasks, outs, judge, sonar)
    for r, s, q in zip(rows, scores, qual):
        r["score"], r["quality"] = s, q
    store.rewrite(path, rows)


def main_grid(models: list[str] | None = None, techniques: list[str] | None = None, budgets: list[int] | None = None,
              benches: list[str] | None = None, seeds: list[int] | None = None, exp: str = "main") -> None:
    cfg = load("benchmarks")["main"]
    benches = benches or [b for d in ("COR", "RM", "EFF", "SEC") for b in cfg[d] if b != "deepswe"]
    seeds = seeds or sampling_cfg()["seeds"]
    judge = model(load("models")["judge"], exp)
    for key in models or grid("main_grid"):
        tunable = model_cfg(key).get("tunable", False)
        for bench in benches:
            ts = tasks_for(bench, key)
            if ts is None:
                continue
            b, tmap = get(bench), {t.id: t for t in ts}
            for tech in techniques or [*INFERENCE, "T8", "T9", "T10"]:
                if tech in TUNED and not tunable:
                    continue
                llm = model(key, exp, TUNED.get(tech))
                for B in [1] if tech == "T0" else (budgets or BUDGETS):
                    for seed in seeds:
                        p = generate(exp, llm, tech, bench, ts, B, seed, b.dims)
                        score(p, b, tmap, judge)


def extended_grid(models: list[str] | None = None, seeds: list[int] | None = None, exp: str = "extended") -> None:
    """Every model's single shot on the function-level stack, then T1-T7 at B=4."""
    benches = ["lcb_pro", "race_r", "race_m", "nofuneval", "evalperf", "mercury", "cweval", "baxbench", "pythonsecurityeval"]
    judge = model(load("models")["judge"], exp)
    for key in models or grid("extended_grid"):
        llm = model(key, exp)
        for bench in benches:
            ts = tasks_for(bench, key)
            b, tmap = get(bench), {t.id: t for t in ts}
            for tech in INFERENCE:
                for seed in seeds or sampling_cfg()["seeds"]:
                    score(generate(exp, llm, tech, bench, ts, 1 if tech == "T0" else 4, seed, b.dims), b, tmap, judge)


def agent_grid(models: list[str] | None = None, seeds: list[int] | None = None, exp: str = "agent") -> None:
    """DeepSWE for every technique on all thirteen models (B counts agent runs); the other agent benchmarks at T0."""
    from .techniques.agentic import run_agentic
    for key in models or grid("main_grid") + grid("extended_grid"):
        tunable = model_cfg(key).get("tunable", False)
        for bench in load("benchmarks")["extended_agent"]:
            techs = [*INFERENCE, "T8", "T9", "T10"] if bench == "deepswe" else ["T0"]
            for tech in techs:
                if tech in TUNED and not tunable:
                    continue
                for seed in seeds or sampling_cfg()["seeds"]:
                    run_agentic(exp, model(key, exp, TUNED.get(tech)), tech, bench, 1 if tech == "T0" else 4, seed)


def train_main(key: str, exp: str = "train") -> dict:
    """Builds the T8-T10 data from the base model, trains the three LoRA adapters on Fireworks and records the served ids."""
    from .training import data, fireworks as fw, grpo
    root = RUNS / exp / key
    evals = [t for d in ("COR", "RM", "EFF", "SEC") for b in load("benchmarks")["main"][d] if b != "deepswe"
             for t in get(b).tasks()]
    train = data.decontaminate(data.taco_tasks(), evals)
    data.sample(model(key, exp), train, root / "samples.jsonl")
    data.sft_dataset(root / "samples.jsonl", root / "sft.jsonl")
    data.dpo_dataset(root / "samples.jsonl", root / "dpo.jsonl")
    data.rl_dataset(train, root / "rl.jsonl")
    dep = load("training")["deployments"][key]
    ids = {
        "sft": fw.deploy(fw.sft(key, fw.upload_dataset(root / "sft.jsonl", f"{key}-t8-sft"), f"{key}-t8-sft", root / "sft.jsonl"), dep),
        "dpo": fw.deploy(fw.dpo(key, fw.upload_dataset(root / "dpo.jsonl", f"{key}-t9-dpo"), f"{key}-t9-dpo", root / "dpo.jsonl"), dep),
        "grpo": fw.deploy(fw.upload_lora(grpo.train(key, [json.loads(l) for l in open(root / "rl.jsonl")],
                                                     lambda text, row: data.reward(text, row["tests"]), root / "grpo"),
                                          f"{key}-t10-grpo", key), dep),
    }
    allids = tuned_ids() | {key: ids}
    (CONFIGS / "tuned.yaml").write_text(yaml.safe_dump(allids))
    return ids


def h2h(hids: list[str] | None = None, seeds: list[int] | None = None, exp: str = "h2h", key: str = "g31") -> None:
    from .h2h import load_all
    from .h2h.base import CEILING, H2H, Ctx
    load_all()
    llm = model(key, exp)
    judge = model(load("models")["judge"], exp)
    from .training.data import decontaminate, taco_tasks
    evals = [t for b in load("benchmarks")["h2h"] for t in (tasks_for(b, key) or [])]
    ctx = Ctx(llm, key, train_tasks=decontaminate(taco_tasks(), evals), tuned_models=tuned_ids().get(f"{key}-h2h", {}))
    for hid in hids or ["T0", "BoN", *sorted(h for h in H2H if h.startswith("H"))]:
        tech = H2H[hid]()
        tech.prepare(ctx)
        for bench in load("benchmarks")["h2h"]:
            ts = tasks_for(bench, key) or []
            b, tmap = get(bench), {t.id: t for t in ts}
            for seed in seeds or sampling_cfg()["seeds"]:
                path = store.cell_path(exp, key, hid, CEILING, seed, bench)
                done = store.done_ids(path)

                def one(t: Task) -> Output:
                    ctx_t = Ctx(llm, key, ctx.artifacts, ctx.train_tasks, ctx.tuned_models)
                    with scope(t.id, hid, budget=CEILING, seed=seed) as sc:
                        try:
                            out = tech.run(t, ctx_t)
                        except BudgetExceeded:
                            out = Output(t.id, hid, key, ctx_t.last_code, meta={"capped": True})
                        out.calls, out.seed, out.budget = sc.budget.used, seed, CEILING
                    return out

                with ThreadPoolExecutor(1 if tech.whitebox else 16) as ex:
                    for out in ex.map(one, [t for t in ts if t.id not in done]):
                        store.append(path, out)
                score(path, b, tmap, judge)
