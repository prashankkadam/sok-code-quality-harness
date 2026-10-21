"""Experiment drivers: main grid (E2-E6, E8, E9), extended grid (E10 and its technique and agent tables), E7, its
budget-matched best-of-n references and the fidelity check."""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from . import store
from .benchmarks.base import Benchmark, get
from .config import CONFIGS, RUNS, grid, load, model_cfg, sampling_cfg
from .llm import LLM, BudgetExceeded, Ledger, make_backend, scope
from .measure.stack import judge_scores, quality
from .tasks import Output, Task
from .techniques.main import TECHNIQUES, TUNED

log = logging.getLogger("harness")
BUDGETS = [1, 4, 8, 16]
INFERENCE = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
CONTROLS = ["T7lo"]
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


def judges(exp: str) -> dict[str, LLM]:
    """The length-controlled judges: Claude Opus 5.5 ("judge") and the non-Anthropic GPT-6.1 Sol ("judge2")."""
    m = load("models")
    return {"judge": model(m["judge"], exp), "judge2": model(m["judge2"], exp)}


def score(path: Path, bench: Benchmark, tasks: dict[str, Task], judge: dict[str, LLM] | None, sonar: bool = True) -> None:
    """Scores a cell once; a scored cell missing a judge's field gets that judge's scores added."""
    rows = store.read(path)
    if not rows:
        return
    outs = [Output(**{k: r[k] for k in Output.__dataclass_fields__}) for r in rows]
    if not all(r.get("score") or r.get("quality") for r in rows):
        scores = bench.score_batch(tasks, outs)
        qual = quality(tasks, outs, judge, sonar)
        for r, s, q in zip(rows, scores, qual):
            r["score"], r["quality"] = s, q
    else:
        missing = {f: llm for f, llm in (judge or {}).items()
                   if any("loc" in (r.get("quality") or {}) and f not in r["quality"] for r in rows)}
        if not missing:
            return
        for r in rows:
            r["quality"] = r.get("quality") or {}
        for f, llm in missing.items():
            judge_scores(tasks, outs, [r["quality"] for r in rows], llm, f)
    store.rewrite(path, rows)


def main_grid(models: list[str] | None = None, techniques: list[str] | None = None, budgets: list[int] | None = None,
              benches: list[str] | None = None, seeds: list[int] | None = None, exp: str = "main") -> None:
    """T7lo (the temperature control) runs on the open models only: closed models sample at their default."""
    cfg = load("benchmarks")["main"]
    benches = benches or [b for d in ("COR", "RM", "EFF", "SEC") for b in cfg[d] if b != "deepswe"]
    seeds = seeds or sampling_cfg()["seeds"]
    judge = judges(exp)
    for key in models or grid("main_grid"):
        tunable = model_cfg(key).get("tunable", False)
        for bench in benches:
            ts = tasks_for(bench, key)
            if ts is None:
                continue
            b, tmap = get(bench), {t.id: t for t in ts}
            for tech in techniques or [*INFERENCE, "T8", "T9", "T10", *CONTROLS]:
                if tech in TUNED and not tunable:
                    continue
                llm = model(key, exp, TUNED.get(tech))
                if tech in CONTROLS and llm.closed:
                    continue
                Bs = [1] if tech == "T0" else [x for x in budgets or BUDGETS if x > 1 or tech not in CONTROLS]
                for B in Bs:
                    for seed in seeds:
                        p = generate(exp, llm, tech, bench, ts, B, seed, b.dims)
                        score(p, b, tmap, judge)


def extended_grid(models: list[str] | None = None, seeds: list[int] | None = None, exp: str = "extended") -> None:
    """Every model's single shot on the function-level stack, then T1-T7 at B=4."""
    benches = ["lcb_pro", "race_r", "race_m", "nofuneval", "evalperf", "mercury", "cweval", "baxbench", "pythonsecurityeval"]
    judge = judges(exp)
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




def _h2h_ctx(key: str, exp: str):
    from .h2h import load_all
    from .h2h.base import Ctx
    from .training.data import decontaminate, taco_tasks
    load_all()
    llm = model(key, exp)
    evals = [t for b in load("benchmarks")["h2h"] for t in (tasks_for(b, key) or [])]
    return Ctx(llm, key, train_tasks=decontaminate(taco_tasks(), evals), tuned_models=tuned_ids().get(f"{key}-h2h", {}))


def _h2h_cell(tech, ctx, exp: str, bench: str, ts: list[Task], seed: int, judge: dict[str, LLM] | None) -> Path:
    from .h2h.base import CEILING, Ctx
    key, llm = ctx.model_key, ctx.llm
    path = store.cell_path(exp, key, tech.hid, CEILING, seed, bench)
    done = store.done_ids(path)

    def one(t: Task) -> Output:
        ctx_t = Ctx(llm, key, ctx.artifacts, ctx.train_tasks, ctx.tuned_models)
        with scope(t.id, tech.hid, budget=CEILING, seed=seed) as sc:
            try:
                out = tech.run(t, ctx_t)
            except BudgetExceeded:
                out = Output(t.id, tech.hid, key, ctx_t.last_code, meta={"capped": True})
            out.calls, out.seed, out.budget = sc.budget.used, seed, CEILING
        return out

    with ThreadPoolExecutor(1 if tech.whitebox else 16) as ex:
        for out in ex.map(one, [t for t in ts if t.id not in done]):
            store.append(path, out)
    score(path, get(bench), {t.id: t for t in ts}, judge, sonar=judge is not None)
    return path


def _h2h_run(tech, ctx, exp: str, seeds: list[int] | None, judge: dict[str, LLM]) -> None:
    tech.prepare(ctx)
    for bench in load("benchmarks")["h2h"]:
        ts = tasks_for(bench, ctx.model_key) or []
        for seed in seeds or sampling_cfg()["seeds"]:
            _h2h_cell(tech, ctx, exp, bench, ts, seed, judge)


def h2h(hids: list[str] | None = None, seeds: list[int] | None = None, exp: str = "h2h", key: str = "g31") -> None:
    from .h2h.base import H2H
    ctx = _h2h_ctx(key, exp)
    judge = judges(exp)
    for hid in hids or ["T0", "BoN", *sorted(h for h in H2H if h.startswith("H"))]:
        _h2h_run(H2H[hid](), ctx, exp, seeds, judge)


def matched_budgets(exp: str = "h2h", key: str = "g31") -> dict[str, int]:
    """Per family: the median over its techniques of the mean calls per task in the stored E7 runs, rounded to the nearest
    integer, at least 2 and at most the ceiling."""
    from .h2h import load_all
    from .h2h.base import CEILING, H2H
    from .stats import Results
    load_all()
    R, fams = Results(exp), defaultdict(list)
    for hid, cls in H2H.items():
        c = R.calls(key, hid, CEILING)
        if hid.startswith("H") and not math.isnan(c):
            fams[cls.family].append(c)
    return {f: min(CEILING, max(2, math.floor(statistics.median(v) + 0.5))) for f, v in fams.items()}


def h2h_bon_matched(families: list[str] | None = None, seeds: list[int] | None = None, exp: str = "h2h",
                    key: str = "g31") -> dict[str, int]:
    """Budget-matched best-of-n: BoN@k with k = the family's median calls per task, stored under the id BoN-<family>."""
    from .h2h.references import matched
    ks = matched_budgets(exp, key)
    ctx = _h2h_ctx(key, exp)
    judge = judges(exp)
    for fam in families or sorted(ks):
        _h2h_run(matched(fam, ks[fam]), ctx, exp, seeds, judge)
    (RUNS / exp / "bon_matched.json").write_text(json.dumps(ks, indent=1))
    return ks


def _fidelity_value(bench: Benchmark, metric: str, rows: list[dict]) -> float:
    if metric == "calls":
        return float(np.mean([r["calls"] for r in rows])) if rows else float("nan")
    sc = [r["score"] for r in rows if r.get("score")]
    if not sc:
        return float("nan")
    if metric == "base@1":
        return 100 * float(np.mean([bool(s.get("base")) for s in sc]))
    agg = bench.aggregate(sc)
    if metric.startswith("secure_rate"):
        return 100 - agg.get(metric.replace("secure_rate", "insecure_rate"), float("nan"))
    return agg.get(metric, float("nan"))


def fidelity_check(hid: str, key: str = "g31", seeds: list[int] | None = None, exp: str = "fidelity") -> Path:
    """Runs one E7 technique in each original setting of lit_h2h.csv that names one of our benchmarks (benchmark, split,
    metric; the model is `key`, the paper's model when it is configured, else the E7 model, and is recorded) and writes
    runs/fidelity/<hid>.json. A setting on a benchmark we do not have is recorded as needing the original harness."""
    from .h2h.base import H2H
    from .stats import lit_rows, setting_spec
    ctx = _h2h_ctx(key, exp)
    tech, res, prepared = H2H[hid](), [], False
    for r in lit_rows(hid):
        rec = {"column": r["column"], "lit": r["lit"], "setting": r["setting"], "cite": r["cite_key"], "model": key,
               "model_name": model_cfg(key)["name"]}
        spec = setting_spec(r["setting"], r["column"])
        if spec is None:
            res.append(rec | {"status": "needs the original harness"})
            continue
        bench, metric, kw = spec
        if not prepared:
            tech.prepare(ctx)
            prepared = True
        b = get(bench, **kw)
        ts = b.tasks()
        vals = [_fidelity_value(b, metric, store.read(_h2h_cell(tech, ctx, exp, bench, ts, seed, None)))
                for seed in seeds or sampling_cfg()["seeds"]]
        res.append(rec | {"status": "ok", "benchmark": bench, "split": kw, "metric": metric, "tasks": len(ts),
                          "per_seed": vals, "ours": float(np.nanmean(vals)) if not all(np.isnan(vals)) else None})
    out = RUNS / exp / f"{hid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hid": hid, "name": tech.name, "rows": res}, indent=1))
    return out
