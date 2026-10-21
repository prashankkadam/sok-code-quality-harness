"""T0-T10 acting on the mini-SWE-agent for repository-level benchmarks (Table E2b). B counts complete agent runs; T4 and
T5 critique or review calls also count one unit each, so the round rule of Table E1(c) is unchanged."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .. import store
from ..benchmarks.agent import AgentRun, findings, run_agent, select, visible_tests
from ..benchmarks.base import get
from ..llm import LLM, scope
from ..tasks import Output, Task
from .common import DIM_INSTRUCTION
from .main import FEEDBACK, REVIEWER, SCOT
from .rag import retrieve

Agentic = Callable[[Task, LLM, int, int], AgentRun]


def _passes(task: Task, run: AgentRun) -> bool:
    k, n = visible_tests(task, run.patch)
    return n == 0 or k == n


def _best_of(task: Task, llm: LLM, B: int, seed: int, **kw) -> AgentRun:
    runs = [run_agent(task, llm, seed=seed * 100 + i, **kw) for i in range(B)]
    return runs[0] if B == 1 else select(task, runs)


def t0(task, llm, B, seed):
    return run_agent(task, llm, seed=seed)


def t1(task, llm, B, seed):
    dims = [task.meta["target_dim"]] if task.meta.get("target_dim") in DIM_INSTRUCTION else list(DIM_INSTRUCTION)
    return _best_of(task, llm, B, seed, instructions="\n".join(DIM_INSTRUCTION[d] for d in dims))


def t2(task, llm, B, seed):
    return _best_of(task, llm, B, seed, instructions="Before editing, plan each change as a structured chain of thought "
                                                      "using only sequence, branch and loop structures.\n" + SCOT.split("\n\n")[0])


def t3(task, llm, B, seed):
    run = run_agent(task, llm, seed=seed)
    rep = findings(task, run.patch)
    for i in range(B - 1):
        top = rep.top(3)
        if not top:
            break
        cand = run_agent(task, llm, seed=seed * 100 + i + 1, prior_patch=run.patch,
                         feedback="Static-analysis findings on your changes:\n" + "\n".join(f.text() for f in top))
        new = findings(task, cand.patch)
        if len(new.findings) < len(rep.findings) and _passes(task, cand):
            run, rep = cand, new
    return run


def _diff_prompt(task: Task, run: AgentRun) -> str:
    return f"Task:\n{task.prompt}\n\nProposed change (unified diff):\n```diff\n{run.patch}\n```"


def t4(task, llm, B, seed):
    run = run_agent(task, llm, seed=seed)
    for i in range((B - 1) // 2):
        fb = llm.ask(_diff_prompt(task, run) + "\n\n" + FEEDBACK.split("\n\n")[0], role="critic")
        if "NO ISSUES" in fb.upper():
            break
        run = run_agent(task, llm, seed=seed * 100 + i + 1, prior_patch=run.patch, feedback=fb)
    return run


def t5(task, llm, B, seed):
    run = run_agent(task, llm, seed=seed)
    for i in range((B - 1) // 2):
        review = llm.ask([{"role": "system", "content": REVIEWER}, {"role": "user", "content": _diff_prompt(task, run)}],
                         role="reviewer")
        if "APPROVE" in review.upper() and len(review) < 200:
            break
        run = run_agent(task, llm, seed=seed * 100 + i + 1, prior_patch=run.patch, feedback=review)
    return run


def t6(task, llm, B, seed):
    docs = retrieve(task, k=3)
    return _best_of(task, llm, B, seed, context="\n\n".join(f"[{d['kind']}: {d['source']}]\n{d['text']}" for d in docs))


def t7(task, llm, B, seed):
    return _best_of(task, llm, B, seed)


AGENTIC: dict[str, Agentic] = {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7,
                               "T8": t7, "T9": t7, "T10": t7}


def run_agentic(exp: str, llm: LLM, tech: str, bench: str, B: int, seed: int, workers: int = 8) -> None:
    """Runs and scores in memory; for private benchmarks only the scores reach disk."""
    b, fn = get(bench), AGENTIC[tech]
    private = getattr(b, "private", False)
    path = store.cell_path(exp, llm.key.split(":")[0], tech, B, seed, bench)
    todo = [t for t in b.tasks() if t.id not in store.done_ids(path)]

    def one(t: Task) -> tuple[Task, Output]:
        with scope(t.id, tech, budget=B, seed=seed) as sc:
            run = fn(t, llm, B, seed)
            out = Output(t.id, tech, llm.key, run.patch, calls=sc.budget.used, seed=seed, budget=B,
                         meta={"agent": run.summary()})
        return t, out

    with ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(one, todo))
    if not done:
        return
    scores = b.score_batch({t.id: t for t, _ in done}, [o for _, o in done])
    for (_, o), s in zip(done, scores):
        store.append(path, o, s, private=private)
