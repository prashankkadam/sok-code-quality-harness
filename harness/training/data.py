"""Training data for T8-T10: prompts with tests from a corpus disjoint from every evaluation task, solutions sampled
from the base model itself, labelled by tests and the T3/T7 analyzers."""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..llm import LLM, scope
from ..measure.static import analyze
from ..sandbox import passes
from ..tasks import Task, Test
from ..techniques import common as C
from ..techniques.rag import ngrams

SAMPLES_PER_PROMPT = 8
MAX_PROMPTS = 6000


def finding_free(code: str) -> bool:
    """No security finding (Bandit, Safety) and no Pylint fatal, error or warning."""
    r = analyze(code)
    return not any(f.security or f.severity in {"fatal", "error", "warning"} for f in r.findings)


def taco_tasks(limit: int = MAX_PROMPTS, seed: int = 0) -> list[Task]:
    """BAAI/TACO train split: Python-solvable problems with input/output tests."""
    from datasets import load_dataset
    ds = load_dataset("BAAI/TACO", split="train", trust_remote_code=True)
    out = []
    for i, r in enumerate(ds):
        io = json.loads(r["input_output"] or "{}")
        ins, outs = io.get("inputs") or [], io.get("outputs") or []
        if not ins or len(ins) != len(outs) or io.get("fn_name"):
            continue
        tests = [Test("io", stdin=a if isinstance(a, str) else "\n".join(a), expected=b if isinstance(b, str) else "\n".join(b))
                 for a, b in zip(ins, outs)]
        out.append(Task(f"taco/{i}", "taco", r["question"], mode="stdio", public_tests=tests[:1], hidden_tests=tests,
                        meta={"difficulty": r.get("difficulty")}))
    random.Random(seed).shuffle(out)
    return out[: limit * 2]


def decontaminate(train: list[Task], evals: list[Task], limit: int = MAX_PROMPTS) -> list[Task]:
    banned: set = set()
    for t in evals:
        banned |= ngrams(t.prompt)
    return [t for t in train if not (ngrams(t.prompt) & banned)][:limit]


def sample(llm: LLM, tasks: list[Task], out: Path, n: int = SAMPLES_PER_PROMPT) -> None:
    """Writes one JSON line per prompt: candidates with test pass rate and analyzer label."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    with open(out, "a") as f:
        for t in tasks:
            if t.id in done:
                continue
            msgs = C.messages(t)
            with scope(t.id, "train-sample", None):
                codes = [C.code_of(t, x) for x in llm.sample(msgs, n)]
            cands = []
            for code in codes:
                k, m = passes(t, code, t.hidden_tests, stop_on_fail=False)
                rep = analyze(code)
                cands.append({"code": code, "pass_rate": k / max(1, m), "weighted": rep.weighted(),
                              "finding_free": finding_free(code)})
            f.write(json.dumps({"id": t.id, "messages": msgs, "candidates": cands}) + "\n")


def _rows(path: Path):
    for l in path.read_text().splitlines():
        if l.strip():
            yield json.loads(l)


def sft_dataset(samples: Path, out: Path) -> int:
    """T8: test-passing, finding-free solutions; at most one per prompt (the one with the lowest weighted score)."""
    n = 0
    with open(out, "w") as f:
        for r in _rows(samples):
            good = [c for c in r["candidates"] if c["pass_rate"] == 1.0 and c["finding_free"]]
            if good:
                c = min(good, key=lambda c: c["weighted"])
                f.write(json.dumps({"messages": r["messages"] + [{"role": "assistant", "content": f"```python\n{c['code']}\n```"}]}) + "\n")
                n += 1
    return n


def dpo_dataset(samples: Path, out: Path) -> int:
    """T9: pairs of passing solutions; the one with fewer weighted findings is preferred."""
    n = 0
    with open(out, "w") as f:
        for r in _rows(samples):
            ok = sorted([c for c in r["candidates"] if c["pass_rate"] == 1.0], key=lambda c: c["weighted"])
            if len(ok) >= 2 and ok[0]["weighted"] < ok[-1]["weighted"]:
                f.write(json.dumps({"input": {"messages": r["messages"]},
                                    "preferred_output": [{"role": "assistant", "content": f"```python\n{ok[0]['code']}\n```"}],
                                    "non_preferred_output": [{"role": "assistant", "content": f"```python\n{ok[-1]['code']}\n```"}]}) + "\n")
                n += 1
    return n


def rl_dataset(tasks: list[Task], out: Path) -> int:
    """T10 prompts with their tests; the reward is computed by training.reward."""
    with open(out, "w") as f:
        for t in tasks:
            f.write(json.dumps({"messages": C.messages(t), "task_id": t.id,
                                "tests": [{"stdin": x.stdin, "expected": x.expected} for x in t.hidden_tests]}) + "\n")
    return len(tasks)


def reward(code_text: str, tests: list[dict]) -> float:
    """1 if every test passes and the analyzers find nothing; 0.5 if tests pass with findings; 0 otherwise."""
    t = Task("rl", "rl", "", mode="stdio", hidden_tests=[Test("io", stdin=x["stdin"], expected=x["expected"]) for x in tests])
    code = C.code_of(t, code_text)
    k, m = passes(t, code, t.hidden_tests)
    if m == 0 or k < m:
        return 0.0
    return 1.0 if finding_free(code) else 0.5
