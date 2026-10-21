from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLM, Message, extract_code
from ..measure.static import Report, analyze
from ..sandbox import passes_public
from ..tasks import Task

SYSTEM = "You are an expert software engineer. Reply with the complete solution in one fenced code block."

LANG_NAME = {"python": "Python 3", "cpp": "C++20", "java": "Java 21", "javascript": "JavaScript (Node.js)", "ruby": "Ruby",
             "go": "Go"}


def task_text(task: Task) -> str:
    lang = LANG_NAME.get(task.language, task.language)
    if task.mode == "function":
        return (f"Complete the following {lang} function. Return the whole function (with any imports and helpers) "
                f"in one ```{task.language} block.\n\n{task.prompt}")
    if task.mode == "stdio":
        return (f"Solve the following problem in {lang}. Read from standard input and write to standard output. "
                f"Return the whole program in one ```{task.language} block.\n\n{task.prompt}")
    if task.mode == "edit":
        return (f"{task.prompt}\n\nCode to edit:\n```{task.language}\n{task.source}\n```\n\n"
                f"Return the whole edited code in one ```{task.language} block.")
    return task.prompt


def messages(task: Task, *, pre: str = "", post: str = "", system: str = SYSTEM) -> list[Message]:
    body = "\n\n".join(x for x in (pre, task_text(task), post) if x)
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def code_of(task: Task, text: str) -> str:
    return extract_code(text, task.language)


def report(task: Task, code: str) -> Report:
    return analyze(code) if task.language == "python" and code.strip() else Report()


@dataclass
class Scored:
    code: str
    public_ok: bool
    weighted: float
    n_findings: int


def score_candidate(task: Task, code: str) -> Scored:
    r = report(task, code)
    return Scored(code, bool(code.strip()) and passes_public(task, code), r.weighted(), len(r.findings))


def rank(task: Task, codes: list[str]) -> tuple[str, list[Scored]]:
    """T7 ranker: keep candidates that pass the public examples (all, if none does), then fewest weighted findings."""
    scored = [score_candidate(task, c) for c in codes]
    pool = [s for s in scored if s.public_ok] or [s for s in scored if s.code.strip()] or scored
    best = min(pool, key=lambda s: (s.weighted, scored.index(s)))
    return best.code, scored


def best_of(llm: LLM, task: Task, msgs: list[Message], n: int) -> tuple[str, list[Scored]]:
    if n <= 1:
        return code_of(task, llm.ask(msgs)), []
    return rank(task, [code_of(task, t) for t in llm.sample(msgs, n)])


DIM_INSTRUCTION = {
    "RM": "Write readable and maintainable code: descriptive names, small single-purpose functions, no duplicated logic, "
          "no magic values, and comments only where the intent is not obvious.",
    "EFF": "Write the most efficient correct implementation you can: choose the algorithm and data structures with the best "
           "time and memory complexity for the input limits, and avoid redundant work.",
    "SEC": "Write secure code: validate and sanitize all inputs, avoid injection (SQL, shell, path), unsafe deserialization, "
           "weak or home-made cryptography, hard-coded secrets and insecure defaults, and handle errors safely.",
}


def target_dims(task: Task, default: tuple[str, ...]) -> list[str]:
    t = task.meta.get("target_dim")
    dims = [t] if t in DIM_INSTRUCTION else [d for d in default if d in DIM_INSTRUCTION]
    return dims or list(DIM_INSTRUCTION)
