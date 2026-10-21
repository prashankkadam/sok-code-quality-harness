"""T0-T10 of Table E1(c). Each technique spends at most B call-equivalents per task; every call counts."""
from __future__ import annotations

from ..llm import LLM
from ..tasks import Output, Task
from . import common as C
from .rag import retrieve

TECHNIQUES: dict[str, type["Technique"]] = {}


def technique(cls):
    TECHNIQUES[cls.key] = cls
    return cls


class Technique:
    key = ""
    name = ""

    def __init__(self, dims: tuple[str, ...] = ()):
        self.dims = dims

    def run(self, task: Task, llm: LLM, B: int) -> Output:
        code, trace = self.solve(task, llm, B)
        return Output(task.id, self.key, llm.key, code, budget=B, trace=trace)

    def solve(self, task: Task, llm: LLM, B: int) -> tuple[str, list[dict]]:
        raise NotImplementedError


@technique
class T0(Technique):
    key, name = "T0", "single shot"

    def solve(self, task, llm, B):
        return C.code_of(task, llm.ask(C.messages(task))), []


class _BestOfB(Technique):
    """Single-call techniques spend extra budget as best-of-B with the T7 ranker."""

    def prompt(self, task: Task) -> list:
        raise NotImplementedError

    def solve(self, task, llm, B):
        code, scored = C.best_of(llm, task, self.prompt(task), B)
        return code, [{"public_ok": s.public_ok, "weighted": s.weighted} for s in scored]


@technique
class T1(_BestOfB):
    key, name = "T1", "quality instr."

    def prompt(self, task):
        return C.messages(task, post="\n".join(C.DIM_INSTRUCTION[d] for d in C.target_dims(task, self.dims)))


SCOT = """First write a structured chain of thought for the program: its input and output, then the solving process using only the three basic program structures (sequence, branch, loop), as numbered pseudo-code. Then write the code that implements it.

Format:
Input: ...
Output: ...
1: ...
2: if ...:
3:     ...
4: for ...:
...
Then the final code in one fenced block."""


@technique
class T2(_BestOfB):
    key, name = "T2", "SCoT"

    def prompt(self, task):
        return C.messages(task, post=SCOT)


REVISE = """The code below has these static-analysis findings:
{findings}

Fix every listed finding without changing what the code does. Return the complete revised code in one fenced block.

```{lang}
{code}
```"""


@technique
class T3(Technique):
    """Analyzer loop: Pylint, Bandit and Safety, <=3 findings per round (security first); a revision is kept only if the
    public examples still pass and the number of findings falls."""
    key, name = "T3", "analyzer loop"

    def solve(self, task, llm, B):
        code = C.code_of(task, llm.ask(C.messages(task)))
        rep, trace = C.report(task, code), []
        for _ in range(B - 1):
            top = rep.top(3)
            if not top:
                break
            cand = C.code_of(task, llm.ask(C.messages(task) + [
                {"role": "assistant", "content": f"```{task.language}\n{code}\n```"},
                {"role": "user", "content": REVISE.format(findings="\n".join(f.text() for f in top), lang=task.language,
                                                           code=code)}], role="revise"))
            new = C.report(task, cand)
            keep = len(new.findings) < len(rep.findings) and C.passes_public(task, cand)
            trace.append({"findings": len(rep.findings), "candidate_findings": len(new.findings), "kept": keep})
            if keep:
                code, rep = cand, new
        return code, trace


FEEDBACK = """Review the code below for readability, maintainability, efficiency and security. List concrete, actionable problems, most important first. If there is nothing worth changing, reply exactly: NO ISSUES.

```{lang}
{code}
```"""

REFINE = """Feedback on your code:
{feedback}

Rewrite the code to address the feedback while keeping it correct. Return the complete code in one fenced block."""


@technique
class T4(Technique):
    """Self-Refine: the same model critiques, then revises; floor((B-1)/2) rounds of two calls."""
    key, name = "T4", "Self-Refine"

    def solve(self, task, llm, B):
        hist = C.messages(task)
        code, trace = C.code_of(task, llm.ask(hist)), []
        for _ in range((B - 1) // 2):
            fb = llm.ask(hist + [{"role": "assistant", "content": f"```{task.language}\n{code}\n```"},
                                 {"role": "user", "content": FEEDBACK.format(lang=task.language, code=code)}], role="critic")
            if "NO ISSUES" in fb.upper():
                trace.append({"stop": True})
                break
            hist = hist + [{"role": "assistant", "content": f"```{task.language}\n{code}\n```"},
                           {"role": "user", "content": FEEDBACK.format(lang=task.language, code=code)},
                           {"role": "assistant", "content": fb},
                           {"role": "user", "content": REFINE.format(feedback=fb)}]
            code = C.code_of(task, llm.ask(hist, role="refine")) or code
            trace.append({"feedback_chars": len(fb)})
        return code, trace


REVIEWER = """You are a senior code reviewer. Assess the submitted code against this rubric:
1. Correctness against the task, including edge cases.
2. Readability: names, structure, function size, duplication.
3. Maintainability: modularity, magic values, error handling.
4. Efficiency: algorithm and data-structure choice for the input limits.
5. Security: input validation, injection, unsafe APIs, secrets, cryptography.
For each item give concrete required changes. If no change is required, reply exactly: APPROVE."""


@technique
class T5(Technique):
    """Coder + reviewer: the reviewer has its own context and rubric; rounds as T4."""
    key, name = "T5", "coder+reviewer"

    def solve(self, task, llm, B):
        coder = C.messages(task)
        code, trace = C.code_of(task, llm.ask(coder)), []
        for _ in range((B - 1) // 2):
            review = llm.ask([{"role": "system", "content": REVIEWER},
                              {"role": "user", "content": f"Task:\n{C.task_text(task)}\n\nSubmitted code:\n"
                                                          f"```{task.language}\n{code}\n```"}], role="reviewer")
            if "APPROVE" in review.upper() and len(review) < 200:
                trace.append({"approved": True})
                break
            coder = coder + [{"role": "assistant", "content": f"```{task.language}\n{code}\n```"},
                             {"role": "user", "content": f"Code review:\n{review}\n\nRevise the code to address the review. "
                                                         "Return the complete code in one fenced block."}]
            code = C.code_of(task, llm.ask(coder, role="coder")) or code
            trace.append({"review_chars": len(review)})
        return code, trace


@technique
class T6(_BestOfB):
    """RAG: style-guide passages and secure exemplars retrieved from a corpus disjoint from the test tasks."""
    key, name = "T6", "RAG"

    def prompt(self, task):
        docs = retrieve(task, k=3)
        ctx = "\n\n".join(f"[{d['kind']}: {d['source']}]\n{d['text']}" for d in docs)
        return C.messages(task, pre=f"Relevant guidance and examples:\n\n{ctx}" if ctx else "")


@technique
class T7(_BestOfB):
    """Best-of-n: n=B samples, public-example filter, rank by weighted findings. Reference at each B."""
    key, name = "T7", "best-of-n"

    def prompt(self, task):
        return C.messages(task)


class _Tuned(T7):
    """T8-T10: inference as T7 with the fine-tuned weights (the runner passes the tuned model)."""


@technique
class T8(_Tuned):
    key, name = "T8", "LoRA-SFT"


@technique
class T9(_Tuned):
    key, name = "T9", "DPO"


@technique
class T10(_Tuned):
    key, name = "T10", "RL"


TUNED = {"T8": "sft", "T9": "dpo", "T10": "grpo"}
