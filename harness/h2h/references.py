from __future__ import annotations

from ..llm import current_scope
from ..techniques import common as C
from ..tasks import Task
from .base import Ctx, H2HTechnique, h2h

BON_N = 4


@h2h
class SingleShot(H2HTechnique):
    hid, name = "T0", "T0 single shot (reference)"

    def solve(self, task: Task, ctx: Ctx) -> str:
        return C.code_of(task, ctx.llm.ask(C.messages(task)))


@h2h
class BestOfN(H2HTechnique):
    """best-of-n with the T7 ranker; a sample with no extractable code is redrawn once while the ceiling allows."""
    hid, name = "BoN", "best-of-n (reference)"
    n = BON_N

    def solve(self, task: Task, ctx: Ctx) -> str:
        msgs = C.messages(task)
        codes = [C.code_of(task, t) for t in ctx.llm.sample(msgs, self.n)]
        ctx.keep(C.rank(task, codes)[0])
        for i, c in enumerate(codes):
            if (not c.strip() or (task.language == "python" and not _parses(c))) and current_scope().budget.left > 0:
                codes[i] = C.code_of(task, ctx.llm.sample(msgs, 1)[0])
        return C.rank(task, codes)[0]


def matched(family: str, k: int) -> BestOfN:
    """Budget-matched reference: best-of-k under the id BoN-<family>."""
    ref = BestOfN()
    ref.hid, ref.name, ref.n = f"BoN-{family}", rf"best-of-{k} (matched to {family})", k
    return ref


def _parses(code: str) -> bool:
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
