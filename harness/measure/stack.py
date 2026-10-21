"""The four-layer measurement stack applied to every output, whatever benchmark it came from."""
from __future__ import annotations

import numpy as np

from ..llm import LLM
from ..tasks import Output, Task
from . import judge as J
from . import sonar
from .static import analyze, cognitive_complexity, loc

JUDGE_FIELDS = ("judge", "judge2")


def quality(tasks: dict[str, Task], outs: list[Output], judges: dict[str, LLM] | None = None, use_sonar: bool = True) -> list[dict]:
    """Per-output quality rows: Sonar maintainability/security issues and ncloc, analyzer findings, pylint score,
    cognitive complexity per function, and one length-controlled score per judge (field -> judge LLM; Python outputs only)."""
    rows = [{} for _ in outs]
    py = [i for i, o in enumerate(outs) if tasks[o.task_id].language == "python" and o.code.strip()]
    if use_sonar and py:
        res = sonar.analyze_outputs({str(i): outs[i].code for i in py})
        for i in py:
            r = res[str(i)]
            rows[i].update(sonar_maint=r["maintainability"], sonar_sec=r["security"], ncloc=r["ncloc"] or loc(outs[i].code))
    for i in py:
        rep = analyze(outs[i].code)
        rows[i].update(findings=len(rep.findings), sec_findings=len(rep.security), weighted=rep.weighted(),
                       pylint_score=rep.pylint_score, cc=cognitive_complexity(outs[i].code), loc=loc(outs[i].code))
        rows[i].setdefault("ncloc", rows[i]["loc"])
    for field, llm in (judges or {}).items():
        judge_scores(tasks, outs, rows, llm, field)
    return rows


def judge_scores(tasks: dict[str, Task], outs: list[Output], rows: list[dict], llm: LLM, field: str) -> None:
    """Adds <field>_raw and the length-controlled <field> to the rows of Python outputs (rows already carry loc)."""
    py = [i for i, o in enumerate(outs) if tasks[o.task_id].language == "python" and o.code.strip() and "loc" in rows[i]]
    if not py:
        return
    raw = np.array([J.judge(llm, tasks[outs[i].task_id].prompt, outs[i].code, key=f"{outs[i].task_id}") or np.nan
                    for i in py], float)
    adj = J.length_control(raw, np.array([rows[i]["loc"] for i in py]), [outs[i].task_id for i in py])
    for k, i in enumerate(py):
        rows[i].update({f"{field}_raw": raw[k], field: adj[k]})
