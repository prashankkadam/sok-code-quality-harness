"""Layer 3: length-controlled readability-and-maintainability judges (Claude Opus 5.5 and, as the non-Anthropic check, GPT-6.1 Sol)."""
from __future__ import annotations

import re

import numpy as np

from ..llm import LLM, scope

RUBRIC = """You rate the readability and maintainability of a code solution on a 1-5 scale. Ignore whether it is correct and ignore its length as such; judge only how easy it is to read, understand and change.

5 = clear names, small focused functions, no duplication, idiomatic, comments only where intent is not obvious.
4 = easy to follow; minor naming, structure or duplication issues.
3 = understandable with effort; some long or nested functions, unclear names or repeated logic.
2 = hard to follow; tangled control flow, magic values, misleading names or heavy duplication.
1 = very hard to read or change; obfuscated, monolithic or inconsistent.

Task:
{task}

Solution:
```{lang}
{code}
```

Reply with one line: SCORE: <1-5>"""


def judge(llm: LLM, task_prompt: str, code: str, lang: str = "python", key: str = "") -> float | None:
    with scope(key, "judge", None):
        txt = llm.ask(RUBRIC.format(task=task_prompt[:6000], code=code, lang=lang), role="judge", effort="medium",
                      max_tokens=4096)
    m = re.search(r"SCORE:\s*([1-5](?:\.\d+)?)", txt)
    return float(m.group(1)) if m else None


def length_control(scores: np.ndarray, lengths: np.ndarray, task_ids: list[str]) -> np.ndarray:
    """Removes the length effect: regress score on standardized log length with task fixed effects, then zero the length term."""
    s, L = np.asarray(scores, float), np.log1p(np.asarray(lengths, float))
    ok = ~np.isnan(s)
    z = (L - L[ok].mean()) / (L[ok].std() or 1.0)
    tasks = {t: i for i, t in enumerate(sorted(set(task_ids)))}
    X = np.zeros((len(s), len(tasks) + 1))
    for i, t in enumerate(task_ids):
        X[i, tasks[t]] = 1.0
    X[:, -1] = z
    beta, *_ = np.linalg.lstsq(X[ok], s[ok], rcond=None)
    out = s - beta[-1] * z
    return np.clip(out, 1.0, 5.0)
