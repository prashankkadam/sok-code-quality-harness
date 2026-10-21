#!/usr/bin/env python3
"""Forecast tables for the main re-evaluation (E2-E5): uncertainty and budget curves.

  forecast_tables.py        # -> sok/paper/generated/tab_ci.tex, tab_budget.tex

Reads the measured cells of Tables E2-E5 from sok/paper/sections/15_empirical.tex and writes two
tables whose unmeasured cells hold FORECAST RANGES (\\expect{lo}{hi}), never data. \\expect renders grey
in drafts and is a hard error in the submission build, so every range must be replaced by a
measurement before submission. Measured cells are copied unchanged. The rules below are also stated
in the table captions.

Uncertainty (tab_ci): forecast 95% bootstrap half-width of a paired per-task effect against T7.
  proportions:  h = 1.96 * sqrt(2 (1 - rho) p (1 - p) / n),   rho in [0.5, 0.8], p = T7 level
  continuous:   h = 1.96 * sqrt(2 (1 - rho)) * sd / sqrt(n),   sd = cv * level or an absolute sd range
  low end = (rho 0.8, small sd, large n); high end = (rho 0.5, large sd, small n).
  n is the verified task count where the benchmark paper states it, otherwise an assumed range.

Budget curves (tab_budget): primary metric per dimension (HE+ pass@1, Pylint messages per kLOC,
EvalPerf DPS_norm, CWEval func-sec@1) at B in {1, 4, 8, 16}.
  B=4: measured (Tables E2-E5).
  B=1: T0 measured; T3, T4, T5 and T7 equal T0 by construction (no refinement round, one sample).
       T1, T2, T6, T8-T10 (single-call, best-of-B at B=4): v4 - g, with g = T7(B=4) - T0 the best-of-4
       gain of the same model, range v4 - 1.5 g .. v4 - 0.5 g.
  B=8, B=16: between saturation (v4) and log-linear growth in B (v4 + 0.5 d, v4 + 1.0 d, d = v4 - v1).
  Where an anchor is a literature value (another model), the model's own anchor is replaced by the
  median offset over the models where it is measured, and the range is widened by the high-end
  half-width on both sides. Every range is at least as wide as the low-end forecast half-width of
  tab_ci, and is clipped to the metric's scale.
"""
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEC = ROOT / "sok/paper/sections/15_empirical.tex"
GEN = ROOT / "sok/paper/generated"

MODELS = ["Op", "oss", "G31", "G26"]
TECHS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
NAMES = {"T0": "T0 single shot", "T1": "T1 quality instr.", "T2": "T2 SCoT", "T3": "T3 analyzer loop",
         "T4": "T4 Self-Refine", "T5": "T5 coder+reviewer", "T6": "T6 RAG", "T7": "T7 best-of-$n$",
         "T8": "T8 LoRA-SFT", "T9": "T9 DPO", "T10": "T10 RL"}
LOWER = {"C%", "Lint", "CC", "Ins", "B/k"}
PCT = {"HE+", "LCB", "C%", "FS", "SP", "Ins", "ET", "Bey"}
TABLES = {  # label: (metrics per model, models of panel a, models of panel b or None)
    "cor": (["HE+", "LCB"], MODELS, None),
    "rm": (["C%", "Lint", "CC", "J"], ["Op", "oss"], ["G31", "G26"]),
    "eff": (["DPS", "ET", "Bey"], ["Op", "oss"], ["G31", "G26"]),
    "sec": (["FS", "SP", "Ins", "B/k"], ["Op", "oss"], ["G31", "G26"]),
}


def parse():
    """{(tech, model, metric): (value, raw_cell, is_literature)} for every cell of E2-E5."""
    lines = SEC.read_text().split("\n")
    out, cur = {}, None
    for l in lines:
        m = re.search(r"\\label\{tab:emp-(\w+)\}", l)
        if m:
            cur, i = m.group(1), 0
        if cur in TABLES and l.endswith("\\\\") and re.match(r"T\d+ ", l):
            mets, pa, pb = TABLES[cur]
            cells = [c.strip() for c in l[:-2].split(" & ")]
            t = cells[0].split()[0]
            models = pa if (pb is None or i < 8) else pb
            i += 1
            for mi, mo in enumerate(models):
                for ki, k in enumerate(mets):
                    c = cells[1 + mi * len(mets) + ki]
                    v = re.search(r"([0-9]+\.[0-9]+|[0-9]+)", c)
                    if v:
                        out[(t, mo, k)] = (float(v.group(1)), c, "\\lit" in c)
    return out


V = parse()


def own(t, m, k):
    v = V.get((t, m, k))
    return None if v is None or v[2] else v[0]


def med_offset(t, ref, k):
    xs = [own(t, m, k) - own(ref, m, k) for m in MODELS
          if own(t, m, k) is not None and own(ref, m, k) is not None]
    return st.median(xs) if xs else None


# ---------------------------------------------------------------- uncertainty (tab_ci)
CI = [  # metric, dimension, benchmark label, n (lo, hi), n verified?, spread model
    ("HE+", "\\dimCor", "HumanEval+", (164, 164), True, ("p",)),
    ("LCB", "\\dimCor", "LiveCodeBench (dated)", (100, 400), False, ("p",)),
    ("C%", "\\dimRM", "PythonSecurityEval", (470, 470), True, ("p",)),
    ("Lint", "\\dimRM", "RACE R/M, NoFunEval", (300, 600), False, ("cv", 0.8, 1.2)),
    ("CC", "\\dimRM", "RACE R/M, NoFunEval", (300, 600), False, ("cv", 0.6, 1.0)),
    ("J", "\\dimRM", "RACE R/M, NoFunEval", (300, 600), False, ("sd", 0.6, 1.0)),
    ("DPS", "\\dimEff", "EvalPerf", (121, 121), True, ("sd", 15, 25)),
    ("ET", "\\dimEff", "EffiBench-X", (300, 700), False, ("sd", 20, 30)),
    ("Bey", "\\dimEff", "Mercury-eval", (256, 256), True, ("sd", 20, 30)),
    ("FS", "\\dimSec", "CWEval", (119, 119), True, ("p",)),
    ("SP", "\\dimSec", "BaxBench", (392, 392), True, ("p",)),
    ("Ins", "\\dimSec", "PythonSecurityEval", (470, 470), True, ("p",)),
    ("B/k", "\\dimSec", "PythonSecurityEval", (470, 470), True, ("cv", 1.2, 1.8)),
]
CI_LABEL = {"HE+": "HE+ pass@1", "LCB": "LCB pass@1", "C%": "C\\%", "Lint": "Lint/kLOC", "CC": "CC",
            "J": "J (1--5)", "DPS": "DPS$_\\text{norm}$", "ET": "ET", "Bey": "Bey", "FS": "FS",
            "SP": "SP", "Ins": "Ins", "B/k": "B/k"}


def level(m, k):
    v = own("T7", m, k)
    if v is None:
        v = st.median([own("T7", x, k) for x in MODELS if own("T7", x, k) is not None])
    return v


def half_width(m, k):
    _, _, _, (nlo, nhi), _, spread = next(r for r in CI if r[0] == k)
    lv = level(m, k)
    if spread[0] == "p":
        p = lv / 100
        h = lambda rho, n: 1.96 * math.sqrt(2 * (1 - rho) * p * (1 - p) / n) * 100
        return h(0.8, nhi), h(0.5, nlo)
    a, b = (spread[1] * lv, spread[2] * lv) if spread[0] == "cv" else (spread[1], spread[2])
    h = lambda rho, sd, n: 1.96 * math.sqrt(2 * (1 - rho)) * sd / math.sqrt(n)
    return h(0.8, a, nhi), h(0.5, b, nlo)


def fmt(x):
    return f"{x:.1f}" if abs(x) >= 0.995 or x == 0 else f"{x:.2f}"


def expect(lo, hi):
    return f"\\expect{{{fmt(lo)}}}{{{fmt(hi)}}}"


def tab_ci():
    h = [r"\begin{table}[t]",
         r"\caption{\textbf{E8. Uncertainty.} 95\% bootstrap half-width ($\pm$) of the paired per-task "
         r"effect of a technique against T7 at $B=4$, per metric and model. An effect in Tables~\ref{tab:emp-cor}"
         r"--\ref{tab:emp-sec} is called a difference only when its interval excludes zero. "
         r"\textbf{Grey ranges are forecast anchors, not data}: the half-width expected from the benchmark "
         r"size $n$ and the model's T7 level (rules below); each is replaced by the measured half-width.}",
         r"\label{tab:emp-ci}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabularx}{\linewidth}{@{}ll>{\raggedright\arraybackslash}X r*{4}{c}@{}}", r"\toprule",
         r" & Metric & Benchmark & $n$ & Op & oss & G31 & G26\\", r"\midrule"]
    prev = None
    for k, dim, bench, (nlo, nhi), ver, _ in CI:
        n = f"{nlo}" if nlo == nhi else f"{nlo}--{nhi}"
        n += "" if ver else "$^{a}$"
        cells = [expect(*half_width(m, k)) for m in MODELS]
        if prev and prev != dim:
            h.append(r"\addlinespace")
        h.append(f"{dim if dim != prev else ''} & {CI_LABEL[k]} & {bench} & {n} & " + " & ".join(cells) + r"\\")
        prev = dim
    h += [r"\bottomrule", r"\end{tabularx}", r"\par\smallskip\raggedright\scriptsize",
          r"Forecast rule: proportions $h = 1.96\sqrt{2(1-\rho)\,p(1-p)/n}$ with $p$ the T7 level; continuous "
          r"metrics $h = 1.96\sqrt{2(1-\rho)}\,\sigma/\sqrt{n}$. Low end: per-task correlation between techniques "
          r"$\rho = 0.8$, small $\sigma$, large $n$; high end: $\rho = 0.5$, large $\sigma$, small $n$. "
          r"$\sigma$ assumed: Lint $0.8$--$1.2\times$ level, CC $0.6$--$1.0\times$, J $0.6$--$1.0$, DPS $15$--$25$, "
          r"ET and Bey $20$--$30$, B/k $1.2$--$1.8\times$ level. $n$ is the task count stated by the benchmark "
          r"authors (HumanEval \citep{liu2023is}, EvalPerf \citep{liu2024evaluating}, Mercury-eval "
          r"\citep{du2024mercury}, CWEval, all languages \citep{peng2025cweval}, BaxBench \citep{vero2025baxbench}, "
          r"PythonSecurityEval \citep{alrashedy2023can}); $^{a}$assumed range, set by the date filter or the "
          r"pinned release. Units as in Tables~\ref{tab:emp-cor}--\ref{tab:emp-sec}.",
          r"\end{table}"]
    (GEN / "tab_ci.tex").write_text("\n".join(h) + "\n")


# ---------------------------------------------------------------- budget curves (tab_budget)
PRIMARY = [("HE+", "\\dimCor{} HumanEval+ pass@1 (\\%, higher is better)", (0, 100)),
           ("Lint", "\\dimRM{} Pylint messages per kLOC (lower is better)", (0, None)),
           ("DPS", "\\dimEff{} EvalPerf DPS$_\\text{norm}$ (higher is better)", (0, 100)),
           ("FS", "\\dimSec{} CWEval func-sec@1 (\\%, higher is better)", (0, 100))]
BUDGETS = [1, 4, 8, 16]
SAME_AS_T0_AT_1 = {"T0", "T3", "T4", "T5", "T7"}


def anchor(t, m, k):
    """Own B=4 value, or (for a literature cell) T7(m) plus the median offset of t from T7 elsewhere."""
    v = own(t, m, k)
    if v is not None:
        return v
    off, t7 = med_offset(t, "T7", k), own("T7", m, k)
    return None if off is None or t7 is None else t7 + off


def t0_anchor(m, k):
    v = own("T0", m, k)
    if v is not None:
        return v
    return own("T7", m, k) - med_offset("T7", "T0", k)


def rng(a, b, m, k, scale, transferred=False):
    lo, hi = min(a, b), max(a, b)
    w, w_hi = half_width(m, k)
    if hi - lo < w:
        c = (lo + hi) / 2
        lo, hi = c - w / 2, c + w / 2
    if transferred:  # anchor borrowed from other models: add the high-end half-width on both sides
        lo, hi = lo - w_hi, hi + w_hi
    if scale[0] is not None:
        lo = max(lo, scale[0])
    if scale[1] is not None:
        hi = min(hi, scale[1])
    return expect(lo, hi)


def cell(t, m, k, b, scale):
    if t in {"T8", "T9", "T10"} and m in {"Op", "oss"}:
        return "--"
    if t == "T0" and b > 1:
        return "--"
    if b == 4 and t != "T0":
        return V[(t, m, k)][1].replace("$^{\\dagger}$", "") if (t, m, k) in V else "--"
    t0 = t0_anchor(m, k)
    tr = own("T0", m, k) is None or own(t, m, k) is None
    if b == 1 and t in SAME_AS_T0_AT_1:
        if own("T0", m, k) is not None:
            return f"{own('T0', m, k):.2f}"
        w = half_width(m, k)[0]
        return rng(t0 - w / 2, t0 + w / 2, m, k, scale, True)
    v4 = anchor(t, m, k)
    if v4 is None:
        return "--"
    g = own("T7", m, k) - t0
    v1 = t0 if t in SAME_AS_T0_AT_1 else v4 - g
    if b == 1:
        return rng(v4 - 1.5 * g, v4 - 0.5 * g, m, k, scale, tr)
    d = v4 - v1
    return rng(v4, v4 + (0.5 if b == 8 else 1.0) * d, m, k, scale, tr)


def tab_budget():
    h = [r"{\scriptsize\setlength{\tabcolsep}{2.2pt}",
         r"\begin{longtable}{@{}l*{8}{>{\centering\arraybackslash}p{0.098\linewidth}}@{}}",
         r"\caption{\textbf{E9. Budget curves.} Primary metric per dimension at $B\in\{1,4,8,16\}$ call-equivalents; "
         r"T7 is the budget-matched reference and T0 is defined at $B=1$ only. $B=4$ cells are the measurements of "
         r"Tables~\ref{tab:emp-cor}--\ref{tab:emp-sec}; at $B=1$, T3, T4, T5 and T7 equal T0 by construction (no "
         r"refinement round, one sample). \textbf{Grey ranges are forecast anchors, not data}, each to be replaced by "
         r"a measurement: at $B=1$, a single-call technique's $B=4$ value minus T7's best-of-4 gain on the same model "
         r"($g$), from $v_4-1.5g$ to $v_4-0.5g$; at $B=8$ and $B=16$, from saturation ($v_4$) to log-linear growth in "
         r"$B$ ($v_4+0.5d$ and $v_4+d$, $d=v_4-v_1$). Where an anchor is a literature value, the median offset over "
         r"the models where it is measured is used instead and the range is widened by the high-end half-width of Table~\ref{tab:emp-ci} on both sides. Each range is at least as wide as the low-end forecast "
         r"half-width of Table~\ref{tab:emp-ci}. Values with a citation are reported by the cited work under its own "
         r"setting.}\label{tab:emp-budget}\\",
         r"\toprule"]
    for pa, label in [(["Op", "oss"], "(a) API-served models"), (["G31", "G26"], "(b) Gemma pair (fine-tunable)")]:
        if label.startswith("(b)"):
            h.append(r"\midrule")
        h.append(r"\multicolumn{9}{@{}l}{\textbf{" + label + r"}}\\")
        h.append(r" & \multicolumn{4}{c}{" + pa[0] + r"} & \multicolumn{4}{c@{}}{" + pa[1] + r"}\\")
        h.append(r"\cmidrule(lr){2-5}\cmidrule(l){6-9}")
        h.append("Technique & " + " & ".join(f"$B={b}$" for b in BUDGETS * 2) + r"\\")
        for k, title, scale in PRIMARY:
            h.append(r"\midrule\multicolumn{9}{@{}l}{\textit{" + title + r"}}\\")
            for t in TECHS:
                if t in {"T8", "T9", "T10"} and pa[0] == "Op":
                    continue
                cells = [cell(t, m, k, b, scale) for m in pa for b in BUDGETS]
                h.append(NAMES[t] + " & " + " & ".join(cells) + r"\\")
    h += [r"\bottomrule", r"\end{longtable}}"]
    (GEN / "tab_budget.tex").write_text("\n".join(h) + "\n")


if __name__ == "__main__":
    tab_ci()
    tab_budget()
    print("wrote", GEN / "tab_ci.tex", GEN / "tab_budget.tex")
