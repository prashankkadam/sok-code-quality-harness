"""LaTeX rows for the paper's tables from the stored results."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stats as S
from .config import RUNS, grid, model_cfg
from .techniques.main import TECHNIQUES

ABBR = {"op": "Op", "oss": "oss", "g31": "G31", "g26": "G26"}
TECHS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]


def f(x: float, nd: int = 2) -> str:
    return "--" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def tech_label(t: str) -> str:
    return f"{t} {TECHNIQUES[t].name}"


def main_table(R: S.Results, labels: list[str], models: list[str], B: int = 4) -> str:
    """E2-E5 body: one row per technique, one column per (metric, model)."""
    ms = [m for m in S.MAIN if m.label in labels]
    lines = []
    for t in TECHS:
        cells = [f(R.cell(m, k, t, 1 if t == "T0" else B)) if not (t in {"T8", "T9", "T10"} and not model_cfg(k).get("tunable"))
                 else "--" for k in models for m in ms]
        lines.append(f"{tech_label(t)} & " + " & ".join(cells) + r"\\")
    return "\n".join(lines)


def tradeoff_table(R: S.Results, models: list[str], hw: dict, targets: dict[str, set[str]]) -> str:
    out = []
    for ref in ("T0", "T7"):
        res = S.tradeoff(R, models, TECHS[1:], hw, ref, targets)
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{({'a' if ref == 'T0' else 'b'}) against {ref}}}}}\\")
        for t in TECHS[1:]:
            if ref == "T7" and t == "T7":
                continue
            cells = []
            for d in ("COR", "RM", "EFF", "SEC"):
                b, w, n = res[(t, d)]
                star = r"$\star$" if d in targets.get(t, set()) else ""
                cells.append(f"{b} / {w} [{n}]{star}")
            out.append(f"{tech_label(t)} & " + " & ".join(cells) + r"\\")
    return "\n".join(out)


def ci_table(hw: dict, models: list[str]) -> str:
    return "\n".join(f"{m.dim} & {m.label} & " + " & ".join(f(hw.get((m.label, k), float('nan'))) for k in models) + r"\\"
                     for m in S.MAIN)


def budget_table(curves: dict, models: list[str]) -> str:
    out = []
    for label in ("HE+", "Lint", "DPS", "FS"):
        out.append(rf"\multicolumn{{{1 + 4 * len(models)}}}{{@{{}}l}}{{\textit{{{label}}}}}\\")
        for t in TECHS:
            cells = []
            for k in models:
                for B in (1, 4, 8, 16):
                    v = curves.get((label, k, t, B), float("nan"))
                    cells.append(f(v) if not (t == "T0" and B > 1) else "--")
            out.append(f"{tech_label(t)} & " + " & ".join(cells) + r"\\")
    return "\n".join(out)


def extended_table(R_ext: S.Results, R_main: S.Results) -> str:
    labels = ["LCBP", "Lint", "J", "DPS", "Bey", "FS", "SP", "B/k"]
    ms = [m for m in S.MAIN if m.label in labels]
    rows = []
    for k in grid("extended_grid") + grid("main_grid"):
        R = R_ext if k in grid("extended_grid") else R_main
        cut = model_cfg(k).get("cutoff") or r"\nr"
        rows.append(f"{model_cfg(k)['name']} & {cut} & " + " & ".join(f(R.cell(m, k, 'T0', 1)) for m in ms) + r"\\")
    return "\n".join(rows)


def h2h_table(R: S.Results, techs: dict[str, tuple[str, str]], model: str = "g31", B: int = 16) -> str:
    """E7 body. techs: hid -> (family, name), in table order."""
    def row(hid, name, label):
        return f"{label} & {name} & " + " & ".join(f(R.cell(m, model, hid, B)) for m in S.H2H) + f" & {f(R.calls(model, hid, B))}" + r"\\"
    out = [row("T0", "T0 single shot (reference)", "--"), row("BoN", r"best-of-$n$ (reference)", "--")]
    fam_prev = None
    for hid, (fam, name) in techs.items():
        if fam != fam_prev:
            out.append(rf"\midrule\multicolumn{{11}}{{@{{}}l}}{{\textit{{{fam}}}}}\\")
            fam_prev = fam
        out.append(row(hid, name, hid))
    return "\n".join(out)


def write_all(out: Path | None = None) -> Path:
    out = out or RUNS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    R = S.Results("main")
    models = grid("main_grid")
    targets = {"T1": {"RM", "EFF", "SEC"}, "T2": {"RM"}, "T3": {"RM", "SEC"}, "T4": {"RM", "EFF", "SEC"},
               "T5": {"RM", "EFF", "SEC"}, "T6": {"RM", "SEC"}, "T8": {"RM", "SEC"}, "T9": {"RM", "SEC"}, "T10": {"RM", "SEC"}}
    hw = S.halfwidths(R, models, TECHS[1:])
    (out / "tab_cor.tex").write_text(main_table(R, ["HE+", "MBPP+", "LCB", "LCBP"], models))
    (out / "tab_rm.tex").write_text(main_table(R, ["C%", "Lint", "CC", "J"], models))
    (out / "tab_eff.tex").write_text(main_table(R, ["DPS", "ET", "Bey"], models))
    (out / "tab_sec.tex").write_text(main_table(R, ["FS", "SP", "Ins", "B/k"], models))
    (out / "tab_tradeoff.tex").write_text(tradeoff_table(R, models, hw, targets))
    (out / "tab_ci.tex").write_text(ci_table(hw, models))
    (out / "tab_budget.tex").write_text(budget_table(S.budget_curves(R, models, TECHS), models))
    A = S.Results("agent")
    (out / "tab_deepswe.tex").write_text("\n".join(
        f"{tech_label(t)} & " + " & ".join(f(A.cell(S.MAIN[4], k, t, 1 if t == 'T0' else 4))
                                         for k in grid("main_grid") + grid("extended_grid")) + r"\\" for t in TECHS))
    if (RUNS / "extended").exists():
        (out / "tab_extended.tex").write_text(extended_table(S.Results("extended"), R))
    if (RUNS / "h2h").exists():
        from .h2h import load_all
        from .h2h.base import H2H
        load_all()
        H = S.Results("h2h")
        techs = {h: (c.family, c.name) for h, c in sorted(H2H.items()) if h.startswith("H")}
        (out / "tab_h2h.tex").write_text(h2h_table(H, techs))
    return out
