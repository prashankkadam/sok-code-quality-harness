"""LaTeX rows for the paper's tables from the stored results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import stats as S
from .config import RUNS, grid, model_cfg
from .techniques.main import TECHNIQUES

ABBR = {"op": "Op", "oss": "oss", "g31": "G31", "g26": "G26"}
TECHS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
CONTROLS = ["T7lo"]
BUDGETS = (4, 8, 16)


def f(x: float, nd: int = 2) -> str:
    return "--" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def tex(s: str) -> str:
    return "".join("\\" + c if c in "&%$#_{}" else c for c in str(s))


def tech_label(t: str) -> str:
    return f"{t} {TECHNIQUES[t].name}"


def _na(t: str, k: str) -> bool:
    """T8-T10 need a tunable model; T7lo needs a settable temperature (not the Claude API or the gateway's closed models)."""
    closed = model_cfg(k)["backend"] in {"anthropic", "vercel"}
    return (t in {"T8", "T9", "T10"} and not model_cfg(k).get("tunable")) or (t in CONTROLS and closed)


def main_table(R: S.Results, labels: list[str], models: list[str], B: int = 4) -> str:
    """E2-E5 body: one row per technique (and the T7lo control), one column per (metric, model)."""
    ms = [m for m in S.MAIN if m.label in labels]
    lines = []
    for t in TECHS + CONTROLS:
        cells = [f(R.cell(m, k, t, 1 if t == "T0" else B)) if not _na(t, k) else "--" for k in models for m in ms]
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


def fdr_table(cells: list[dict], targets: dict[str, set[str]]) -> str:
    """As the trade-off matrix, counting only the differences that survive Benjamini-Hochberg at q=0.05."""
    out = []
    for ref in ("T0", "T7"):
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{({'a' if ref == 'T0' else 'b'}) against {ref}}}}}\\")
        for t in TECHS[1:]:
            if t == ref:
                continue
            cs = [c for c in cells if c["ref"] == ref and c["technique"] == t]
            row = []
            for d in ("COR", "RM", "EFF", "SEC"):
                x = [c for c in cs if c["dim"] == d]
                b = sum(c["survives"] and c["better"] for c in x)
                w = sum(c["survives"] and not c["better"] for c in x)
                row.append(f"{b} / {w} [{len(x)}]" + (r"$\star$" if d in targets.get(t, set()) else ""))
            out.append(f"{tech_label(t)} & " + " & ".join(row) + r"\\")
    return "\n".join(out)


def ci_table(hws: dict[int, dict], models: list[str]) -> str:
    """Half-widths against T7 at every budget: one column per (model, B)."""
    return "\n".join(f"{m.dim} & {m.label} & " + " & ".join(f(hws[B].get((m.label, k), float('nan'))) for k in models
                                                            for B in sorted(hws)) + r"\\" for m in S.MAIN)


def seed_table(sd: dict, models: list[str]) -> str:
    """Per metric and model: median [max] standard deviation across the three seeds over all cells."""
    return "\n".join(f"{m.dim} & {m.label} & " + " & ".join(
        f"{f(sd[(m.label, k)][0])} [{f(sd[(m.label, k)][1])}]" for k in models) + r"\\" for m in S.MAIN)


def control_table(R: S.Results, models: list[str]) -> str:
    """T7lo minus T7 at each budget: mean paired per-task difference and 95% bootstrap half-width."""
    open_models = [k for k in models if not _na("T7lo", k)]
    out = []
    for m in S.MAIN:
        cells = []
        for k in open_models:
            for B in BUDGETS:
                d = S.paired(R.tasks(m, k, "T7lo", B), R.tasks(m, k, "T7", B))
                cells.append(f"{f(float(np.mean(d)) if d else float('nan'))} $\\pm$ {f(S.bootstrap_halfwidth(d))}")
        out.append(f"{m.dim} & {m.label} & " + " & ".join(cells) + r"\\")
    return "\n".join(out)


def budget_table(curves: dict, models: list[str]) -> str:
    out = []
    for label in ("HE+", "Lint", "DPS", "FS"):
        out.append(rf"\multicolumn{{{1 + 4 * len(models)}}}{{@{{}}l}}{{\textit{{{label}}}}}\\")
        for t in TECHS + CONTROLS:
            cells = []
            for k in models:
                for B in (1, 4, 8, 16):
                    v = curves.get((label, k, t, B), float("nan"))
                    cells.append(f(v) if not ((t == "T0" and B > 1) or _na(t, k)) else "--")
            out.append(f"{tech_label(t)} & " + " & ".join(cells) + r"\\")
    return "\n".join(out)


def extended_table(R_ext: S.Results, R_main: S.Results) -> str:
    labels = ["LCBP", "Lint", "J", "Jg", "DPS", "Bey", "FS", "SP", "B/k"]
    ms = [m for m in S.MAIN if m.label in labels]
    rows = []
    for k in grid("extended_grid") + grid("main_grid"):
        R = R_ext if k in grid("extended_grid") else R_main
        cut = model_cfg(k).get("cutoff") or r"\nr"
        rows.append(f"{model_cfg(k)['name']} & {cut} & " + " & ".join(f(R.cell(m, k, 'T0', 1)) for m in ms) + r"\\")
    return "\n".join(rows)


def judge_table(agree: dict) -> str:
    """Agreement of the Anthropic (Claude Opus 5.5) and the non-Anthropic (GPT-6.1 Sol) judge."""
    return "\n".join(f"{name} & {v['n']} & {f(v['rho'])} & {f(v['kappa_qw'])}" + r"\\"
                     for name, v in (("raw 1--5 scores", agree["raw"]), ("length-controlled", agree["length_controlled"])))


def h2h_table(R: S.Results, techs: dict[str, tuple[str, str]], model: str = "g31", B: int = 16,
              matched: dict[str, str] | None = None) -> str:
    """E12 body. techs: hid -> (family, name), in table order; a last panel lists the budget-matched BoN@k references
    (matched: family -> reference id, from run.matched_budgets)."""
    def row(hid, name, label):
        return f"{label} & {name} & " + " & ".join(f(R.cell(m, model, hid, B)) for m in S.H2H) + f" & {f(R.calls(model, hid, B))}" + r"\\"
    out = [row("T0", "T0 single shot (reference)", "--"), row("BoN", r"best-of-$n$ (reference)", "--")]
    fam_prev = None
    for hid, (fam, name) in techs.items():
        if fam != fam_prev:
            out.append(rf"\midrule\multicolumn{{11}}{{@{{}}l}}{{\textit{{{fam}}}}}\\")
            fam_prev = fam
        out.append(row(hid, name, hid))
    if matched:
        from .h2h.references import MATCHED_KS
        out.append(r"\midrule\multicolumn{11}{@{}l}{\textit{Budget-matched best-of-$n$ references (same ranker; "
                   + ", ".join(map(str, MATCHED_KS[:-1])) + f" and {MATCHED_KS[-1]} calls; "
                   + r"decoding techniques use the main reference)}}\\")
        out += [row(f"BoN@{k}", f"best-of-{k}", "--") for k in MATCHED_KS]
    return "\n".join(out)


def h2h_ci_table(iv: dict, hids: list[str], ref: str) -> str:
    """E12 intervals against one reference: per technique and column, difference [95% bootstrap interval]."""
    return "\n".join(f"{h} & " + " & ".join(f"{f(iv[(h, ref, m.label)][0])} [{f(iv[(h, ref, m.label)][1])}, "
                                             f"{f(iv[(h, ref, m.label)][2])}]" for m in S.H2H) + r"\\" for h in hids)


def fidelity_table(names: dict[str, str]) -> str:
    """Per technique and reported number: the authors' value in their own setting next to ours in that setting
    (runs/fidelity/<hid>.json); a setting on a benchmark we do not have shows the original harness is needed."""
    out = []
    for r in S.lit_rows():
        p = RUNS / "fidelity" / f"{r['h2h_id']}.json"
        got = next((x for x in json.loads(p.read_text())["rows"] if x["column"] == r["column"] and x["setting"] == r["setting"]),
                   None) if p.exists() else None
        if got is None:
            ours, where = "--", "not run"
        elif got["status"] != "ok":
            ours, where = "--", "original harness needed"
        else:
            ours, where = f(got["ours"]), f"{tex(got['benchmark'])}, {tex(got['metric'])}, {tex(got['model_name'])}"
        out.append(f"{r['h2h_id']} & {tex(names.get(r['h2h_id'], ''))} & {r['column']} & {f(r['lit'])} & {ours} & {where}"
                   + r"\\")
    return "\n".join(out)



H2H_PRIMARY = {"RM": ("RM", "Pylint score (points)"), "EFF": ("DPS", "EvalPerf DPS$_{norm}$ (points)"),
               "SEC": ("CWE", "CWEval func-sec@1 (points)")}
H2H_FAMILIES = ["prompting", "feedback", "refinement", "multiagent", "retrieval", "decoding", "training"]


def h2h_forest(iv: dict, techs: dict[str, tuple[str, tuple[str, ...]]], path: Path) -> dict[str, list[str]]:
    """Fig. E2: per technique, the difference from best-of-n on the primary metric of its first targeted dimension with
    its 95% interval (iv from stats.h2h_intervals); filled markers lie beyond the interval. techs: hid -> (family,
    targets). Returns the techniques above, below and within the interval (the counts quoted in the text)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    higher = {m.label: m.higher for m in S.H2H}
    colors = dict(zip(H2H_FAMILIES, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]))
    out = {"above": [], "below": [], "within": []}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.6))
    for ax, (dim, (col, xlabel)) in zip(axes, H2H_PRIMARY.items()):
        hids = sorted((h for h, (_, t) in techs.items() if t and t[0] == dim),
                      key=lambda h: (H2H_FAMILIES.index(techs[h][0]), h))
        for y, h in enumerate(hids):
            mean, lo, hi = iv[(h, "BoN", col)]
            sign = 1 if higher[col] else -1
            d, a, b = sign * mean, min(sign * lo, sign * hi), max(sign * lo, sign * hi)
            beyond = a > 0 or b < 0
            out["above" if a > 0 else "below" if b < 0 else "within"].append(h)
            c = colors[techs[h][0]]
            ax.errorbar(d, -y, xerr=[[d - a], [b - d]], fmt="o", ms=3.2, color=c, mfc=c if beyond else "white",
                        elinewidth=0.8, capsize=1.5)
        ax.axvline(0, color="0.4", lw=0.7, ls="--")
        ax.set_yticks([-y for y in range(len(hids))], hids, fontsize=5.5)
        ax.set_title(f"{dim}-targeted (n={len(hids)})", fontsize=7.5)
        ax.set_xlabel(xlabel + "\nminus best-of-$n$", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    return out



def b1_independent_table(R: S.Results, models: list[str], techs=("T1", "T2", "T6"), labels=("CC", "J", "Jg")) -> str:
    """Single-call view on the measures the analyzers do not compute: T0's value, then each single-call technique's
    paired difference from T0 at B=1 with a dagger when its 95% bootstrap interval excludes zero (supplement table)."""
    col = {m.label: m for m in S.MAIN}
    rows = ["T0 single shot & " + " & ".join(f(R.cell(col[l], k, "T0", 1)) for k in models for l in labels) + r"\\"]
    for t in techs:
        cells = []
        for k in models:
            for l in labels:
                d = S.paired(R.tasks(col[l], k, t, 1), R.tasks(col[l], k, "T0", 1))
                lo, hi = S.bootstrap_ci(d)
                mark = r"$^\dagger$" if d and (lo > 0 or hi < 0) else ""
                cells.append((f"${np.mean(d):+.2f}$" if d else "--") + mark)
        rows.append(f"{tech_label(t)} & " + " & ".join(cells) + r"\\")
    return "\n".join(rows)


def write_all(out: Path | None = None) -> Path:
    out = out or RUNS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    R = S.Results("main")
    models = grid("main_grid")
    targets = {"T1": {"RM", "EFF", "SEC"}, "T2": {"RM"}, "T3": {"RM", "SEC"}, "T4": {"RM", "EFF", "SEC"},
               "T5": {"RM", "EFF", "SEC"}, "T6": {"RM", "SEC"}, "T8": {"RM", "SEC"}, "T9": {"RM", "SEC"}, "T10": {"RM", "SEC"}}
    hws = S.halfwidths_by_budget(R, models, TECHS[1:], BUDGETS)
    (out / "tab_cor.tex").write_text(main_table(R, ["HE+", "LCB", "LCBP"], models))
    (out / "tab_rm.tex").write_text(main_table(R, ["C%", "Lint", "CC", "J", "Jg"], models))
    (out / "tab_eff.tex").write_text(main_table(R, ["DPS", "ET", "Bey"], models))
    (out / "tab_sec.tex").write_text(main_table(R, ["FS", "SP", "Ins", "B/k"], models))
    (out / "tab_tradeoff.tex").write_text(tradeoff_table(R, models, hws[4], targets))
    (out / "tab_ci.tex").write_text(ci_table(hws, models))
    (out / "tab_seed_sd.tex").write_text(seed_table(S.seed_dispersion(R, models, TECHS + CONTROLS), models))
    cells = S.fdr_cells(R, models, TECHS[1:])
    (out / "tab_fdr.tex").write_text(fdr_table(cells, targets))
    with open(out / "fdr_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["ref", "technique", "metric", "dim", "model", "n", "diff", "better", "p", "survives"])
        w.writeheader()
        w.writerows(cells)
    (out / "tab_t7lo.tex").write_text(control_table(R, models))
    (out / "tab_budget.tex").write_text(budget_table(S.budget_curves(R, models, TECHS + CONTROLS), models))
    (out / "tab_b1_indep.tex").write_text(b1_independent_table(R, models))
    ds = next(m for m in S.MAIN if m.label == "DS")
    A = S.Results("agent")
    (out / "tab_deepswe.tex").write_text("\n".join(
        f"{tech_label(t)} & " + " & ".join(f(A.cell(ds, k, t, 1 if t == 'T0' else 4))
                                         for k in grid("main_grid") + grid("extended_grid")) + r"\\" for t in TECHS))
    if (RUNS / "extended").exists():
        (out / "tab_extended.tex").write_text(extended_table(S.Results("extended"), R))
    (out / "tab_judges.tex").write_text(judge_table(S.judge_agreement()))
    from .h2h import load_all
    from .h2h.base import H2H
    load_all()
    names = {h: c.name for h, c in H2H.items()}
    if (RUNS / "h2h").exists():
        H = S.Results("h2h")
        techs = {h: (c.family, c.name) for h, c in sorted(H2H.items()) if h.startswith("H")}
        mp = RUNS / "h2h" / "bon_matched.json"
        (out / "tab_h2h.tex").write_text(h2h_table(H, techs, matched=json.loads(mp.read_text()) if mp.exists() else None))
        iv = S.h2h_intervals(H, list(techs))
        (out / "tab_h2h_ci_t0.tex").write_text(h2h_ci_table(iv, list(techs), "T0"))
        (out / "tab_h2h_ci_bon.tex").write_text(h2h_ci_table(iv, list(techs), "BoN"))
        primary = h2h_forest(iv, {h: (c.family, c.target) for h, c in H2H.items() if h in techs},
                             out.parent / "figures" / "h2h_forest.pdf")
        (out / "h2h_primary.json").write_text(json.dumps(primary, indent=1))
    (out / "tab_fidelity.tex").write_text(fidelity_table(names))
    return out
