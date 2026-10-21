#!/usr/bin/env python3
"""Phase 2: synthesis data, figures and generated tables, all from sok/slr/extraction.csv and the
screening outputs, so the paper rebuilds with one command:

  synth.py --run main

Writes vector PDFs to sok/paper/figures/ and LaTeX to sok/paper/generated/:
  figures/prisma.pdf, taxonomy.pdf, evidence_map_family.pdf, evidence_map_signal.pdf, trends.pdf
  generated/counts.tex          \\newcommand macros for every count used in the text (provisional)
  generated/tab_effects.tex     correctness effect x targeted-dimension effect (SWiM vote counts)
  generated/tab_nontargeted.tex share of studies measuring any non-targeted dimension, by family
  generated/tab_qa.tex          QA score distribution and reporting rates by family
  generated/app_studies.tex     longtable: included studies (appendix), no \\cite (keeps references in budget)
  generated/app_qa.tex          longtable: per-study QA scores (appendix)
All counts are provisional until the human screening calibration (B-4) is done; macros render as
\\tbd{provisional: N}.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, read_jsonl  # noqa: E402

PAPER = REPO / "sok" / "paper"
FIG = PAPER / "figures"
GEN = PAPER / "generated"
SLR = REPO / "sok" / "slr"

# Reference palette (dataviz skill, light mode). Categorical order is fixed.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("seqblue", SEQ)

FAMILIES = ["prompting", "feedback-loop", "self-refinement", "multi-agent", "retrieval", "decoding", "training", "other"]
FAM_LABEL = {"prompting": "Prompting", "feedback-loop": "Tool feedback", "self-refinement": "Self-refinement",
             "multi-agent": "Multi-agent", "retrieval": "Retrieval", "decoding": "Decoding",
             "training": "Training", "other": "Hybrid/other"}
DIMS = ["readability-maintainability", "efficiency", "security"]
DIM_LABEL = {"readability-maintainability": "R/M", "efficiency": "EFF", "security": "SEC"}
STAGES = ["pre-generation", "in-decoding", "post-generation-loop", "training-time"]
STAGE_LABEL = {"pre-generation": "Pre-gen.", "in-decoding": "Decoding", "post-generation-loop": "Post-gen. loop",
               "training-time": "Training"}
SIGNALS = ["none", "self", "other-LLM", "static-analyzer", "compiler", "test-execution", "profiler",
           "retrieved-knowledge", "human", "learned-verifier", "reward-model"]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fam(x):
    return x if x in FAMILIES else "other"


def split(s):
    return [t.strip() for t in re.split(r"[;,]", s or "") if t.strip()]


def load_extraction():
    rows = list(csv.DictReader(open(SLR / "extraction.csv")))
    # one record per study (rid); technique rows of the same study are merged for study-level counts
    studies = {}
    for r in rows:
        s = studies.setdefault(r["rid"], {"rid": r["rid"], "rows": [], "year": r["year"], "qa": r["qa_total"],
                                          "title": r["title"], "venue_type": r["venue_type"], "row0": r})
        s["rows"].append(r)
    return rows, studies


def tbdnum(n):
    return f"\\tbd{{provisional: {n}}}"


# ----------------------------------------------------------------------------------- PRISMA
def prisma_counts(run):
    rd = SLR / "runs" / run
    c = {}
    q = {}
    for s in ("S1", "S1r1", "S2r1"):
        p = rd / f"queries_{s}.json"
        if p.exists():
            q[s] = json.loads(p.read_text())
    c["hits_openalex"] = sum(v.get("openalex", {}).get("raw_hits", 0) or 0 for v in q.values())
    c["hits_s2"] = sum(v.get("s2", {}).get("raw_hits", 0) or 0 for v in q.values())
    c["hits_arxiv"] = sum(v.get("arxiv", {}).get("raw_hits", 0) or 0 for v in q.values())
    st = json.loads((rd / "dedup_all_stats.json").read_text())
    c["records_in"] = st["records_in"]
    c["duplicates"] = st["duplicates_removed"]
    c["screened_db"] = st["unique_after_dedup"]
    dec = list(csv.DictReader(open(rd / "decisions_ta.csv")))
    db = [d for d in dec if d["rid"].startswith("R")]
    c["ta_excluded_db"] = sum(1 for d in db if d["forward_to_fulltext"] == "False")
    c["ta_forward_db"] = sum(1 for d in db if d["forward_to_fulltext"] == "True")
    other = [d for d in dec if d["rid"][0] in "VO"]
    c["other_identified"] = len(other)
    c["other_forward"] = sum(1 for d in other if d["forward_to_fulltext"] == "True")
    c["other_excluded_ta"] = sum(1 for d in other if d["forward_to_fulltext"] == "False")
    snow = [d for d in dec if d["rid"].startswith("S")]
    c["snow_identified"] = len(snow)
    c["snow_forward"] = sum(1 for d in snow if d["forward_to_fulltext"] == "True")
    sa = {}
    pa = rd / "extraction" / "batch_snowA" / "parsed.jsonl"
    if pa.exists():
        for l in pa.open():
            x = json.loads(l)
            sa[x["rid"]] = x["extraction"].get("eligibility")
    c["snow_stageA_excluded"] = sum(1 for v in sa.values() if v == "exclude")
    c["snow_awaiting"] = c["snow_forward"] - c["snow_stageA_excluded"]
    man = list(csv.DictReader(open(rd / "fulltext_manifest.csv")))
    c["ft_not_retrieved"] = sum(1 for m in man if m["source"] == "abstract-only")
    side = [s for s in csv.DictReader(open(rd / "side_lists.csv")) if not s["rid"].startswith("S")]
    c["ft_excluded"] = len(side)
    ex = Counter(s["exclusion_criterion"] or "other" for s in side)
    c["ft_excluded_by"] = dict(ex.most_common())
    c["side_context"] = sum(1 for s in side if s["side_list"] == "context")
    c["side_adjacent"] = sum(1 for s in side if s["side_list"] == "adjacent")
    c["side_correctness"] = sum(1 for s in side if s["side_list"] == "correctness-only")
    c["side_secondary"] = sum(1 for s in side if s["side_list"] == "secondary")
    _, studies = load_extraction()
    c["included"] = len(studies)
    c["included_db"] = sum(1 for k in studies if k.startswith("R"))
    c["included_other"] = c["included"] - c["included_db"]
    return c


def box(ax, x, y, w, h, text, fc="#f4f3ef"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=INK2, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6))


def fig_prisma(c):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    L, R, W = 0.0, 0.53, 0.47
    box(ax, L, 0.86, W, 0.13, f"Records identified (database search)\nOpenAlex n={c['hits_openalex']}; Semantic Scholar "
        f"n={c['hits_s2']};\narXiv n={c['hits_arxiv']} (raw hits over all string\niterations, overlapping)")
    box(ax, R, 0.88, W, 0.09, f"Removed before screening:\nduplicates n={c['duplicates']}")
    box(ax, L, 0.69, W, 0.10, f"Records screened (title/abstract)\nn={c['screened_db']}")
    box(ax, R, 0.69, W, 0.10, f"Excluded at title/abstract\nn={c['ta_excluded_db']}")
    box(ax, L, 0.49, W, 0.13, f"Reports assessed in full text\nn={c['ta_forward_db'] + c['other_forward']}\n"
        f"(database {c['ta_forward_db']}; other methods {c['other_forward']})")
    items = list(c["ft_excluded_by"].items())
    ex1 = ", ".join(f"{k} {v}" for k, v in items[:4])
    ex2 = ", ".join(f"{k} {v}" for k, v in items[4:9])
    box(ax, R, 0.44, W, 0.21, f"Excluded at full text n={c['ft_excluded']}\n{ex1}\n{ex2}\n"
        f"(no open full text, judged on\nabstract: {c['ft_not_retrieved']})")
    box(ax, R, 0.17, W, 0.24, f"Other methods: venue sweep and section-\nlevel searches n={c['other_identified']} "
        f"(excluded at\ntitle/abstract {c['other_excluded_ta']}); snowballing (1 back-\nward + 1 forward) n={c['snow_identified']}: "
        f"forwarded {c['snow_forward']},\nexcluded at full text (stage A) {c['snow_stageA_excluded']},\n"
        f"not extracted {c['snow_awaiting']}")
    box(ax, L, 0.24, W, 0.13, f"Studies included\nn={c['included']}\n(database {c['included_db']}; "
        f"other methods {c['included_other']})", fc="#e6effb")
    box(ax, L, 0.0, 1.0, 0.13, f"Side lists (not primary studies): context {c['side_context']}, adjacent "
        f"{c['side_adjacent']}, correctness-only {c['side_correctness']},\nsecondary {c['side_secondary']}.")
    arrow(ax, L + W / 2, 0.86, L + W / 2, 0.79)
    arrow(ax, L + W, 0.925, R, 0.925)
    arrow(ax, L + W / 2, 0.69, L + W / 2, 0.62)
    arrow(ax, L + W, 0.74, R, 0.74)
    arrow(ax, L + W / 2, 0.49, L + W / 2, 0.37)
    arrow(ax, L + W, 0.555, R, 0.555)
    arrow(ax, R, 0.29, L + W, 0.31)
    fig.savefig(FIG / "prisma.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------- maps
def heat(ax, M, rows, cols, rlab, clab, shade=None, vmax=None, title=""):
    import numpy as np
    A = np.array(shade if shade is not None else M, dtype=float)
    vmax = vmax or (A.max() if A.size and A.max() > 0 else 1)
    ax.imshow(A, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i][j]
            dark = A[i][j] > 0.55 * vmax
            ax.text(j, i, str(v) if v else "–", ha="center", va="center", fontsize=7,
                    color="white" if dark else (INK if v else INK2))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([clab[c] for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([rlab[r] if isinstance(rlab, dict) else r for r in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title, fontsize=8, color=INK)


def fig_maps(rows, studies):
    import numpy as np
    # Map 1: family x targeted dimension; count of studies; shaded by mean QA.
    cnt = defaultdict(set)
    qa = defaultdict(list)
    for r in rows:
        for d in split(r["dims_targeted"]):
            if d in DIMS:
                cnt[(fam(r["family_v0"]), d)].add(r["rid"])
    for (f, d), ids in cnt.items():
        qa[(f, d)] = [float(studies[i]["qa"] or 0) for i in ids]
    M = [[len(cnt[(f, d)]) for d in DIMS] for f in FAMILIES]
    S = [[(np.mean(qa[(f, d)]) if qa[(f, d)] else 0) for d in DIMS] for f in FAMILIES]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    heat(ax, M, FAMILIES, DIMS, FAM_LABEL, DIM_LABEL, shade=S, vmax=10)
    sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(0, 10))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("mean QA score (0-10)", fontsize=7)
    cb.outline.set_visible(False)
    ax.set_xlabel("targeted dimension; cell = number of studies")
    fig.savefig(FIG / "evidence_map_family.pdf", bbox_inches="tight")
    plt.close(fig)
    # Map 2: signal source x targeted dimension
    cnt2 = defaultdict(set)
    for r in rows:
        for s in split(r["signal_source"]):
            for d in split(r["dims_targeted"]):
                if s in SIGNALS and d in DIMS:
                    cnt2[(s, d)].add(r["rid"])
    M2 = [[len(cnt2[(s, d)]) for d in DIMS] for s in SIGNALS]
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    heat(ax, M2, SIGNALS, DIMS, {s: s for s in SIGNALS}, DIM_LABEL)
    ax.set_xlabel("targeted dimension; cell = number of studies")
    fig.savefig(FIG / "evidence_map_signal.pdf", bbox_inches="tight")
    plt.close(fig)
    return M, M2


def fig_taxonomy(rows):
    # Small multiples: stage x signal source, one panel per targeted dimension.
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True)
    out = {}
    for ax, d in zip(axes, DIMS):
        c = defaultdict(set)
        for r in rows:
            if d not in split(r["dims_targeted"]):
                continue
            for s in split(r["signal_source"]):
                if s in SIGNALS and r["stage"] in STAGES:
                    c[(s, r["stage"])].add(r["rid"])
        M = [[len(c[(s, st)]) for st in STAGES] for s in SIGNALS]
        out[d] = M
        heat(ax, M, SIGNALS, STAGES, {s: s for s in SIGNALS}, STAGE_LABEL, vmax=max(1, max(max(m) for m in M)),
             title={"readability-maintainability": "Target: readability/maintainability",
                    "efficiency": "Target: efficiency", "security": "Target: security"}[d])
        ax.tick_params(axis="x", rotation=40, labelsize=6.5)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIG / "taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


BENCH_PAT = {
    "HumanEval(+)": r"humaneval", "MBPP(+)": r"\bmbpp", "LiveCodeBench": r"livecodebench", "SWE-bench": r"swe-?bench",
    "BigCodeBench": r"bigcodebench", "LLMSecEval": r"llmseceval", "SecurityEval": r"securityeval",
    "CyberSecEval": r"cyberseceval", "PythonSecurityEval": r"pythonsecurityeval", "CWEval": r"cweval",
    "BaxBench": r"baxbench", "SecCodePLT": r"seccodeplt", "EffiBench": r"effibench", "Mercury": r"mercury",
    "EvalPerf": r"evalperf", "PIE": r"\bpie\b", "RACE": r"\brace\b", "NoFunEval": r"nofuneval",
    "Asleep (Pearce) / CodeQL scenarios": r"asleep|pearce", "LeetCode": r"leetcode", "APPS": r"\bapps\b",
    "CodeContests": r"codecontests",
}


def fig_trends(rows, studies):
    import numpy as np
    years = list(range(2021, 2027))
    per = {f: [0] * len(years) for f in FAMILIES}
    for rid, s in studies.items():
        y = int(s["year"] or 0)
        f = fam(s["rows"][0]["family_v0"])
        if y in years:
            per[f][years.index(y)] += 1
    bench = {b: [0] * len(years) for b in BENCH_PAT}
    for rid, s in studies.items():
        y = int(s["year"] or 0)
        txt = " ".join(r["benchmarks"] for r in s["rows"]).lower()
        if y in years:
            for b, pat in BENCH_PAT.items():
                if re.search(pat, txt):
                    bench[b][years.index(y)] += 1
    top = sorted(bench, key=lambda b: -sum(bench[b]))[:12]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
    bottom = np.zeros(len(years))
    for i, f in enumerate(FAMILIES):
        v = np.array(per[f])
        a1.bar(years, v, bottom=bottom, color=CAT[i], width=0.72, edgecolor="white", linewidth=0.8, label=FAM_LABEL[f])
        bottom += v
    a1.set_ylabel("included studies")
    a1.set_xticks(years)
    a1.set_xticklabels([str(y) if y < 2026 else "2026*" for y in years])
    a1.spines[["top", "right"]].set_visible(False)
    a1.yaxis.grid(True, color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    a1.legend(fontsize=6, frameon=False, ncol=2, loc="upper left")
    a1.set_title("Studies per year by family", fontsize=8)
    M = [bench[b] for b in top]
    heat(a2, M, top, years, {b: b for b in top}, {y: (str(y) if y < 2026 else "2026*") for y in years})
    a2.set_title("Benchmark usage (studies per year)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "trends.pdf", bbox_inches="tight")
    plt.close(fig)
    return per, {b: bench[b] for b in top}


# ----------------------------------------------------------------------------------- tables
def effect_table(rows):
    order = ["+", "0", "-", "mixed", "NR"]
    M = {(a, b): set() for a in order for b in order}
    for r in rows:
        a = r["effect_correctness"] if r["effect_correctness"] in order else "NR"
        b = r["effect_target"] if r["effect_target"] in order else "NR"
        M[(a, b)].add(r["study_id"])
    glyph = {"+": "\\up", "0": "\\same", "-": "\\down", "mixed": "mixed", "NR": "\\nr"}
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{Direction of effect on correctness (rows) against direction of effect on the targeted "
             "quality dimension (columns), counted per technique row of \\texttt{extraction.csv} (vote counting by "
             "direction, SWiM; no effect sizes are pooled). Counts are provisional (LLM-assisted extraction, "
             "human review pending).}\\label{tab:effects}",
             "\\begin{tabular}{l" + "r" * len(order) + "r}", "\\toprule",
             "\\dimCor{} \\textbackslash{} target & " + " & ".join(glyph[o] for o in order) + " & total \\\\", "\\midrule"]
    for a in order:
        tot = sum(len(M[(a, b)]) for b in order)
        lines.append(glyph[a] + " & " + " & ".join(str(len(M[(a, b)])) for b in order) + f" & {tot} \\\\")
    lines += ["\\midrule", "total & " + " & ".join(str(sum(len(M[(a, b)]) for a in order)) for b in order) +
              f" & {len(rows)} \\\\", "\\bottomrule", "\\end{tabular}",
              "\\\\[2pt]{\\scriptsize \\up{} improved, \\same{} no clear change, \\down{} worse, mixed = differs across "
              "settings, \\nr{} = not reported.}", "\\end{table}"]
    (GEN / "tab_effects.tex").write_text("\n".join(lines) + "\n")
    return {f"{a}|{b}": len(M[(a, b)]) for a in order for b in order}


def nontargeted_table(rows, studies):
    stats = {}
    for f in FAMILIES:
        ids = {r["rid"] for r in rows if fam(r["family_v0"]) == f}
        if not ids:
            continue
        any_other, corr = 0, 0
        for rid in ids:
            rs = studies[rid]["rows"]
            tgt = {d for r in rs for d in split(r["dims_targeted"])}
            meas = {d for r in rs for d in split(r["dims_measured"])}
            if meas - tgt:
                any_other += 1
            if any(str(r["correctness_measured"]).lower() == "true" for r in rs):
                corr += 1
        stats[f] = (len(ids), any_other, corr)
    allids = set(studies)
    tot_other = sum(1 for rid in allids if {d for r in studies[rid]["rows"] for d in split(r["dims_measured"])} -
                    {d for r in studies[rid]["rows"] for d in split(r["dims_targeted"])})
    tot_corr = sum(1 for rid in allids if any(str(r["correctness_measured"]).lower() == "true" for r in studies[rid]["rows"]))
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{How often included studies measured anything beyond their targeted dimension: share of studies "
             "per family that measured at least one non-targeted quality dimension, and share that measured "
             "correctness. Counts are provisional (LLM-assisted extraction, human review pending).}"
             "\\label{tab:nontargeted}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Family & Studies & \\multicolumn{2}{c}{Non-targeted dim.\\ measured} & \\multicolumn{2}{c}{\\dimCor{} measured} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", " & & $n$ & \\% & $n$ & \\% \\\\", "\\midrule"]
    for f, (n, o, c) in stats.items():
        lines.append(f"{FAM_LABEL[f]} & {n} & {o} & {100 * o / n:.0f} & {c} & {100 * c / n:.0f} \\\\")
    n = len(allids)
    lines += ["\\midrule", f"All studies & {n} & {tot_other} & {100 * tot_other / max(n, 1):.0f} & {tot_corr} & "
              f"{100 * tot_corr / max(n, 1):.0f} \\\\", "\\bottomrule", "\\end{tabular}",
              "\\\\[2pt]{\\scriptsize A study can appear in more than one family when it evaluates several techniques; "
              "the last row counts each study once. Percentages are our computation over the extracted rows.}",
              "\\end{table}"]
    (GEN / "tab_nontargeted.tex").write_text("\n".join(lines) + "\n")
    return stats, tot_other, tot_corr, n


def qa_table(rows, studies):
    import numpy as np
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{Evidence quality by family: median QA score (0--10, items QA1--QA10 of Appendix~\\ref{sec:app-studies}) and the share of studies "
             "that report compute or cost (QA7 $>0$), consider contamination (QA9 $>0$), use human evaluation, and "
             "release artifacts (QA8 $>0$). Provisional (LLM-assisted extraction).}\\label{tab:qa}",
             "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Family & $n$ & Med.\\ QA & Comp.\\ \\% & Cont.\\ \\% & Human \\% & Artif.\\ \\% \\\\",
             "\\midrule"]
    out = {}
    for f in FAMILIES + ["ALL"]:
        ids = sorted({r["rid"] for r in rows if f == "ALL" or fam(r["family_v0"]) == f})
        if not ids:
            continue
        r0 = [studies[i]["row0"] for i in ids]
        q = [float(studies[i]["qa"] or 0) for i in ids]

        def share(fn):
            return 100 * sum(1 for r in r0 if fn(r)) / len(r0)
        comp = share(lambda r: float(r["qa_7"] or 0) > 0)
        cont = share(lambda r: float(r["qa_9"] or 0) > 0)
        hum = share(lambda r: r["human_eval"] not in ("", "none"))
        art = share(lambda r: float(r["qa_8"] or 0) > 0)
        name = "All studies" if f == "ALL" else FAM_LABEL[f]
        if f == "ALL":
            lines.append("\\midrule")
        lines.append(f"{name} & {len(ids)} & {np.median(q):.1f} & {comp:.0f} & {cont:.0f} & {hum:.0f} & {art:.0f} \\\\")
        out[f] = dict(n=len(ids), median_qa=float(np.median(q)), compute=comp, contamination=cont, human=hum, artifacts=art)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "tab_qa.tex").write_text("\n".join(lines) + "\n")
    return out


def tex_escape(s):
    s = s or ""
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s


def appendix_tables(rows, studies):
    lines = ["{\\scriptsize", "\\begin{longtable}{@{}p{0.07\\linewidth}p{0.37\\linewidth}p{0.05\\linewidth}"
             "p{0.12\\linewidth}p{0.08\\linewidth}p{0.16\\linewidth}@{}}",
             "\\caption{Included primary studies (provisional; LLM-assisted screening and extraction, human review "
             "pending). ID = record identifier in \\texttt{sok/slr/extraction.csv}; Src = arXiv ID or DOI.}"
             "\\label{tab:app-studies}\\\\", "\\toprule", "ID & Title & Year & Family & Targets & Src \\\\",
             "\\midrule", "\\endfirsthead", "\\toprule", "ID & Title & Year & Family & Targets & Src \\\\", "\\midrule",
             "\\endhead"]
    for rid in sorted(studies, key=lambda k: (studies[k]["rows"][0]["family_v0"], studies[k]["year"], k)):
        s = studies[rid]
        r = s["rows"][0]
        fams = sorted({FAM_LABEL[fam(x["family_v0"])] for x in s["rows"]})
        tg = sorted({DIM_LABEL.get(d, d) for x in s["rows"] for d in split(x["dims_targeted"])})
        src = ("arXiv:" + r["arxiv_id"]) if r["arxiv_id"] else (r["doi"] or "")
        src_tex = re.sub(r"([/.\-_:])", lambda m: tex_escape(m.group(1)) + "\\allowbreak{}", src)
        lines.append(f"{rid} & {tex_escape(s['title'])} & {s['year']} & {', '.join(fams)} & {', '.join(tg)} & "
                     f"{{\\ttfamily\\raggedright {src_tex}}} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", "}"]
    (GEN / "app_studies.tex").write_text("\n".join(lines) + "\n")
    q = ["{\\scriptsize", "\\begin{longtable}{@{}l" + "c" * 10 + "r@{}}",
         "\\caption{Per-study quality assessment (items QA1--QA10 of Appendix~\\ref{sec:app-studies}; 0, 0.5 or 1 per item; provisional, LLM-assisted)."
         "}\\label{tab:app-qa}\\\\", "\\toprule",
         "ID & " + " & ".join(f"Q{i}" for i in range(1, 11)) + " & Total \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "ID & " + " & ".join(f"Q{i}" for i in range(1, 11)) + " & Total \\\\", "\\midrule", "\\endhead"]
    for rid in sorted(studies):
        r = studies[rid]["row0"]
        q.append(rid + " & " + " & ".join(str(r[f"qa_{i}"]) for i in range(1, 11)) + f" & {r['qa_total']} \\\\")
    q += ["\\bottomrule", "\\end{longtable}", "}"]
    (GEN / "app_qa.tex").write_text("\n".join(q) + "\n")


def counts_tex(c, rows, studies, extra):
    fams = Counter()
    for rid, s in studies.items():
        for f in {fam(r["family_v0"]) for r in s["rows"]}:
            fams[f] += 1
    dims = Counter()
    for rid, s in studies.items():
        for d in {d for r in s["rows"] for d in split(r["dims_targeted"])}:
            dims[d] += 1
    m = {
        "nHitsOpenAlex": c["hits_openalex"], "nHitsSS": c["hits_s2"], "nHitsArxiv": c["hits_arxiv"],
        "nRecordsIn": c["records_in"], "nDuplicates": c["duplicates"], "nScreened": c["screened_db"],
        "nTaExcluded": c["ta_excluded_db"], "nTaForward": c["ta_forward_db"], "nOtherIdentified": c["other_identified"],
        "nOtherForward": c["other_forward"], "nFtAssessed": c["ta_forward_db"] + c["other_forward"],
        "nSnowIdentified": c["snow_identified"], "nSnowForward": c["snow_forward"],
        "nSnowStageAExcluded": c["snow_stageA_excluded"], "nSnowAwaiting": c["snow_awaiting"],
        "nFtExcluded": c["ft_excluded"], "nFtAbstractOnly": c["ft_not_retrieved"], "nIncluded": c["included"],
        "nIncludedDb": c["included_db"], "nIncludedOther": c["included_other"], "nTechRows": len(rows),
        "nSideContext": c["side_context"], "nSideAdjacent": c["side_adjacent"],
        "nSideCorrectness": c["side_correctness"], "nSideSecondary": c["side_secondary"],
        "nFamPrompting": fams["prompting"], "nFamFeedback": fams["feedback-loop"],
        "nFamSelfRefine": fams["self-refinement"], "nFamMultiAgent": fams["multi-agent"],
        "nFamRetrieval": fams["retrieval"], "nFamDecoding": fams["decoding"], "nFamTraining": fams["training"],
        "nFamOther": fams["other"], "nDimRM": dims["readability-maintainability"], "nDimEff": dims["efficiency"],
        "nDimSec": dims["security"],
        "nNonTargetedMeasured": extra["tot_other"], "nCorrectnessMeasured": extra["tot_corr"],
        "nLowConfidence": sum(1 for s in studies.values() if s["row0"]["confidence"] == "low"),
    }
    tag = {"prompting": "Prompting", "feedback-loop": "Feedback", "self-refinement": "SelfRefine",
           "multi-agent": "MultiAgent", "retrieval": "Retrieval", "decoding": "Decoding", "training": "Training",
           "other": "Other"}
    for f, (n, o, cm) in extra.get("nt", {}).items():
        m[f"nNonTarg{tag[f]}"] = o
        m[f"nCorr{tag[f]}"] = cm
    lines = ["% GENERATED by src/slr/synth.py. Counts are provisional (human calibration pending, B-4).",
             "% Use \\cnt{name} in text; it renders as N in drafts and is an error in the submission build."]
    for k, v in m.items():
        lines.append(f"\\expandafter\\def\\csname cnt@{k}\\endcsname{{{v}}}")
    # Counts are real but provisional (human calibration B-4 pending, disclosed in the text): plain numbers
    # in drafts, a hard error in the submission build until the calibration is done.
    lines.append("\\makeatletter\\ifsubmission\\newcommand{\\cnt}[1]{\\PackageError{sok}{Provisional count #1}{}}"
                 "\\else\\newcommand{\\cnt}[1]{\\@nameuse{cnt@#1}}\\fi"
                 "\\newcommand{\\cntraw}[1]{\\@nameuse{cnt@#1}}\\makeatother")
    (GEN / "counts.tex").write_text("\n".join(lines) + "\n")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    rows, studies = load_extraction()
    c = prisma_counts(a.run)
    fig_prisma(c)
    M1, M2 = fig_maps(rows, studies)
    tax = fig_taxonomy(rows)
    per, bench = fig_trends(rows, studies)
    eff = effect_table(rows)
    nt, tot_other, tot_corr, n = nontargeted_table(rows, studies)
    qa = qa_table(rows, studies)
    appendix_tables(rows, studies)
    m = counts_tex(c, rows, studies, {"tot_other": tot_other, "tot_corr": tot_corr, "nt": nt})
    summary = {"prisma": c, "counts": m, "effects": eff, "nontargeted": nt, "qa": qa,
               "map_family": dict(zip(FAMILIES, M1)), "map_signal": dict(zip(SIGNALS, M2)),
               "per_year": per, "benchmarks": bench}
    (SLR / "runs" / a.run / "synthesis_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"prisma": c, "counts": m}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
