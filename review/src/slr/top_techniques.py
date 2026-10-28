#!/usr/bin/env python3
"""Highlighted techniques per family.

  top_techniques.py cites                 # citation counts for every included study (Semantic Scholar batch,
                                          #   Crossref fallback for DOIs) -> sok/slr/citations.csv
  top_techniques.py pool                  # eligible candidate rows per family -> sok/slr/top/pool.csv
  top_techniques.py rank                  # landmark + recent-leader lanes -> sok/slr/top/selected.csv
  top_techniques.py table                 # LaTeX tables -> sok/paper/generated/tab_top_<family>.tex

Selection rule (also stated in the paper, \\S\\ref{sec:highlighted}):
  eligible  = technique proposal (extraction) with target effect '+' and correctness effect not '-';
              pass 1 (configs/DOC-001/top_pass1_brief.md): in scope, family, quoted gain;
              pass 2 (top_pass2_brief.md): proposes its own method, general-purpose (not tied to HDL,
              GPU kernels, PLC, smart contracts, ...), not a duplicate record;
              not excluded in sok/slr/top/exclusions.csv (E1 add-on inside an empirical study or
              benchmark paper, E2 evaluation on a handful of tasks, E3 code quality incidental,
              E4 flagged low credibility).
  landmarks = the 5 most-cited eligible techniques of the family (Semantic Scholar count, October 2026).
  recent    = the 5 eligible techniques published 2025-2026 with the largest reported relative gain on
              their target dimension over their own strongest baseline, as a percentile within that
              dimension (quote-checked); excluding landmarks; ties broken by citations.
  A study is highlighted in one family only. A short recent lane is filled by citations; a family with
  fewer than 10 eligible techniques takes all of them, and the shortfall is reported. Domain-specific
  techniques are listed separately (top 3 per domain by citations).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, REPO, UA, http_get, now_iso  # noqa: E402

SLR = REPO / "sok" / "slr"
EXTRACTION = SLR / "extraction.csv"
CITES = SLR / "citations.csv"
TOP = SLR / "top"
FAMILIES = ["prompting", "feedback-loop", "self-refinement", "multi-agent", "retrieval", "decoding", "training"]


def load_rows() -> list[dict]:
    from synth import retrieved
    ok = retrieved()
    return [r for r in csv.DictReader(open(EXTRACTION)) if r["rid"] in ok]


def s2_batch(ids: list[str]) -> list[dict | None]:
    """POST /paper/batch with retries; cached per id list."""
    key = CACHE / "s2batch" / (str(abs(hash(tuple(ids)))) + ".json")
    key.parent.mkdir(parents=True, exist_ok=True)
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    delay = 5.0
    for _ in range(8):
        r = requests.post(url, params={"fields": "citationCount,year,title,externalIds"},
                          json={"ids": ids}, headers={"User-Agent": UA}, timeout=120)
        if r.status_code == 200:
            key.write_text(r.text)
            return r.json()
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise SystemExit(f"S2 batch failed: {r.status_code} {r.text[:200]}")


def crossref_count(doi: str) -> int | None:
    st, body = http_get(f"https://api.crossref.org/works/{doi}")
    if st != 200:
        return None
    try:
        return int(json.loads(body)["message"].get("is-referenced-by-count"))
    except Exception:
        return None


def cmd_cites(_a) -> None:
    studies: dict[str, dict] = {}
    for r in load_rows():
        studies.setdefault(r["rid"], r)
    ids, back = [], {}
    for rid, r in studies.items():
        if r["arxiv_id"].strip():
            i = "arXiv:" + r["arxiv_id"].strip().split("v")[0]
        elif r["doi"].strip():
            i = "DOI:" + r["doi"].strip()
        else:
            continue
        ids.append(i)
        back[i] = rid
    out = {rid: {"rid": rid, "title": studies[rid]["title"], "year": studies[rid]["year"], "s2_citations": "",
                 "crossref_citations": "", "source": "", "fetched_at": ""} for rid in studies}
    for k in range(0, len(ids), 400):
        chunk = ids[k:k + 400]
        res = s2_batch(chunk)
        for i, p in zip(chunk, res):
            if p and p.get("citationCount") is not None:
                o = out[back[i]]
                o["s2_citations"], o["source"], o["fetched_at"] = p["citationCount"], "semantic-scholar", now_iso()
        print(f"S2 {min(k + 400, len(ids))}/{len(ids)}", flush=True)
    for rid, o in out.items():
        doi = studies[rid]["doi"].strip()
        if doi:
            c = crossref_count(doi)
            if c is not None:
                o["crossref_citations"] = c
                if not o["source"]:
                    o["source"], o["fetched_at"] = "crossref", now_iso()
    with open(CITES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(out.values())).keys()))
        w.writeheader()
        w.writerows(out.values())
    have = sum(1 for o in out.values() if o["source"])
    print(f"wrote {CITES}: {have}/{len(out)} studies with a count")


def citations() -> dict[str, int]:
    c = {}
    for r in csv.DictReader(open(CITES)):
        v = r["s2_citations"] or r["crossref_citations"]
        c[r["rid"]] = int(v) if v else 0
    return c


POOL_COLS = ["tid", "rid", "title", "year", "family_v0", "technique_name", "stage", "signal_source",
             "dims_targeted", "metrics", "benchmarks", "models", "baselines", "effect_target",
             "effect_target_headline", "effect_correctness", "effect_correctness_headline", "artifact_url",
             "fulltext_source", "notes", "citations"]


def tech_rows() -> list[dict]:
    """Every extraction row with a technique id tid = <rid>#<k> (k = row order within the study)."""
    seen: dict[str, int] = {}
    out = []
    for r in load_rows():
        k = seen.get(r["rid"], 0)
        seen[r["rid"]] = k + 1
        out.append({**r, "tid": f"{r['rid']}#{k}"})
    return out


def eligible(r: dict) -> bool:
    return (r["study_type"].strip() == "technique-proposal" and r["effect_target"].strip() == "+"
            and (r["effect_correctness"].strip() or "NR") != "-"
            and r["family_v0"].strip() in FAMILIES + ["hybrid"])


def cmd_pool(_a) -> None:
    TOP.mkdir(parents=True, exist_ok=True)
    cit = citations()
    rows = [dict({c: r.get(c, "") for c in POOL_COLS}, citations=cit.get(r["rid"], 0))
            for r in tech_rows() if eligible(r)]
    with open(TOP / "pool.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"pool: {len(rows)} eligible technique rows -> {TOP / 'pool.csv'}")


PER_LANE, PER_FAMILY, RECENT_FROM = 5, 10, 2025
# Training: reinforcement learning, the dominant training method of 2025-2026, replaces the three non-RL recent
# picks (the paper's stated adjustment of the rule).
RL_OVERRIDE = {"R02496#0": "R02468#0", "R05410#0": "R01746#0", "R02014#0": "R04564#0"}


def _num(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def rel_gain(r: dict) -> float | None:
    """Relative improvement on the reported target metric over the authors' own strongest baseline,
    from the reviewer's quoted pair. None when there is no checked, numeric, positive pair."""
    b, t = _num(r.get("baseline_value", "")), _num(r.get("technique_value", ""))
    if b is None or t is None or b == 0 or r.get("quote_level") not in ("exact", "near"):
        return None
    g = (t - b) / abs(b) if r.get("higher_is_better", "").strip() == "yes" else (b - t) / abs(b)
    return g if g > 0 else None


def merged() -> list[dict]:
    from quotecheck import match_level  # noqa: E402
    from extract import all_records  # noqa: E402
    pool = {r["tid"]: r for r in csv.DictReader(open(TOP / "pool.csv"))}
    recs = all_records("main")
    out = []
    for p in sorted(TOP.glob("pass1_out_*.csv")):
        for r in csv.DictReader(open(p)):
            if r["tid"] not in pool:
                continue
            m = {**pool[r["tid"]], **r}
            ft = CACHE / "fulltext_slr" / f"{m['rid']}.txt"
            src = ft.read_text(errors="replace") if ft.exists() else recs.get(m["rid"], {}).get("abstract") or ""
            q = m.get("gain_quote", "").strip()
            m["quote_level"] = match_level(q, src) if q and src else ""
            g = rel_gain(m)
            m["rel_gain"] = "" if g is None else f"{g:.4f}"
            out.append(m)
    missing = set(pool) - {m["tid"] for m in out}
    if missing:
        raise SystemExit(f"{len(missing)} pool rows have no pass-1 output, e.g. {sorted(missing)[:5]}")
    return out


def pass2() -> dict[str, dict]:
    out = {}
    for p in sorted(TOP.glob("pass2_out_*.csv")):
        for r in csv.DictReader(open(p)):
            out[r["tid"]] = r
    return out


def gain_percentiles(rows: list[dict]) -> None:
    """Percentile of each relative gain within its target dimension over the whole eligible pool, so
    that a 10x kernel speedup does not outrank every security or readability gain."""
    by_dim: dict[str, list[float]] = {}
    for r in rows:
        if r["rel_gain"]:
            by_dim.setdefault(r["target_dim"].strip(), []).append(float(r["rel_gain"]))
    for r in rows:
        if r["rel_gain"]:
            xs = by_dim[r["target_dim"].strip()]
            g = float(r["rel_gain"])
            r["gain_pct"] = f"{sum(x <= g for x in xs) / len(xs):.4f}"
        else:
            r["gain_pct"] = ""


def cmd_rank(_a) -> None:
    rows = merged()
    p2 = pass2()
    for r in rows:
        r.update({k: v for k, v in p2.get(r["tid"], {}).items() if k != "tid"})
    with open(TOP / "merged.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    dup = {t: r.get("duplicate_of", "").split(";")[0].strip() for t, r in p2.items()}
    pool_tids = {r["tid"] for r in rows}

    def is_dup(t: str) -> bool:
        """Drop a record that duplicates another; for mutual flags keep the lexically first tid."""
        d = dup.get(t, "")
        if not d or d not in p2 or d not in pool_tids or d == t:
            return False
        return not (dup.get(d) == t and t < d)

    base = [r for r in rows if r["tid"] in p2 and r.get("is_technique", "").strip() == "yes"
            and not is_dup(r["tid"]) and r.get("family_ok", "").strip() in FAMILIES]
    excl = {r["tid"]: r for r in csv.DictReader(open(TOP / "exclusions.csv"))}
    base = [r for r in base if r["tid"] not in excl]
    ok = [r for r in base if r.get("general_purpose", "").strip() == "yes"]
    gain_percentiles(ok)
    cit = lambda r: (int(r["citations"] or 0), int(r["year"] or 0))  # noqa: E731
    sel, report, used = [], [], set()
    for fam in FAMILIES:
        cands = [r for r in ok if r["family_ok"].strip() == fam and r["rid"] not in used]
        best: dict[str, dict] = {}  # one technique per study: highest gain percentile, then first row
        for r in cands:
            cur = best.get(r["rid"])
            if cur is None or float(r["gain_pct"] or -1) > float(cur["gain_pct"] or -1):
                best[r["rid"]] = r
        cands = list(best.values())
        land = sorted(cands, key=cit, reverse=True)[:PER_LANE]
        lids = {r["rid"] for r in land}
        recent = [r for r in cands if r["rid"] not in lids and int(r["year"] or 0) >= RECENT_FROM]
        rg = sorted([r for r in recent if r["gain_pct"]], key=lambda r: (float(r["gain_pct"]), cit(r)), reverse=True)
        lane_r = rg[:PER_LANE]
        notes = []
        if len(lane_r) < PER_LANE:
            extra = sorted([r for r in recent if not r["gain_pct"]], key=cit, reverse=True)
            k = min(len(extra), PER_LANE - len(lane_r))
            if k:
                notes.append(f"recent lane: {len(lane_r)} with a checked gain, {k} filled by citations")
            lane_r += extra[:k]
        picked = [dict(r, lane="landmark") for r in land] + [dict(r, lane="recent") for r in lane_r]
        if len(picked) < PER_FAMILY:
            ids = {r["rid"] for r in picked}
            fill = sorted([r for r in cands if r["rid"] not in ids], key=cit, reverse=True)[:PER_FAMILY - len(picked)]
            if fill:
                notes.append(f"{len(fill)} filled from older techniques by citations")
            picked += [dict(r, lane="landmark") for r in fill]
        if len(picked) < PER_FAMILY:
            notes.append(f"only {len(picked)} eligible general-purpose techniques")
        if fam == "training":
            byt = {r["tid"]: r for r in rows}
            picked = [dict(byt[RL_OVERRIDE[r["tid"]]], lane=r["lane"], note="RL adjustment") if r["tid"] in RL_OVERRIDE
                      else r for r in picked]
        used |= {r["rid"] for r in picked}
        sel += [dict(r, family=fam, rank=i + 1) for i, r in enumerate(picked)]
        report.append({"family": fam, "eligible_general": len(cands),
                       "eligible_domain": len({r["rid"] for r in base if r["family_ok"].strip() == fam
                                               and r.get("general_purpose", "").strip() != "yes"}),
                       "selected": len(picked), "landmarks": sum(r["lane"] == "landmark" for r in picked),
                       "recent": sum(r["lane"] == "recent" for r in picked), "notes": "; ".join(notes)})
    cols = ["family", "rank", "lane", "tid", "rid", "title", "year", "citations", "technique_name",
            "core_idea_checked", "target_dim", "gain_metric", "higher_is_better", "baseline_name", "baseline_value",
            "technique_value", "rel_gain", "gain_pct", "gain_quote", "quote_level", "gain_note",
            "effect_correctness", "effect_correctness_headline", "code_public", "code_url", "fulltext_source",
            "benchmarks", "models", "note"]
    with open(TOP / "selected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sel)
    # Domain-specific techniques: listed separately (they cannot run on the shared benchmarks).
    dom = [r for r in base if r.get("general_purpose", "").strip() != "yes"]
    by: dict[str, list[dict]] = {}
    for r in sorted(dom, key=cit, reverse=True):
        d = r.get("domain", "").strip() or "other"
        if r["rid"] not in {x["rid"] for x in by.get(d, [])}:
            by.setdefault(d, []).append(r)
    with open(TOP / "domain_specific.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "family_ok"] + cols[3:], extrasaction="ignore")
        w.writeheader()
        for d, rs in sorted(by.items()):
            for r in rs[:3]:
                w.writerow(dict(r, domain=d))
    with open(TOP / "selection_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    for r in report:
        print(r)
    print(f"selected {len(sel)} -> {TOP / 'selected.csv'}; domain-specific: {sum(min(len(v), 3) for v in by.values())}")


GEN = REPO / "sok" / "paper" / "generated"
H2H_COLS = ["EP", "LCB", "CWE", "CSE", "EB", "DPS", "RM", "Calls"]
H2H_ALIAS = {"EFF": "DPS", "SEC": "CWE"}  # earlier column names in lit_h2h.csv
FAM_TITLE = {"prompting": "Prompting", "feedback-loop": "Tool feedback", "self-refinement": "Self-refinement",
             "multi-agent": "Multi-agent", "retrieval": "Retrieval", "decoding": "Decoding", "training": "Training"}
DOM_TITLE = {"hdl": "Hardware design (HDL/HLS)", "gpu-kernel": "GPU and accelerator kernels", "plc": "PLC",
             "smart-contract": "Smart contracts", "iac": "Infrastructure as code", "ui": "UI and front end",
             "graphics": "Graphics", "compiler-ir": "Compiler and IR transforms", "embedded": "Embedded",
             "data-science": "Data science"}
DIM = {"R/M": "\\dimRM", "EFF": "\\dimEff", "SEC": "\\dimSec"}


def tex_escape(x: str) -> str:
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_",
           "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in x)


def cmd_table(_a) -> None:
    sel = list(csv.DictReader(open(TOP / "selected.csv")))
    rep = {r["family"]: r for r in csv.DictReader(open(TOP / "selection_report.csv"))}
    dom = list(csv.DictReader(open(TOP / "domain_specific.csv")))
    merged_rows = list(csv.DictReader(open(TOP / "merged.csv")))
    excl = list(csv.DictReader(open(TOP / "exclusions.csv")))
    cnt = {
        "nHlPool": len(merged_rows),
        "nHlEligibleGeneral": sum(int(r["eligible_general"]) for r in rep.values()),
        "nHlEligibleDomain": sum(int(r["eligible_domain"]) for r in rep.values()),
        "nHlExcluded": len(excl),
        "nHlSelected": len(sel),
        # From the written tables when present: the writers re-checked each code link.
        "nHlWithCode": (sum(f.read_text().count("\\ding{51}") for f in (REPO / "sok/paper/sections").glob("13h[1-7]_*.tex"))
                        or sum(r["code_public"].strip() == "yes" for r in sel)),
        "nHlDomainListed": len(dom),
    }
    # Funnel from the candidate rows to the eligible techniques (Section "Highlighted Techniques"), so the counts
    # in the text bridge exactly: rows out of scope at pass 1 get no pass-2 record; then not the authors' technique,
    # duplicates, domain-specific rows without a family, rows excluded under the five rules; the remaining rows
    # count one technique per study and family, and a study highlighted in one family is not counted in another.
    p2 = pass2()
    dup = {t: r.get("duplicate_of", "").split(";")[0].strip() for t, r in p2.items()}
    pool_tids = {r["tid"] for r in merged_rows}

    def is_dup(t: str) -> bool:
        d = dup.get(t, "")
        if not d or d not in p2 or d not in pool_tids or d == t:
            return False
        return not (dup.get(d) == t and t < d)
    excl_t = {r["tid"] for r in excl}
    f1 = [r for r in merged_rows if r["tid"] in p2]
    f2 = [r for r in f1 if r.get("is_technique", "").strip() == "yes"]
    f3 = [r for r in f2 if not is_dup(r["tid"])]
    f4 = [r for r in f3 if r.get("family_ok", "").strip() in FAMILIES]
    f5 = [r for r in f4 if r["tid"] not in excl_t]
    cnt.update({"nHlOutOfScope": len(merged_rows) - len(f1), "nHlNotTechnique": len(f1) - len(f2),
                "nHlDuplicate": len(f2) - len(f3), "nHlNoFamily": len(f3) - len(f4),
                "nHlExcludedApplied": len(f4) - len(f5), "nHlBaseRows": len(f5),
                "nHlSameStudy": len(f5) - cnt["nHlEligibleGeneral"] - cnt["nHlEligibleDomain"]})
    for f, r in rep.items():
        cnt["nHlSel" + "".join(w.capitalize() for w in f.split("-"))] = int(r["selected"])
    # Benchmark usage among the included studies (E7's benchmark choice), from the extraction table.
    import re
    bench: dict[str, str] = {}
    for r in load_rows():
        bench[r["rid"]] = bench.get(r["rid"], "") + " ; " + r["benchmarks"]
    pats = {"HumanEval": r"humaneval", "EvalPlus": r"humaneval\+|humaneval-plus|evalplus|mbpp\+",
            "LiveCodeBench": r"livecodebench", "CWEval": r"cweval", "CyberSecEval": r"cyberseceval|purple llama",
            "EffiBench": r"effibench(?!-x)", "EvalPerf": r"evalperf", "RACE": r"\brace\b", "NoFunEval": r"nofuneval"}
    for k, p in pats.items():
        cnt["nBench" + k] = sum(bool(re.search(p, b, re.I)) for b in bench.values())
    with open(GEN / "top_counts.tex", "w") as fh:
        fh.write("% GENERATED by src/slr/top_techniques.py table. Provisional (built on LLM-assisted extraction).\n")
        for k, v in cnt.items():
            fh.write(f"\\expandafter\\def\\csname cnt@{k}\\endcsname{{{v}}}\n")
    lines = ["{\\scriptsize", "\\begin{longtable}{@{}p{0.17\\linewidth}p{0.07\\linewidth}p{0.42\\linewidth}"
             "p{0.05\\linewidth}p{0.1\\linewidth}p{0.05\\linewidth}@{}}",
             "\\caption{Domain-specific techniques: eligible techniques tied to a narrow artifact or domain, which "
             "cannot run on the shared general-purpose benchmarks of \\S\\ref{sec:empirical-h2h}; up to three per "
             "domain, by Semantic Scholar citations, October 2026. ID = record identifier in "
             "\\texttt{sok/slr/extraction.csv}. Provisional (LLM-assisted screening and extraction).}"
             "\\label{tab:hl-domain}\\\\",
             "\\toprule", "Domain & ID & Title & Year & Family & Cit.\\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Domain & ID & Title & Year & Family & Cit.\\\\", "\\midrule", "\\endhead"]
    for r in dom:
        d = r["domain"]
        dt = DOM_TITLE.get(d, d.split(":", 1)[-1].replace("-", " ").capitalize())
        lines.append(f"{tex_escape(dt)} & {r['rid']} & {tex_escape(r['title'])} & {r['year']} & "
                     f"{FAM_TITLE.get(r.get('family_ok', ''), '')} & {r['citations']}\\\\")
    lines += ["\\bottomrule", "\\end{longtable}}"]
    (GEN / "tab_top_domain.tex").write_text("\n".join(lines) + "\n")
    # Head-to-head result shell: one row per highlighted technique, five shared benchmarks. A cell a
    # paper reports for the same benchmark and metric carries that value as a literature placeholder
    # (sok/slr/top/lit_h2h.csv, quote-checked in notes/evidence/15_empirical.csv); never an estimate.
    lit = {}
    if (TOP / "lit_h2h.csv").exists():
        for r in csv.DictReader(open(TOP / "lit_h2h.csv")):
            mark = "$^{b}$" if "base tests" in r["setting"] else ""
            if r["column"] == "CSE" and "secure rate" in r["setting"].lower():
                mark = "$^{s}$"
            lit[(r["h2h_id"], H2H_ALIAS.get(r["column"], r["column"]))] = f"\\lit{{{tex_escape(r['value'])}}}{{{r['cite_key']}}}{mark}"
    n = len(H2H_COLS)
    h = ["{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{longtable}{@{}p{0.05\\linewidth}p{0.25\\linewidth}*{%d}{>{\\centering\\arraybackslash}p{0.068\\linewidth}}@{}}" % n,
         "\\caption{\\textbf{E7. Head-to-head of the highlighted techniques} (\\S\\ref{sec:empirical-h2h}): one "
         "model, a single-shot and a best-of-$n$ reference, every output scored on all four dimensions. \\dimCor: EP = HumanEval+/MBPP+ "
         "pass@1, LCB = LiveCodeBench pass@1 (post-cutoff split). \\dimSec: CWE = CWEval func-sec@1, CSE = "
         "CyberSecEval insecure-code rate (lower is better). \\dimEff: EB = EffiBench normalized execution time "
         "(lower is better), DPS = EvalPerf DPS$_\\text{norm}$. \\dimRM: RM = Pylint score on the EP and LCB outputs. "
         "Calls = mean model calls per task. Runs pending. Orange italic values are literature placeholders taken "
         "from the cited work under different settings (other models and versions); they are not our results and "
         "will be replaced by our measurements; they are printed as the paper prints them (some papers report "
         "fractions, not percentages). $^{b}$ = original HumanEval or MBPP tests, not the extended EvalPlus tests. $^{s}$ = the paper "
         "prints the secure rate (higher is better), the complement of CSE. "
         "A dash (--) marks a cell to be measured.}\\label{tab:emp-h2h}\\\\",
         "\\toprule", "ID & Technique & EP & LCB & CWE & CSE & EB & DPS & RM & Calls\\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "ID & Technique & EP & LCB & CWE & CSE & EB & DPS & RM & Calls\\\\", "\\midrule", "\\endhead",
         "-- & T0 single shot (reference) & " + " & ".join(["\\pend"] * n) + "\\\\",
         "-- & best-of-$n$ (reference) & " + " & ".join(["\\pend"] * n) + "\\\\"]
    fam = None
    for i, r in enumerate(sel):
        if r["family"] != fam:
            fam = r["family"]
            h.append(f"\\midrule\\multicolumn{{{n + 2}}}{{@{{}}l}}{{\\textit{{{FAM_TITLE[fam]}}}}}\\\\")
        name = r["technique_name"].split("(")[0].split(":")[0].strip()[:48]
        hid = f"H{i + 1:02d}"
        cells = [lit.get((hid, c), "\\pend") for c in H2H_COLS]
        h.append(f"{hid} & {tex_escape(name)} & " + " & ".join(cells) + "\\\\")
    h += ["\\bottomrule", "\\end{longtable}}"]
    (GEN / "tab_h2h.tex").write_text("\n".join(h) + "\n")
    with open(TOP / "h2h_ids.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["h2h_id", "tid", "family", "technique_name", "code_public", "code_url"])
        for i, r in enumerate(sel):
            w.writerow([f"H{i + 1:02d}", r["tid"], r["family"], r["technique_name"], r["code_public"], r["code_url"]])
    print(cnt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("cites", "pool", "rank", "table"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"cites": cmd_cites, "pool": cmd_pool, "rank": cmd_rank, "table": cmd_table}[a.cmd](a)
