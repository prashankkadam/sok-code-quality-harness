#!/usr/bin/env python3
"""Generate appendix LaTeX from the run files: exact per-source query strings and the frozen prompts.
  appendix_gen.py --run main  -> sok/paper/generated/app_queries.tex, app_prompts.tex"""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import REPO  # noqa: E402

def esc(s):
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}"),
                 ("~", "\\textasciitilde{}"), ("^", "\\^{}"), ("<", "\\textless{}"), (">", "\\textgreater{}"), ("|", "\\textbar{}")):
        s = s.replace(a, b) if a != "\\" else s
    s = s.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    return s

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
run = REPO / "sok" / "slr" / "runs" / a.run
gen = REPO / "sok" / "paper" / "generated"
out = []
names = {"S1": "S1 (iteration 0)", "S1r1": "S1 after calibration iteration 1", "S2r1": "S2 after calibration iteration 1"}
for s in ("S1", "S1r1", "S2r1"):
    q = json.loads((run / f"queries_{s}.json").read_text())
    out.append(f"\\paragraph{{{names[s]}.}}")
    for src, m in q.items():
        label = {"openalex": "OpenAlex", "s2": "Semantic Scholar (bulk)", "arxiv": "arXiv"}[src]
        hits = m.get("raw_hits")
        if src == "arxiv":
            strings = m["query_strings"]
        else:
            strings = [m.get("filter") or m.get("query_string")]
        extra = f" Date filter: {esc(json.dumps(m.get('params', {}).get('publicationDateOrYear', '')))}." if src == "s2" else ""
        out.append(f"\\noindent\\textit{{{label}}} (raw hits {hits}).{extra}")
        for st in strings:
            out.append("{\\scriptsize\\ttfamily\\raggedright " + esc(st) + "\\par}\\smallskip")
(gen / "app_queries.tex").write_text("\n".join(out) + "\n")
p = (run / "prompts" / "screen_ta.md").read_text()
lines = []
for l in p.splitlines():
    l = esc(l)
    lines.append(("{}" + l if l.startswith("[") else l) if l.strip() else "\\par")
(gen / "app_prompts.tex").write_text("{\\scriptsize\\ttfamily\\raggedright\n" + "\\par\n".join(x for x in lines if x != "\\par") + "\\par}\n")
print("ok")
