#!/usr/bin/env python3
"""Compile one section on its own with acmart, so parallel writers can check LaTeX, citations
and length without touching the shared build.

  secbuild.py 08_feedback_loops        # builds sok/paper/build/sec_08_feedback_loops/sec.pdf

Reports: LaTeX errors, citation keys that are unresolved or unverified, the page count of the
section alone (acmsmall), the word count of its prose, and overfull boxes.
Cross-references to other sections are expected to be undefined here and are not reported.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bibgen import load_resolved, to_bibtex  # noqa: E402
from common import PAPER  # noqa: E402

CITE = re.compile(r"\\(?:cite[tp]?|citeauthor|citeyear|citealp|nocite)\*?(?:\[[^\]]*\])*\{([^}]*)\}")
LIT = re.compile(r"\\lit\{[^{}]*\}\{([^}]*)\}")


def main() -> int:
    sec = sys.argv[1].replace(".tex", "")
    src = PAPER / "sections" / f"{sec}.tex"
    out = PAPER / "build" / f"sec_{sec}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    body = src.read_text()
    nocomment = re.sub(r"(?<!\\)%.*", "", body)
    keys = set()
    for m in list(CITE.finditer(nocomment)) + list(LIT.finditer(nocomment)):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    bad, entries = [], []
    for k in sorted(keys):
        rec = load_resolved(k)
        if not rec or rec.get("status") != "verified":
            bad.append(k)
        else:
            entries.append(to_bibtex(rec))
    (out / "sec.bib").write_text("\n".join(entries))
    for f in ("macros.tex", "local-macros.tex"):
        shutil.copy(PAPER / f, out / f)
    for d in ("figures", "generated"):
        if (PAPER / d).exists():
            shutil.copytree(PAPER / d, out / d)
    shutil.copytree(PAPER / "sections", out / "sections")
    (out / "sec.tex").write_text(
        "\\documentclass[acmsmall,review,anonymous,screen]{acmart}\n\\input{macros}\n\\input{local-macros}\n"
        "\\input{generated/counts}\n"
        "\\title{Section test build}\\author{Anon}\\affiliation{\\institution{A}\\country{A}}\n"
        "\\begin{document}\\maketitle\n\\pagemark{secstart}\n"
        f"\\input{{sections/{sec}}}\n\\pagemark{{secend}}\n"
        "\\bibliographystyle{ACM-Reference-Format}\\bibliography{sec}\n\\end{document}\n")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "sec.tex"],
                       cwd=out, capture_output=True, text=True)
    log = (out / "sec.log").read_text(errors="replace") if (out / "sec.log").exists() else r.stdout
    errors = [l for l in log.splitlines() if l.startswith("!") or (re.match(r"^\./", l) and ("Error" in l or "Undefined control sequence" in l))]
    overfull = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)[^\n]*lines? (\d+)", log)
    aux = (out / "sec.aux").read_text(errors="replace") if (out / "sec.aux").exists() else ""
    ps = re.search(r"\\gdef\s*\\pm@secstart\{(\d+)\}", aux)
    pe = re.search(r"\\gdef\s*\\pm@secend\{(\d+)\}", aux)
    pages = (int(pe.group(1)) - int(ps.group(1)) + 1) if ps and pe else None
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]+", re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", nocomment)))
    undefined_cites = sorted(set(re.findall(r"Citation `([^']+)' .*undefined", log)))
    print(f"section: {sec}\npdf: {out / 'sec.pdf'}\nlatex errors: {len(errors)}")
    for e in errors[:15]:
        print("  ", e)
    print(f"pages (section alone, acmsmall, incl. its tables/figures): {pages}\nwords (approx.): {words}")
    print(f"cite keys: {len(keys)}; unresolved/unverified keys: {bad}")
    print(f"undefined citations in build: {undefined_cites}")
    big = [(float(a), b) for a, b in overfull if float(a) > 10]
    print(f"overfull hboxes > 10pt: {len(big)} {big[:8]}")
    return 1 if errors or bad else 0


if __name__ == "__main__":
    sys.exit(main())
