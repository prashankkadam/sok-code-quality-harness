#!/usr/bin/env python3
"""Build the quasi-gold standard from the seed set, and compute recall.

  qgs.py build                 # writes sok/slr/qgs.csv (frozen before S1 results are inspected)
  qgs.py recall --run DATE     # recall of the deduplicated search results against the QGS

QGS = seed-set technique studies that meet IC1-IC4 (technique applied to an LLM, LLM-generated or
edited code, a quantitative non-functional outcome, full paper). Seed papers that are
correctness-only, vision papers without results, or secondary studies are excluded; the reason is
recorded in sok/slr/qgs_decisions.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bibgen import arxiv_meta, from_crossref  # noqa: E402
from common import REPO, norm_title, now_iso, read_jsonl  # noqa: E402

SLR = REPO / "sok" / "slr"

# (identifier, family, include, reason) — decisions from the seed-set descriptions and the papers' abstracts.
SEED = [
    ("arXiv:2303.06689", "prompting", False, "Self-planning: correctness-only outcomes (EC5)"),
    ("arXiv:2305.06599", "prompting", False, "SCoT: correctness-only outcomes (EC5)"),
    ("arXiv:2407.07064", "prompting", True, "prompting techniques evaluated on security of generated code"),
    ("arXiv:2502.06039", "prompting", True, "prompt engineering for secure code; security outcome"),
    ("arXiv:2412.20545", "prompting", True, "CodePromptEval: prompting effects on correctness and quality"),
    ("arXiv:2411.00006", "prompting", False, "personality-guided generation: correctness-only outcomes (EC5)"),
    ("arXiv:2509.10099", "prompting", True, "prompting for energy-efficient code; efficiency outcome"),
    ("arXiv:2508.14419", "feedback-loop", True, "Pylint+Bandit loop; quality and security outcomes"),
    ("arXiv:2309.12938", "feedback-loop", True, "CORE: CodeQL/SonarQube-driven fixes; quality outcome"),
    ("arXiv:2312.00024", "feedback-loop", True, "FDSP: Bandit feedback; security outcome"),
    ("arXiv:2405.03927", "feedback-loop", True, "Codexity: Infer/CppCheck feedback; security outcome"),
    ("arXiv:2506.10330", "feedback-loop", True, "SonarQube-guided refinement; quality outcome"),
    ("arXiv:2509.16275", "feedback-loop", True, "SecureFixAgent: Bandit-guided repair; security outcome"),
    ("arXiv:2405.01103", "feedback-loop", False, "LLMSecGuard: vision paper without quantitative results (IC3)"),
    ("arXiv:2403.16792", "feedback-loop", False, "ProCoder: correctness-only outcomes (EC5)"),
    ("arXiv:2603.23613", "feedback-loop", True, "LLMLOOP: compiler/static-analysis/test loop; quality outcomes"),
    ("arXiv:2409.10737", "multi-agent", True, "AutoSafeCoder: multi-agent security pipeline; security outcome"),
    ("arXiv:2303.17651", "self-refinement", True, "Self-Refine: includes code readability and optimization tasks"),
    ("arXiv:2304.05128", "self-refinement", False, "Self-Debug: correctness-only outcomes (EC5)"),
    ("arXiv:2303.11366", "self-refinement", False, "Reflexion: correctness-only outcomes (EC5)"),
    ("arXiv:2305.11738", "self-refinement", False, "CRITIC: correctness-only for code (EC5)"),
    ("arXiv:2306.09896", "self-refinement", False, "Self-repair analysis: correctness-only outcomes (EC5)"),
    ("arXiv:2405.15189", "self-refinement", True, "EffiLearner: efficiency self-optimization; efficiency outcome"),
    ("arXiv:2405.11403", "multi-agent", False, "MapCoder: correctness-only outcomes (EC5)"),
    ("arXiv:2312.13010", "multi-agent", False, "AgentCoder: correctness-only outcomes (EC5)"),
    ("arXiv:2307.07924", "multi-agent", False, "ChatDev: no non-functional quality outcome as defined in §3"),
    ("arXiv:2308.00352", "multi-agent", False, "MetaGPT: no non-functional quality outcome as defined in §3"),
    ("arXiv:2304.07590", "multi-agent", False, "Self-collaboration: correctness-only outcomes (EC5)"),
    ("arXiv:2608.25457", "multi-agent", True, "MACGen: multi-agent secure generation; security outcome"),
    ("arXiv:2603.22717", "multi-agent", True, "Multi-LLMSecCodeEval: multi-LLM pipelines; security outcome"),
    ("arXiv:2207.05987", "retrieval", False, "DocPrompting: correctness-only outcomes (EC5)"),
    ("arXiv:2303.12570", "retrieval", False, "RepoCoder: correctness/similarity outcomes only (EC5)"),
    ("arXiv:2406.14497", "retrieval", False, "CodeRAG-Bench: correctness-only benchmark (EC5)"),
    ("arXiv:2410.01488", "retrieval", True, "SecCoder: retrieval for secure code; security outcome"),
    ("arXiv:2503.13654", "retrieval", True, "SOSecure: retrieval of SO discussions; security outcome"),
    ("arXiv:2510.18204", "retrieval", True, "RESCUE: retrieval-augmented secure code; security outcome"),
    ("arXiv:2201.11227", "decoding", False, "Synchromesh: validity/correctness outcomes only (EC5)"),
    ("arXiv:2207.10397", "decoding", False, "CodeT: correctness-only outcomes (EC5)"),
    ("arXiv:2302.08468", "decoding", False, "LEVER: correctness-only outcomes (EC5)"),
    ("arXiv:2211.16490", "decoding", False, "Coder-Reviewer reranking: correctness-only outcomes (EC5)"),
    ("arXiv:2405.00218", "decoding", True, "constrained decoding for secure code; security outcome"),
    ("doi:10.1145/3650212.3680371", "decoding", True, "CoSec: supervised co-decoding for security; security outcome"),
    ("arXiv:2504.09246", "decoding", False, "type-constrained decoding: compilability/correctness outcomes (EC5)"),
    ("arXiv:2302.05319", "training", True, "SVEN: security hardening via prefix tuning; security outcome"),
    ("arXiv:2402.09497", "training", True, "SafeCoder: security instruction tuning; security outcome"),
    ("arXiv:2302.07867", "training", True, "PIE: performance-improving edits; efficiency outcome"),
    ("arXiv:2403.09032", "training", True, "CodeUltraFeedback: preference data incl. non-functional preferences"),
    ("arXiv:2410.05605", "training", True, "CodeDPO: correctness and efficiency preference optimization"),
    ("arXiv:2410.10209", "training", True, "EffiCoder: efficiency-aware fine-tuning; efficiency outcome"),
    ("arXiv:2412.17264", "training", True, "ACECode: RL for efficiency and correctness; efficiency outcome"),
    ("arXiv:2411.12882", "training", True, "ProSec: proactive security alignment; security outcome"),
    ("arXiv:2505.23387", "training", True, "Afterburner: RL for efficiency; efficiency outcome"),
    ("arXiv:2505.22704", "training", True, "REAL: RL with program-analysis rewards; quality/security outcomes"),
    ("arXiv:2507.19060", "training", True, "PurpCode: secure-code reasoning alignment; security outcome"),
]


def cmd_build(_a) -> int:
    rows, dec = [], []
    for ident, fam, inc, why in SEED:
        if ident.startswith("arXiv:"):
            m = arxiv_meta(ident.split(":", 1)[1])
            title = m["title"] if m else ""
        else:
            c = from_crossref(ident.split(":", 1)[1])
            title = c["title"] if c else ""
        dec.append({"identifier": ident, "title": title, "family": fam, "in_qgs": inc, "reason": why})
        if inc:
            rows.append({"key": ident.replace(":", "_"), "title": title, "doi_or_arxiv": ident, "family": fam})
    with open(SLR / "qgs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "title", "doi_or_arxiv", "family"])
        w.writeheader()
        w.writerows(rows)
    with open(SLR / "qgs_decisions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["identifier", "title", "family", "in_qgs", "reason"])
        w.writeheader()
        w.writerows(dec)
    print(f"QGS: {len(rows)} of {len(SEED)} seed technique papers; frozen {now_iso()}")
    return 0


def load_jsonl(p: Path) -> list[dict]:
    return read_jsonl(p) if (p.exists() or Path(str(p) + ".gz").exists()) else []


def recall(records: list[dict], qgs: list[dict]) -> tuple[int, list[dict]]:
    ax = {r.get("arxiv_id") for r in records if r.get("arxiv_id")}
    dois = {r.get("doi") for r in records if r.get("doi")}
    titles = {norm_title(r.get("title", "")) for r in records}
    found, missed = 0, []
    for q in qgs:
        ident = q["doi_or_arxiv"]
        hit = (ident.startswith("arXiv:") and ident.split(":", 1)[1] in ax) or \
              (ident.startswith("doi:") and ident.split(":", 1)[1].lower() in dois) or \
              norm_title(q["title"]) in titles
        if hit:
            found += 1
        else:
            missed.append(q)
    return found, missed


def cmd_recall(a) -> int:
    run = SLR / "runs" / a.run
    qgs = list(csv.DictReader(open(SLR / "qgs.csv")))
    out = {"qgs_size": len(qgs), "computed_at": now_iso(), "per_source": {}}
    for s in ("openalex", "s2", "arxiv"):
        recs = load_jsonl(run / f"records_{a.string}_{s}.jsonl")
        f, _ = recall(recs, qgs)
        out["per_source"][s] = {"records": len(recs), "found": f, "recall": round(f / len(qgs), 3)}
    combined = load_jsonl(run / a.combined)
    f, missed = recall(combined, qgs)
    out["combined"] = {"file": a.combined, "records": len(combined), "found": f, "recall": round(f / len(qgs), 3),
                       "missed": [m["doi_or_arxiv"] + " | " + m["title"] for m in missed]}
    (run / f"qgs_recall_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("recall")
    p.add_argument("--run", required=True)
    p.add_argument("--string", default="S1")
    p.add_argument("--combined", default="dedup_S1.jsonl")
    p.add_argument("--tag", default="S1")
    p.set_defaults(fn=cmd_recall)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
