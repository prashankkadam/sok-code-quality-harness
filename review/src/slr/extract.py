#!/usr/bin/env python3
"""Full-text eligibility + data extraction.

  extract.py fetch   --run main --ids forward_ids.txt     # full texts -> .cache/fulltext_slr/<rid>.txt
  extract.py submit  --run main --ids forward_ids.txt --tag ft1
  extract.py collect --run main --tag ft1 [--wait]
  extract.py table   --run main                          # writes sok/slr/extraction.csv + side_lists.csv

Full text is never cut: the whole extracted PDF text goes to the model (Opus 5.5, 1M context).
Sources, in order: arXiv PDF; open-access PDF via Unpaywall / Crossref links; otherwise the abstract
only, which is recorded (`fulltext_source=abstract-only`) and forces confidence=low.
Full texts are cached locally (copyright); a manifest with source URL and SHA-256 is committed.
"""
from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, REPO, http_get, now_iso  # noqa: E402
from quotecheck import pdf_to_text  # noqa: E402
from screen import API, MODEL, cost, headers, log_spend, parse_json_array, total_spend  # noqa: E402

FT = CACHE / "fulltext_slr"
SLR = REPO / "sok" / "slr"


def run_dir(run):
    return SLR / "runs" / run


def all_records(run) -> dict:
    recs = {}
    for f in sorted(run_dir(run).glob("dedup_all*.jsonl")) + sorted(run_dir(run).glob("snowball_*.jsonl")) + \
            sorted(run_dir(run).glob("other_sources*.jsonl")) + sorted(run_dir(run).glob("venue_sweep_new*.jsonl")):
        if f.name.endswith("_raw.jsonl"):
            continue
        for l in f.open():
            r = json.loads(l)
            if "rid" in r:
                recs[r["rid"]] = r
    return recs


def get_fulltext(r: dict) -> tuple[str, str]:
    """Return (text, source); network or TLS failures on one source fall through to the next."""
    try:
        return _get_fulltext(r)
    except Exception as exc:  # noqa: BLE001
        return "", f"abstract-only (fetch error: {type(exc).__name__})"


def _get_fulltext(r: dict) -> tuple[str, str]:
    if r.get("arxiv_id"):
        st, data = http_get(f"https://arxiv.org/pdf/{r['arxiv_id']}", binary=True, timeout=180)
        if st == 200 and data[:4] == b"%PDF":
            t = pdf_to_text(data)
            if len(t) > 3000:
                return t, f"arxiv-pdf:{r['arxiv_id']}"
    for doi in (r.get("dois") or ([r["doi"]] if r.get("doi") else [])):
        st, body = http_get(f"https://api.unpaywall.org/v2/{doi}", params={"email": os.environ.get("CONTACT_EMAIL", "")})
        if st == 200:
            d = json.loads(body)
            for loc in [d.get("best_oa_location")] + (d.get("oa_locations") or []):
                if loc and loc.get("url_for_pdf"):
                    try:
                        st2, data = http_get(loc["url_for_pdf"], binary=True, timeout=180, retries=2)
                    except Exception:  # noqa: BLE001
                        continue
                    if st2 == 200 and data[:4] == b"%PDF":
                        t = pdf_to_text(data)
                        if len(t) > 3000:
                            return t, f"oa-pdf:{loc['url_for_pdf']}"
        if doi.startswith("10.18653/"):  # ACL Anthology
            aid = doi.split("/v1/", 1)[-1]
            st2, data = http_get(f"https://aclanthology.org/{aid}.pdf", binary=True, timeout=180)
            if st2 == 200 and data[:4] == b"%PDF":
                return pdf_to_text(data), f"acl-anthology-pdf:{aid}"
    return "", "abstract-only"


def cmd_fetch(a):
    recs = all_records(a.run)
    ids = [l.strip() for l in (run_dir(a.run) / a.ids).open() if l.strip()]
    FT.mkdir(parents=True, exist_ok=True)
    man_p = run_dir(a.run) / (a.manifest or "fulltext_manifest.csv")
    man = {r["rid"]: r for r in csv.DictReader(open(man_p))} if man_p.exists() else {}
    for i, rid in enumerate(ids):
        if rid in man and (FT / f"{rid}.txt").exists():
            continue
        r = recs[rid]
        t, src = get_fulltext(r)
        if t:
            (FT / f"{rid}.txt").write_text(t)
        man[rid] = {"rid": rid, "title": r.get("title", "")[:200], "source": src, "chars": len(t),
                    "sha256": hashlib.sha256(t.encode()).hexdigest() if t else "", "fetched_at": now_iso()}
        if i % 25 == 0:
            print(f"{i}/{len(ids)} {rid} {src} {len(t)}")
            _write_manifest(man_p, man)
    _write_manifest(man_p, man)
    from collections import Counter
    print(Counter(v["source"].split(":")[0] for v in man.values()))
    return 0


def _write_manifest(p, man):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rid", "title", "source", "chars", "sha256", "fetched_at"])
        w.writeheader()
        for k in sorted(man):
            w.writerow(man[k])


def user_message(r: dict, text: str, src: str) -> str:
    meta = (f"Record id: {r['rid']}\nTitle: {r.get('title')}\nYear: {r.get('year')}\n"
            f"Venue (metadata): {'; '.join(r.get('venues') or [r.get('venue') or ''])}\n"
            f"Full-text source: {src}\n")
    if text:
        return meta + "\nFULL TEXT (extracted from PDF; complete, not truncated):\n" + text
    return meta + "\nFULL TEXT NOT AVAILABLE. Abstract only:\n" + (r.get("abstract") or "[no abstract]") + \
        "\n\nDecide eligibility from the abstract and set confidence to low."


def cmd_submit(a):
    if total_spend() >= a.cap:
        print("spend cap reached")
        return 2
    recs = all_records(a.run)
    ids = [l.strip() for l in (run_dir(a.run) / a.ids).open() if l.strip()]
    man = {r["rid"]: r for r in csv.DictReader(open(run_dir(a.run) / "fulltext_manifest.csv"))}
    sys_text = (run_dir(a.run) / "prompts" / a.prompt).read_text()
    reqs = []
    for rid in ids:
        r = recs[rid]
        p = FT / f"{rid}.txt"
        text = p.read_text() if p.exists() else ""
        src = man.get(rid, {}).get("source", "abstract-only")
        params = {"model": a.model, "max_tokens": 8000 if a.model == MODEL else 1500,
                  "system": [{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
                  "messages": [{"role": "user", "content": user_message(r, text, src)}]}
        if a.model == MODEL:
            params["output_config"] = {"effort": a.effort}
        reqs.append({"custom_id": f"{a.tag}-{rid}", "params": params})
    out = run_dir(a.run) / "extraction" / f"batch_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "requests_index.jsonl", "w") as fh:
        for q in reqs:
            fh.write(json.dumps({"custom_id": q["custom_id"], "chars": len(q["params"]["messages"][0]["content"])}) + "\n")
    est_in = sum(len(q["params"]["messages"][0]["content"]) for q in reqs) / 3.6
    print(f"requests {len(reqs)}; est. input tokens {est_in/1e6:.1f}M; est. input cost (batch) ${est_in/1e6*2:.2f}")
    if a.dry:
        return 0
    # Batches are limited in size; send in chunks of 400 requests.
    ids_out = []
    for i in range(0, len(reqs), 400):
        payload = {"requests": reqs[i:i + 400]}
        r = requests.post(f"{API}/messages/batches", headers=headers(), json=payload, timeout=900)
        d = r.json()
        ids_out.append(d.get("id"))
        print("batch", d.get("id"), d.get("processing_status"), d.get("error"))
    (out / "batch.json").write_text(json.dumps({"batch_ids": ids_out, "submitted": now_iso(), "model": a.model,
                                                "prompt": a.prompt, "effort": a.effort}, indent=1))
    return 0


def cmd_collect(a):
    out = run_dir(a.run) / "extraction" / f"batch_{a.tag}"
    b = json.loads((out / "batch.json").read_text())
    results = []
    for bid in b["batch_ids"]:
        while True:
            st = requests.get(f"{API}/messages/batches/{bid}", headers=headers(), timeout=120).json()
            print(now_iso(), bid, st.get("processing_status"), st.get("request_counts"))
            if st.get("processing_status") == "ended":
                break
            if not a.wait:
                return 3
            time.sleep(60)
        res = requests.get(st["results_url"], headers=headers(), timeout=900)
        (out / f"results_{bid}.jsonl").write_bytes(res.content)
        results += [json.loads(l) for l in res.text.splitlines()]
    model = b.get("model", MODEL)
    usum, calls, ok, bad, usd = {}, 0, 0, [], 0.0
    with open(out / "parsed.jsonl", "w") as fh:
        for r in results:
            rid = r["custom_id"].split("-", 1)[1]
            if r["result"]["type"] != "succeeded":
                bad.append(rid)
                continue
            msg = r["result"]["message"]
            calls += 1
            for k, v in (msg.get("usage") or {}).items():
                if isinstance(v, int):
                    usum[k] = usum.get(k, 0) + v
            usd += cost(msg.get("usage") or {}, batch=True, model=model)
            text = "".join(c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text")
            m = re.search(r"\{.*\}", text, re.S)
            try:
                obj = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                obj = None
            if not obj:
                bad.append(rid)
                continue
            fh.write(json.dumps({"rid": rid, "extraction": obj, "stop_reason": msg.get("stop_reason"),
                                 "model": model, "at": now_iso()}) + "\n")
            ok += 1
    log_spend(f"fulltext-batch:{a.tag}", calls, usum, usd, model=model)
    (out / "failed_rids.txt").write_text("\n".join(bad))
    print(f"parsed {ok}; failed {len(bad)}; cost ${usd:.2f}; total ${total_spend():.2f}")
    return 0


COLS = ["study_id", "rid", "title", "authors", "year", "venue", "venue_type", "doi", "arxiv_id", "url",
        "study_type", "technique_name", "family_v0", "stage", "signal_source", "tools", "iterations",
        "dims_targeted", "dims_measured", "correctness_measured", "metrics", "benchmarks", "languages", "models",
        "baselines", "effect_target", "effect_target_headline", "effect_correctness", "effect_correctness_headline",
        "effect_other_dims", "compute_reported", "contamination_handled", "human_eval", "human_eval_details",
        "artifact_url", "reimplementable"] + [f"qa_{i}" for i in range(1, 11)] + \
       ["qa_total", "fulltext_source", "identified_via", "notes", "extractor", "confidence", "needs_human_review"]


def cmd_table(a):
    recs = all_records(a.run)
    man = {r["rid"]: r for r in csv.DictReader(open(run_dir(a.run) / "fulltext_manifest.csv"))}
    parsed, stage_a = {}, {}
    for f in sorted((run_dir(a.run) / "extraction").glob("batch_*/parsed.jsonl")):
        meta = json.loads((f.parent / "batch.json").read_text())
        for l in f.open():
            x = json.loads(l)
            if meta.get("prompt") == "screen_ft.md":
                stage_a[x["rid"]] = x
            else:
                parsed[x["rid"]] = x  # later batches override earlier ones
    for rid, x in stage_a.items():
        e = x["extraction"]
        if rid not in parsed and e.get("eligibility") == "exclude":
            parsed[rid] = {"rid": rid, "extraction": {"eligibility": "exclude", "exclusion_criterion": e.get("criterion", ""),
                                                      "side_list": e.get("side_list", "none"),
                                                      "eligibility_reason": "[stage A, Haiku 5.5] " + e.get("reason", ""),
                                                      "confidence": "medium"}}
    rows, side, excl = [], [], []
    for rid, x in sorted(parsed.items()):
        e, r = x["extraction"], recs.get(rid, {})
        src = man.get(rid, {}).get("source", "abstract-only")
        via = {"R": "database-search", "V": "venue-sweep", "O": "section-level-search", "S": "snowballing"}.get(rid[0], "other")
        if e.get("eligibility") != "include":
            side.append({"rid": rid, "title": r.get("title", ""), "year": r.get("year"), "doi": r.get("doi", ""),
                         "arxiv_id": r.get("arxiv_id", ""), "side_list": e.get("side_list", "none"),
                         "exclusion_criterion": e.get("exclusion_criterion", ""),
                         "reason": e.get("eligibility_reason", ""), "confidence": e.get("confidence", ""),
                         "fulltext_source": src, "identified_via": via})
            continue
        qa = e.get("qa") or {}
        qv = [qa.get(f"QA{i}") for i in range(1, 11)]
        qt = sum(float(v) for v in qv if isinstance(v, (int, float)))
        for j, t in enumerate(e.get("techniques") or [{}]):
            conf = e.get("confidence", "")
            if src == "abstract-only":
                conf = "low"
            rows.append({
                "study_id": f"{rid}" + (f".{j + 1}" if len(e.get("techniques") or []) > 1 else ""), "rid": rid,
                "title": e.get("title") or r.get("title", ""), "authors": "; ".join(r.get("authors") or []),
                "year": r.get("year") or e.get("year"), "venue": "; ".join(r.get("venues") or [r.get("venue", "")]),
                "venue_type": e.get("venue_type", ""), "doi": r.get("doi", ""), "arxiv_id": r.get("arxiv_id", ""),
                "url": f"https://arxiv.org/abs/{r['arxiv_id']}" if r.get("arxiv_id") else (f"https://doi.org/{r['doi']}" if r.get("doi") else ""),
                "study_type": e.get("study_type", ""), "technique_name": t.get("technique_name", ""),
                "family_v0": t.get("family_v0", ""), "stage": t.get("stage", ""),
                "signal_source": ";".join(t.get("signal_source") or []), "tools": t.get("tools", ""),
                "iterations": t.get("iterations", ""), "dims_targeted": ";".join(t.get("dims_targeted") or []),
                "dims_measured": ";".join(t.get("dims_measured") or []),
                "correctness_measured": t.get("correctness_measured", ""), "metrics": t.get("metrics", ""),
                "benchmarks": t.get("benchmarks", ""), "languages": t.get("languages", ""), "models": t.get("models", ""),
                "baselines": t.get("baselines", ""), "effect_target": t.get("effect_target", ""),
                "effect_target_headline": t.get("effect_target_headline", ""),
                "effect_correctness": t.get("effect_correctness", ""),
                "effect_correctness_headline": t.get("effect_correctness_headline", ""),
                "effect_other_dims": json.dumps(t.get("effect_other_dims") or {}),
                "compute_reported": ";".join(e.get("compute_reported") or []),
                "contamination_handled": e.get("contamination_handled", ""), "human_eval": e.get("human_eval", ""),
                "human_eval_details": e.get("human_eval_details", ""), "artifact_url": e.get("artifact_url", ""),
                "reimplementable": e.get("reimplementable", ""),
                **{f"qa_{i}": qv[i - 1] if qv[i - 1] is not None else "" for i in range(1, 11)},
                "qa_total": qt, "fulltext_source": src, "identified_via": via, "notes": e.get("notes", ""),
                "extractor": f"{MODEL} (LLM-assisted, unreviewed)", "confidence": conf,
                "needs_human_review": conf == "low"})
    with open(SLR / "extraction.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(run_dir(a.run) / "side_lists.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(side[0].keys()) if side else ["rid"])
        w.writeheader()
        w.writerows(side)
    from collections import Counter
    print(f"included studies: {len({r['rid'] for r in rows})}; technique rows: {len(rows)}; excluded at full text: {len(side)}")
    print("families:", Counter(r["family_v0"] for r in rows))
    print("side lists:", Counter(s["side_list"] for s in side))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", cmd_fetch), ("submit", cmd_submit), ("collect", cmd_collect), ("table", cmd_table)):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--ids", default="forward_ids.txt")
        p.add_argument("--tag", default="ft1")
        p.add_argument("--effort", default="low")
        p.add_argument("--cap", type=float, default=75.0)
        p.add_argument("--wait", action="store_true")
        p.add_argument("--dry", action="store_true")
        p.add_argument("--model", default=MODEL)
        p.add_argument("--manifest", default=None)
        p.add_argument("--prompt", default="extract_ft.md")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
