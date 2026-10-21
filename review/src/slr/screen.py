#!/usr/bin/env python3
"""LLM-assisted title/abstract screening.

Screeners:
  claude : claude-opus-5-5 via the Claude API (Message Batches, prompt caching, effort=low)
  gpt    : openai/gpt-6.1-sol via Vercel AI Gateway pinned only:["openai"]  (BLOCKED: no gateway key)

  screen.py pilot   --run main --input dedup_all.jsonl --n 30
  screen.py submit  --run main --input dedup_all.jsonl [--tag main]
  screen.py collect --run main --tag main
  screen.py merge   --run main            # decisions_ta.csv with the recall-favouring rule

Every request and raw result is saved under sok/slr/runs/<run>/screening/<screener>/ and never
overwritten; spend is appended to results/DOC-001/spend.csv at dated list prices.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, now_iso  # noqa: E402

MODEL = "claude-opus-5-5"
# Claude API list prices for claude-opus-5-5, USD per MTok.
PRICE = {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20}  # cache hit = 5% of input
BATCH_DISCOUNT = 0.5
PER_REQUEST = 10  # overridable with --per-request
SPEND = REPO / "results" / "DOC-001" / "spend.csv"
API = "https://api.anthropic.com/v1"


def headers():
    return {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"}


def run_dir(run):
    return REPO / "sok" / "slr" / "runs" / run


def system_prompt(run, name="screen_ta.md"):
    return (run_dir(run) / "prompts" / name).read_text()


def fmt_record(r):
    ab = (r.get("abstract") or "").strip() or "[abstract not available]"
    venue = "; ".join(r.get("venues") or [r.get("venue") or ""])[:120]
    return f"Record id: {r['rid']}\nTitle: {r.get('title','').strip()}\nYear: {r.get('year')}\nVenue: {venue}\nAbstract: {ab}\n"


def build_params(sys_text, recs, max_tokens=6000, effort="low"):
    body = "Screen these records. Return only the JSON array.\n\n" + "\n---\n".join(fmt_record(r) for r in recs)
    return {"model": MODEL, "max_tokens": max_tokens, "output_config": {"effort": effort},
            "system": [{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": body}]}


# USD per MTok, list prices from https://platform.claude.com/docs/en/about-claude/pricing.
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-haiku-5-5>100k": {"input": 0.50, "output": 2.50, "cache_write": 0.625, "cache_read": 0.05},
}


def cost(usage, batch=True, model=MODEL):
    f = BATCH_DISCOUNT if batch else 1.0
    pr = PRICES.get(model, PRICE)
    if model == "claude-haiku-5-5":
        prompt = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
        if prompt > 100_000:
            pr = PRICES["claude-haiku-5-5>100k"]
    c = (usage.get("input_tokens", 0) * pr["input"] + usage.get("output_tokens", 0) * pr["output"]
         + usage.get("cache_creation_input_tokens", 0) * pr["cache_write"]
         + usage.get("cache_read_input_tokens", 0) * pr["cache_read"]) / 1e6
    return c * f


def log_spend(step, calls, usage_sum, usd, model=MODEL):
    new = not SPEND.exists()
    with open(SPEND, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "step", "model", "calls", "input_tokens", "cache_write_tokens",
                        "cache_read_tokens", "output_tokens", "cost_usd", "price_note"])
        w.writerow([now_iso(), step, model, calls, usage_sum.get("input_tokens", 0),
                    usage_sum.get("cache_creation_input_tokens", 0), usage_sum.get("cache_read_input_tokens", 0),
                    usage_sum.get("output_tokens", 0), round(usd, 4),
                    "list prices; batch 50%"])


def total_spend():
    if not SPEND.exists():
        return 0.0
    return sum(float(r["cost_usd"]) for r in csv.DictReader(open(SPEND)))


def parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def load(run, name):
    return [json.loads(l) for l in (run_dir(run) / name).open()]


def cmd_pilot(a):
    recs = load(a.run, a.input)[: a.n]
    sys_text = system_prompt(a.run, a.prompt)
    out = run_dir(a.run) / "screening" / "claude" / f"pilot_{int(time.time())}"
    out.mkdir(parents=True)
    usum, calls, decisions = {}, 0, []
    for i in range(0, len(recs), PER_REQUEST):
        chunk = recs[i:i + PER_REQUEST]
        params = build_params(sys_text, chunk, effort=a.effort)
        t0 = time.time()
        r = requests.post(f"{API}/messages", headers=headers(), json=params, timeout=600)
        d = r.json()
        (out / f"req_{i}.json").write_text(json.dumps(params))
        (out / f"res_{i}.json").write_text(json.dumps(d))
        calls += 1
        for k, v in (d.get("usage") or {}).items():
            if isinstance(v, int):
                usum[k] = usum.get(k, 0) + v
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        decisions += parse_json_array(text)
        print(f"call {calls}: {time.time()-t0:.1f}s usage={d.get('usage')}")
    usd = cost(usum, batch=False)
    log_spend(f"screen-pilot:{a.input}", calls, usum, usd)
    print(json.dumps(decisions[:5], indent=1))
    print(f"decisions: {len(decisions)} / {len(recs)}; cost ${usd:.4f}; running total ${total_spend():.2f}")
    return 0


def cmd_submit(a):
    if total_spend() >= a.cap:
        print("spend cap reached; not submitting")
        return 2
    recs = load(a.run, a.input)
    done = set()
    dpath = run_dir(a.run) / "screening" / "claude" / "decisions.jsonl"
    if dpath.exists():
        done = {json.loads(l)["rid"] for l in dpath.open()}
    recs = [r for r in recs if r["rid"] not in done]
    if a.only:
        keep = set(a.only.split(","))
        recs = [r for r in recs if r["rid"] in keep]
    sys_text = system_prompt(a.run, a.prompt)
    reqs = []
    k = a.per_request
    for i in range(0, len(recs), k):
        chunk = recs[i:i + k]
        reqs.append({"custom_id": f"{a.tag}-{i // k:05d}", "params": build_params(sys_text, chunk, effort=a.effort),
                     "_rids": [r["rid"] for r in chunk]})
    out = run_dir(a.run) / "screening" / "claude" / f"batch_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in reqs))
    payload = {"requests": [{"custom_id": r["custom_id"], "params": r["params"]} for r in reqs]}
    r = requests.post(f"{API}/messages/batches", headers=headers(), json=payload, timeout=600)
    d = r.json()
    (out / "batch.json").write_text(json.dumps(d, indent=1))
    print(f"submitted {len(reqs)} requests covering {len(recs)} records: {d.get('id')} {d.get('processing_status')} {d.get('error')}")
    return 0 if d.get("id") else 1


def cmd_collect(a):
    out = run_dir(a.run) / "screening" / "claude" / f"batch_{a.tag}"
    b = json.loads((out / "batch.json").read_text())
    while True:
        st = requests.get(f"{API}/messages/batches/{b['id']}", headers=headers(), timeout=120).json()
        print(now_iso(), st.get("processing_status"), st.get("request_counts"))
        if st.get("processing_status") == "ended":
            break
        if not a.wait:
            return 3
        time.sleep(60)
    (out / "batch_final.json").write_text(json.dumps(st, indent=1))
    res = requests.get(st["results_url"], headers=headers(), timeout=600)
    (out / "results.jsonl").write_bytes(res.content)
    reqs = {json.loads(l)["custom_id"]: json.loads(l) for l in (out / "requests.jsonl").open()}
    dpath = run_dir(a.run) / "screening" / "claude" / "decisions.jsonl"
    usum, calls, n_dec, missing = {}, 0, 0, []
    with open(dpath, "a") as fh:
        for line in res.text.splitlines():
            r = json.loads(line)
            cid = r["custom_id"]
            want = reqs[cid]["_rids"]
            if r["result"]["type"] != "succeeded":
                missing += want
                continue
            msg = r["result"]["message"]
            calls += 1
            for k, v in (msg.get("usage") or {}).items():
                if isinstance(v, int):
                    usum[k] = usum.get(k, 0) + v
            text = "".join(c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text")
            arr = parse_json_array(text)
            got = {}
            for x in arr:
                if isinstance(x, dict) and x.get("rid") in want:
                    got[x["rid"]] = x
            for rid in want:
                if rid in got and got[rid].get("decision") in ("include", "exclude", "uncertain"):
                    x = got[rid]
                    fh.write(json.dumps({"rid": rid, "screener": "claude", "model": MODEL,
                                         "decision": x["decision"], "criterion": x.get("criterion", ""),
                                         "side_list": x.get("side_list", "none"), "reason": x.get("reason", ""),
                                         "custom_id": cid, "batch_id": b["id"], "stop_reason": msg.get("stop_reason"),
                                         "at": now_iso()}) + "\n")
                    n_dec += 1
                else:
                    missing.append(rid)
    usd = cost(usum, batch=True)
    log_spend(f"screen-batch:{a.tag}", calls, usum, usd)
    (out / "missing_rids.txt").write_text("\n".join(missing))
    print(f"decisions written: {n_dec}; missing (to resubmit): {len(missing)}; cost ${usd:.2f}; total ${total_spend():.2f}")
    return 0


def cmd_merge(a):
    rd = run_dir(a.run)
    recs = {r["rid"]: r for f in a.input.split(",") for r in load(a.run, f)}
    dec = {}
    for scr in ("claude", "gpt"):
        p = rd / "screening" / scr / "decisions.jsonl"
        if p.exists():
            for l in p.open():
                x = json.loads(l)
                dec.setdefault(x["rid"], {})[scr] = x  # last decision wins (resubmits)
    rows = []
    for rid, r in recs.items():
        d = dec.get(rid, {})
        c, g = d.get("claude"), d.get("gpt")
        decisions = [x["decision"] for x in (c, g) if x]
        # Recall-favouring rule; a record the screener refused to classify is forwarded unscreened.
        forward = any(x in ("include", "uncertain") for x in decisions) if decisions else True
        rows.append({"rid": rid, "title": r.get("title", ""), "year": r.get("year"), "doi": r.get("doi", ""),
                     "arxiv_id": r.get("arxiv_id", ""), "sources": "+".join(r.get("sources", [])),
                     "claude_decision": c["decision"] if c else "refused-or-unparsed", "claude_criterion": c["criterion"] if c else "",
                     "claude_side_list": c["side_list"] if c else "", "claude_reason": c["reason"] if c else "",
                     "gpt_decision": g["decision"] if g else "BLOCKED", "gpt_criterion": g["criterion"] if g else "",
                     "gpt_reason": g["reason"] if g else "",
                     "forward_to_fulltext": forward, "screened": bool(decisions)})
    out = rd / a.out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("claude:", Counter(r["claude_decision"] for r in rows))
    print("forward:", Counter(r["forward_to_fulltext"] for r in rows))
    print("side lists:", Counter(r["claude_side_list"] for r in rows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("pilot", cmd_pilot), ("submit", cmd_submit), ("collect", cmd_collect), ("merge", cmd_merge)):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--input", default="dedup_all.jsonl")
        p.add_argument("--prompt", default="screen_ta.md")
        p.add_argument("--tag", default="main")
        p.add_argument("--n", type=int, default=30)
        p.add_argument("--effort", default="low")
        p.add_argument("--cap", type=float, default=75.0)
        p.add_argument("--only", default="")
        p.add_argument("--wait", action="store_true")
        p.add_argument("--out", default="decisions_ta.csv")
        p.add_argument("--per-request", type=int, default=PER_REQUEST)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    if a.cmd == "submit" and a.screener_blocked if hasattr(a, "screener_blocked") else False:
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
