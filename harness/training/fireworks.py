"""LoRA fine-tuning on Fireworks: SFT (T8) and DPO (T9); upload and serving of all adapters, including the locally trained GRPO ones (T10)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from ..config import load

API = "https://api.fireworks.ai/v1"


def _c() -> httpx.Client:
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}, timeout=300)


def _acct() -> str:
    return os.environ["FIREWORKS_ACCOUNT_ID"]


def _mask(row: dict) -> dict:
    """SFT rows (ending in an assistant turn) get explicit per-message loss weights: assistant 1, everything else 0, unless
    the row already sets a message's weight."""
    m = row.get("messages")
    if m and m[-1]["role"] == "assistant":
        row["messages"] = [{"weight": int(x["role"] == "assistant")} | x for x in m]
    return row


def upload_dataset(path: Path, dataset_id: str) -> str:
    """Uploads a JSONL dataset; Fireworks keeps the uploaded name for the job's dataset field."""
    c, a = _c(), _acct()
    rows = [json.dumps(_mask(json.loads(l))) for l in open(path) if l.strip()]
    c.post(f"/accounts/{a}/datasets", json={"datasetId": dataset_id, "dataset": {"userUploaded": {}, "exampleCount": str(len(rows))}}
           ).raise_for_status()
    c.post(f"/accounts/{a}/datasets/{dataset_id}:upload", files={"file": (path.name, ("\n".join(rows) + "\n").encode())}
           ).raise_for_status()
    return f"accounts/{a}/datasets/{dataset_id}"


def _wait(path: str, done=("JOB_STATE_COMPLETED",), fail=("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
                                                          "JOB_STATE_EARLY_STOPPED", "JOB_STATE_DELETED", "JOB_STATE_ARCHIVED")) -> dict:
    c = _c()
    while True:
        j = c.get(f"/{path}").raise_for_status().json()
        if j.get("state") in done:
            return j
        if j.get("state") in fail:
            raise RuntimeError(j)
        time.sleep(60)


def _cfg(model: str, kind: str) -> dict:
    return load("training")[kind] | {"baseModel": load("training")["base_models"][model]}


def _merge(body: dict, extra: dict | None) -> dict:
    for k, v in (extra or {}).items():
        body[k] = _merge(body[k], v) if isinstance(v, dict) and isinstance(body.get(k), dict) else v
    return body


def context_for(path: Path, model: str, floor: int) -> int:
    """Smallest context (multiple of 4096, at least floor) that holds the longest example, so nothing is cut."""
    from transformers import AutoTokenizer
    from ..config import model_cfg
    tok = AutoTokenizer.from_pretrained(model_cfg(model)["hf_id"])
    longest = 0
    for l in open(path):
        r = json.loads(l)
        convs = [r["messages"]] if r.get("messages") else [r["input"]["messages"] + r[k] for k in ("preferred_output", "non_preferred_output")]
        longest = max(longest, *(len(tok(tok.apply_chat_template(m, tokenize=False), add_special_tokens=False).input_ids) for m in convs))
    return max(floor, -(-longest // 4096) * 4096)


def sft(model: str, dataset: str, output_model: str, src: Path | None = None, extra: dict | None = None) -> str:
    """extra: further job fields (e.g. learningRateWarmupSteps, warmStartFrom), merged into the request body."""
    cfg, a = _cfg(model, "sft"), _acct()
    body = _merge({"dataset": dataset, "baseModel": cfg["baseModel"], "outputModel": f"accounts/{a}/models/{output_model}",
                   "loraRank": cfg["lora_rank"], "epochs": cfg["epochs"], "learningRate": cfg["learning_rate"],
                   "maxContextLength": cfg["max_context"], "batchSizeSamples": cfg["batch_samples"]}, extra)
    if body.get("warmStartFrom"):
        body.pop("baseModel")
    if src:
        body["maxContextLength"] = context_for(src, model, body["maxContextLength"])
    j = _c().post(f"/accounts/{a}/supervisedFineTuningJobs", json=body).raise_for_status().json()
    return _wait(j["name"])["outputModel"]


def dpo(model: str, dataset: str, output_model: str, src: Path | None = None, extra: dict | None = None) -> str:
    """extra: further job fields, merged into the request body (training fields go under "trainingConfig")."""
    cfg, a = _cfg(model, "dpo"), _acct()
    body = _merge({"dataset": dataset, "trainingConfig": {
        "baseModel": cfg["baseModel"], "outputModel": f"accounts/{a}/models/{output_model}", "loraRank": cfg["lora_rank"],
        "epochs": cfg["epochs"], "learningRate": cfg["learning_rate"], "maxContextLength": cfg["max_context"]},
        "lossConfig": {"method": "DPO", "dpo": {"beta": cfg["beta"]}}}, extra)
    tc = body["trainingConfig"]
    if tc.get("warmStartFrom"):
        tc.pop("baseModel")
    if src:
        tc["maxContextLength"] = context_for(src, model, tc["maxContextLength"])
    j = _c().post(f"/accounts/{a}/dpoJobs", json=body).raise_for_status().json()
    return _wait(j["name"])["trainingConfig"]["outputModel"]


def upload_lora(adapter_dir: Path, model_id: str, base_model_key: str) -> str:
    """Uploads a locally trained PEFT LoRA adapter as a Fireworks model (used for the GRPO adapters)."""
    c, a = _c(), _acct()
    files = {p.name: p for p in adapter_dir.iterdir() if p.is_file()}
    c.post(f"/accounts/{a}/models", json={"modelId": model_id, "model": {
        "kind": "HF_PEFT_ADDON", "baseModelDetails": {"checkpointFormat": "HUGGINGFACE"},
        "peftDetails": {"baseModel": load("training")["base_models"][base_model_key]}}}).raise_for_status()
    urls = c.post(f"/accounts/{a}/models/{model_id}:getUploadEndpoint",
                  json={"filenameToSize": {n: p.stat().st_size for n, p in files.items()}}).raise_for_status().json()
    for name, url in urls["filenameToSignedUrls"].items():
        with open(files[name], "rb") as f:
            httpx.put(url, content=f.read(), headers={"Content-Type": "application/octet-stream"}, timeout=600).raise_for_status()
    c.get(f"/accounts/{a}/models/{model_id}:validateUpload").raise_for_status()
    return f"accounts/{a}/models/{model_id}"


def download_lora(model_name: str, dest: Path) -> Path:
    """Downloads a Fireworks LoRA adapter (e.g. an SFT stage) so that local GRPO can start from it; returns its PEFT dir."""
    if not any(dest.rglob("adapter_config.json")):
        urls = _c().get(f"/{model_name}:getDownloadEndpoint").raise_for_status().json()["filenameToSignedUrls"]
        for name, url in urls.items():
            p = dest / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r, open(p, "wb") as f:
                r.raise_for_status()
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
    return next(dest.rglob("adapter_config.json")).parent


def deploy(model_name: str, base_deployment: str) -> str:
    """Loads the LoRA adapter onto the base model's dedicated deployment (multi-LoRA) and returns the served model id."""
    a = _acct()
    _c().post(f"/accounts/{a}/deployedModels", json={"model": model_name, "deployment": base_deployment,
                                                      "default": False, "serverless": False}).raise_for_status()
    return f"{model_name}#{base_deployment}"
