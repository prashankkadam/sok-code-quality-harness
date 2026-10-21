"""GRPO with a LoRA adapter, trained locally with TRL (Fireworks stopped accepting new reinforcement fine-tuning jobs on
2026-10-01); the adapter is then uploaded to Fireworks and served on the base model's deployment like the SFT and DPO adapters."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..config import load, model_cfg

Reward = Callable[[str, dict], float]


def _text(completion) -> str:
    return completion[-1]["content"] if isinstance(completion, list) else completion


def _longest_prompt(hf_id: str, rows: list[dict]) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    return max(len(tok(tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True),
                       add_special_tokens=False).input_ids) for r in rows)


def train(model_key: str, rows: list[dict], reward: Reward, out: Path, cfg: dict | None = None,
          init_adapter: Path | None = None) -> Path:
    """rows: {"messages": chat prompt, ...extra fields passed to reward}; reward(completion_text, row) -> float.
    cfg overrides configs/training.yaml grpo; init_adapter: a PEFT LoRA adapter to continue from (e.g. an SFT stage).
    No prompt is cut: the vLLM context is the longest prompt plus max_tokens."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    c = load("training")["grpo"] | (cfg or {})
    hf_id = model_cfg(model_key)["hf_id"]
    extra = sorted({k for r in rows for k in r} - {"messages"})
    ds = Dataset.from_list([{"prompt": r["messages"], "row": json.dumps({k: r.get(k) for k in extra})} for r in rows])

    def reward_fn(completions, row, **_):
        with ThreadPoolExecutor(c.get("reward_workers", 16)) as ex:
            return list(ex.map(lambda x: float(reward(_text(x[0]), json.loads(x[1]))), zip(completions, row)))

    # older TRL truncates prompts to max_prompt_length (default 512); newer TRL has no such field and never truncates
    no_cut = {"max_prompt_length": None} if "max_prompt_length" in GRPOConfig.__dataclass_fields__ else {}
    args = GRPOConfig(
        output_dir=str(out), learning_rate=c["learning_rate"], num_train_epochs=c["epochs"], beta=c["kl_beta"],
        num_generations=c["group_size"], per_device_train_batch_size=c["group_size"],
        gradient_accumulation_steps=c.get("grad_accum", 4), temperature=c["temperature"],
        max_completion_length=c["max_tokens"], **no_cut,
        vllm_max_model_length=_longest_prompt(hf_id, rows) + c["max_tokens"], bf16=True, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", logging_steps=10, save_strategy="epoch", report_to=[], seed=c.get("seed", 0))
    model, lora = hf_id, LoraConfig(r=c["lora_rank"], lora_alpha=2 * c["lora_rank"], lora_dropout=0.0,
                                    target_modules="all-linear", task_type="CAUSAL_LM")
    if init_adapter:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16)
        base.enable_input_require_grads()
        model, lora = PeftModel.from_pretrained(base, str(init_adapter), is_trainable=True), None
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=args, train_dataset=ds, peft_config=lora)
    t0 = time.time()
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    _log(model_key, out, time.time() - t0)
    return out / "adapter"


def _log(model_key: str, out: Path, seconds: float) -> None:
    """Records the local job (GPU type and count, GPU-hours, cost at local_gpu_usd_per_hour when set) for the ledger."""
    import torch
    from .fireworks import write_job
    n = torch.cuda.device_count()
    gpu_hours = n * seconds / 3600
    rate = load("training").get("local_gpu_usd_per_hour")
    write_job(model_key, {"job": str(out), "kind": "grpo", "provider": "local (TRL)", "output": str(out / "adapter"),
                          "gpu_type": torch.cuda.get_device_name(0) if n else None, "gpus": n, "gpu_hours": gpu_hours,
                          "wall_hours": seconds / 3600, "cost": gpu_hours * rate if rate is not None else None})
