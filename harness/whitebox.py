"""Local Hugging Face model with access to logits and activations, for decoding-time techniques."""
from __future__ import annotations

import contextlib
import time
from functools import lru_cache

from .llm import Backend, Completion


class LocalLM:
    def __init__(self, hf_id: str, adapter: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=getattr(torch, dtype), device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
        self.eos_ids = sorted({i for i in [self.tok.eos_token_id, *gen_eos] if isinstance(i, int)})

    def encode_chat(self, messages, add_generation_prompt: bool = True):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, return_tensors="pt")
        return ids.to(self.device)

    def encode(self, text: str):
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def layers(self):
        m = getattr(self.model, "base_model", self.model)
        for path in ("model.model.layers", "model.layers", "model.language_model.layers", "transformer.h"):
            obj = m
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise AttributeError("decoder layers not found")

    def forward(self, ids, past=None, output_hidden_states: bool = False):
        with self.torch.no_grad():
            return self.model(input_ids=ids, past_key_values=past, use_cache=True, output_hidden_states=output_hidden_states)

    def hidden(self, text_or_ids, layer: int):
        ids = self.encode(text_or_ids) if isinstance(text_or_ids, str) else text_or_ids
        return self.forward(ids, output_hidden_states=True).hidden_states[layer][0]

    @contextlib.contextmanager
    def steer(self, layer: int, vector, alpha: float = 1.0, positions: str = "all"):
        """Adds alpha*vector to the residual stream after `layer`."""
        v = vector.to(self.device, dtype=next(self.model.parameters()).dtype)

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if positions == "last":
                h[:, -1, :] += alpha * v
            else:
                h += alpha * v
            return out

        handle = self.layers()[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextlib.contextmanager
    def steer_heads(self, layer: int, head_vectors: dict[int, object], alpha: float = 1.0):
        """Adds per-head vectors to the attention output (before o_proj) of `layer`."""
        attn = self.layers()[layer].self_attn
        hd = attn.head_dim

        def pre(_m, args):
            x = args[0]
            for h, v in head_vectors.items():
                x[..., h * hd:(h + 1) * hd] += alpha * v.to(x.device, x.dtype)
            return (x,) + tuple(args[1:])

        handle = attn.o_proj.register_forward_pre_hook(pre)
        try:
            yield
        finally:
            handle.remove()

    def generate(self, messages, n: int = 1, temperature: float = 0.2, max_tokens: int = 2048,
                 logits_processor=None, prefix_ids=None, seed: int | None = None) -> list[str]:
        from transformers import LogitsProcessorList
        if seed is not None:
            self.torch.manual_seed(seed)
        ids = self.encode_chat(messages) if prefix_ids is None else prefix_ids
        with self.torch.no_grad():
            out = self.model.generate(ids, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=1.0,
                                      max_new_tokens=max_tokens, num_return_sequences=n, eos_token_id=self.eos_ids,
                                      pad_token_id=self.tok.pad_token_id or self.eos_ids[0],
                                      logits_processor=LogitsProcessorList(logits_processor or []))
        return [self.decode(o[ids.shape[1]:]) for o in out]


@lru_cache(maxsize=2)
def local_lm(hf_id: str, adapter: str | None = None) -> LocalLM:
    return LocalLM(hf_id, adapter)


class LocalBackend(Backend):
    supports_n = True

    def __init__(self, hf_id: str, adapter: str | None = None):
        self.lm = local_lm(hf_id, adapter)

    def complete(self, messages, *, n, temperature, seed, max_tokens, stop, effort, logprobs):
        t0 = time.time()
        texts = self.lm.generate(messages, n=n, temperature=temperature or 0.0, max_tokens=max_tokens, seed=seed)
        if stop:
            texts = [min((t.split(s)[0] for s in stop), key=len) for t in texts]
        n_in = int(self.lm.encode_chat(messages).shape[1])
        return [Completion(t, n_in, len(self.lm.tok(t).input_ids), None, time.time() - t0, "stop") for t in texts]
