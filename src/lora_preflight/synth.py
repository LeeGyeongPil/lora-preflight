"""Practice material: synthetic base models and LoRA adapters, written as bytes.

These files contain no real weights (every tensor is zero-filled), but tensor
names, shapes, dtypes and adapter_config.json fields follow what PEFT 0.21.2
wrote when the variants below were saved from a tiny random Llama (see the
README). That is enough for lora-preflight, which reads names and shapes only.
They exist for `lora-preflight demo` and for the tests; they are not real
training results.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

VARIANTS = (
    "plain_r8",
    "r128_all",
    "dora",
    "mts_embed_lmhead",
    "bias_all",
    "rank_pattern",
    "lora_on_embed",
)

DEFAULT_BASE_NAME = "example/base-model"
_PREFIX = "base_model.model."

# Tiny Llama: hidden 64, MLP 128, 4 heads of 16 dims, 2 KV heads, 2 layers.
_HIDDEN, _INTER, _KV_OUT, _LAYERS = 64, 128, 32, 2
# module path inside a layer -> (out_features, in_features)
_LINEAR = {
    "self_attn.q_proj": (_HIDDEN, _HIDDEN),
    "self_attn.k_proj": (_KV_OUT, _HIDDEN),
    "self_attn.v_proj": (_KV_OUT, _HIDDEN),
    "self_attn.o_proj": (_HIDDEN, _HIDDEN),
    "mlp.gate_proj": (_INTER, _HIDDEN),
    "mlp.up_proj": (_INTER, _HIDDEN),
    "mlp.down_proj": (_HIDDEN, _INTER),
}

Tensors = dict[str, tuple[str, tuple[int, ...]]]


def write_safetensors(path: str | Path, tensors: Tensors) -> None:
    """Write a valid safetensors file whose tensor data is all zero bytes."""
    sizes = {"F32": 4, "F16": 2, "BF16": 2}
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        count = 1
        for dim in shape:
            count *= dim
        end = offset + count * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)  # the reference writer pads the header to 8 bytes
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\x00" * offset)


def make_base(directory: str | Path, vocab: int = 128) -> Path:
    """Write a tiny Llama-style base model folder (config.json + model.safetensors)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tensors: Tensors = {
        "model.embed_tokens.weight": ("F32", (vocab, _HIDDEN)),
        "lm_head.weight": ("F32", (vocab, _HIDDEN)),
        "model.norm.weight": ("F32", (_HIDDEN,)),
    }
    for layer in range(_LAYERS):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "input_layernorm.weight"] = ("F32", (_HIDDEN,))
        tensors[prefix + "post_attention_layernorm.weight"] = ("F32", (_HIDDEN,))
        for module, (out_f, in_f) in _LINEAR.items():
            tensors[f"{prefix}{module}.weight"] = ("F32", (out_f, in_f))
    write_safetensors(root / "model.safetensors", tensors)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTER,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": _LAYERS,
        "vocab_size": vocab,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return root


def _lora_pair(tensors: Tensors, module: str, out_f: int, in_f: int, rank: int, dora: bool = False) -> None:
    key = _PREFIX + module
    tensors[f"{key}.lora_A.weight"] = ("F32", (rank, in_f))
    tensors[f"{key}.lora_B.weight"] = ("F32", (out_f, rank))
    if dora:
        tensors[f"{key}.lora_magnitude_vector"] = ("F32", (out_f,))


def _config(base_name: str, **overrides: object) -> dict[str, object]:
    """adapter_config.json with the fields PEFT 0.21.2 saves for a LoRA adapter."""
    config: dict[str, object] = {
        "peft_type": "LORA",
        "peft_version": "0.21.2",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_name,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "modules_to_save": None,
        "bias": "none",
        "lora_bias": False,
        "use_dora": False,
        "use_rslora": False,
        "init_lora_weights": True,
        "rank_pattern": {},
        "alpha_pattern": {},
        "layer_replication": None,
        "inference_mode": True,
    }
    config.update(overrides)
    return config


def make_adapter(
    directory: str | Path,
    variant: str,
    base_name: str | Path = DEFAULT_BASE_NAME,
    vocab: int = 128,
) -> Path:
    """Write one practice adapter folder.

    ``vocab`` only matters for ``mts_embed_lmhead`` and ``lora_on_embed``: set it
    above the base vocabulary to imitate an adapter saved after adding tokens.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}'; choose from {', '.join(VARIANTS)}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = str(base_name)
    tensors: Tensors = {}
    qv = ("self_attn.q_proj", "self_attn.v_proj")
    if variant == "r128_all":
        for layer in range(_LAYERS):
            for module, (out_f, in_f) in _LINEAR.items():
                _lora_pair(tensors, f"model.layers.{layer}.{module}", out_f, in_f, 128)
        targets = [f"model.layers.{i}.{m}" for i in range(_LAYERS) for m in _LINEAR]
        config = _config(name, r=128, lora_alpha=256, target_modules=targets)
    elif variant == "lora_on_embed":
        tensors[f"{_PREFIX}lm_head.base_layer.weight"] = ("F32", (vocab, _HIDDEN))
        tensors[f"{_PREFIX}lm_head.lora_A.weight"] = ("F32", (8, _HIDDEN))
        tensors[f"{_PREFIX}lm_head.lora_B.weight"] = ("F32", (vocab, 8))
        tensors[f"{_PREFIX}model.embed_tokens.base_layer.weight"] = ("F32", (vocab, _HIDDEN))
        tensors[f"{_PREFIX}model.embed_tokens.lora_embedding_A"] = ("F32", (8, vocab))
        tensors[f"{_PREFIX}model.embed_tokens.lora_embedding_B"] = ("F32", (_HIDDEN, 8))
        for layer in range(_LAYERS):
            out_f, in_f = _LINEAR["self_attn.q_proj"]
            _lora_pair(tensors, f"model.layers.{layer}.self_attn.q_proj", out_f, in_f, 8)
        config = _config(name, target_modules=["lm_head", "embed_tokens", "q_proj"])
    else:
        ranks = {"self_attn.q_proj": 8, "self_attn.v_proj": 64 if variant == "rank_pattern" else 8}
        for layer in range(_LAYERS):
            for module in qv:
                out_f, in_f = _LINEAR[module]
                _lora_pair(tensors, f"model.layers.{layer}.{module}", out_f, in_f, ranks[module],
                           dora=variant == "dora")
        overrides: dict[str, object] = {}
        if variant == "dora":
            overrides["use_dora"] = True
        elif variant == "bias_all":
            overrides["bias"] = "all"  # config only: a Llama has no bias tensors to save
        elif variant == "rank_pattern":
            overrides["rank_pattern"] = {"v_proj": 64}
        elif variant == "mts_embed_lmhead":
            tensors[f"{_PREFIX}model.embed_tokens.weight"] = ("F32", (vocab, _HIDDEN))
            tensors[f"{_PREFIX}lm_head.weight"] = ("F32", (vocab, _HIDDEN))
            overrides["modules_to_save"] = ["embed_tokens", "lm_head"]
        config = _config(name, **overrides)
    write_safetensors(root / "adapter_model.safetensors", tensors)
    (root / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return root


def make_all(root: str | Path) -> tuple[Path, dict[str, Path]]:
    """Write ``base/`` plus one folder per variant under ``root``."""
    top = Path(root)
    base = make_base(top / "base")
    adapters = {v: make_adapter(top / v, v, base_name=base) for v in VARIANTS}
    return base, adapters
