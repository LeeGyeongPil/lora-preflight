"""The rule table and the per-runtime evaluation.

Every rule has an id, a target runtime, and a source (URL, version or commit,
and the date a person last checked it). When a runtime changes its behaviour,
this is the only file that needs editing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import fixes
from .adapter import AdapterSummary
from .base import BaseSummary
from .report import Finding

VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
VLLM_DEFAULT_MAX_LORA_RANK = 16
# init_lora_weights strings vLLM accepts; other strings (pissa, olora, corda, loftq) change base weights.
_VLLM_OK_INITS = ("gaussian", "eva", "orthogonal", "mica", "lora_ga")
# Module names (last component) vLLM-served Llama / Mistral / Qwen / Gemma style models expose to LoRA.
_VLLM_KNOWN_MODULES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
})
CHECKED = "2026-11-10"

_VLLM_SRC = "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/lora/peft_helper.py"
_VLLM_MODEL_SRC = "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/lora/lora_model.py"
_VLLM_UTILS_SRC = "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/lora/utils.py"
_LLAMA_SRC = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_lora_to_gguf.py"
_OLLAMA_SRC = "https://docs.ollama.com/modelfile"
_VLLM_REF = "vLLM main 83f5de466a (source read)"
_LLAMA_REF = "llama.cpp master 45c8ea8374 (source read + conversions run)"
_OLLAMA_REF = "Ollama 0.40.15 (run locally)"


@dataclass(frozen=True)
class Rule:
    """One row of the rule table."""

    id: str
    target: str
    title: str
    source_url: str
    source_ref: str
    checked: str = CHECKED

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id, "target": self.target, "title": self.title,
            "source_url": self.source_url, "source_ref": self.source_ref, "checked": self.checked,
        }


RULES: tuple[Rule, ...] = (
    Rule("common.weights-format", "common", "adapter_model.bin cannot be read without torch; only config is checked",
         "https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint", "PEFT 0.21.2 save format (observed by saving tiny test adapters)"),
    Rule("common.pairing", "common", "every LoRA module needs matching A and B tensors of the same rank",
         "https://huggingface.co/docs/peft/main/en/conceptual_guides/lora", "PEFT 0.21.2 save format (observed by saving tiny test adapters)"),
    Rule("common.unmatched-target", "common", "a target_modules entry that matches no tensor was silently skipped in training",
         _VLLM_UTILS_SRC, _VLLM_REF),
    Rule("common.base-module-missing", "common", "adapter modules that do not exist in the base model",
         _LLAMA_SRC, _LLAMA_REF),
    Rule("vllm.rank", "vllm", "max rank (r and rank_pattern) must be <= --max-lora-rank (default 16; allowed 1,8,16,32,64,128,256,320,512)",
         _VLLM_SRC, _VLLM_REF),
    Rule("vllm.dora", "vllm", "DoRA is not supported", _VLLM_SRC, _VLLM_REF),
    Rule("vllm.bias", "vllm", "bias other than 'none' is not supported", _VLLM_SRC, _VLLM_REF),
    Rule("vllm.lora-bias", "vllm", "lora_bias is not supported", _VLLM_SRC, _VLLM_REF),
    Rule("vllm.modules-to-save", "vllm", "modules_to_save / whole-module tensors only allowed for classifier and score",
         _VLLM_SRC, _VLLM_REF),
    Rule("vllm.init-weights", "vllm", "init_lora_weights such as PiSSA, OLoRA, CorDA, LoftQ are not supported",
         _VLLM_SRC, _VLLM_REF),
    Rule("vllm.variants", "vllm", "aLoRA, layer_replication, BD-LoRA and QALoRA are not supported", _VLLM_SRC, _VLLM_REF),
    Rule("vllm.embedding-vocab", "vllm", "embedding LoRA vocabulary size must equal the base vocabulary size",
         _VLLM_MODEL_SRC, _VLLM_REF),
    Rule("vllm.shape", "vllm", "A/B shapes and whole-module tensors must fit the base model",
         _VLLM_MODEL_SRC, _VLLM_REF),
    Rule("vllm.unknown-module", "vllm", "module names outside the Llama/Mistral/Qwen/Gemma table cannot be confirmed",
         _VLLM_UTILS_SRC, _VLLM_REF),
    Rule("llamacpp.dora", "llamacpp", "convert_lora_to_gguf.py stops on lora_magnitude_vector (DoRA)", _LLAMA_SRC, _LLAMA_REF),
    Rule("llamacpp.unexpected-tensor", "llamacpp", "tensors that are not lora_A / lora_B (whole modules, bias) stop the converter",
         _LLAMA_SRC, _LLAMA_REF),
    Rule("llamacpp.shape", "llamacpp", "adapter shapes must match the base model ('maybe wrong base model?')",
         "https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.cpp", _LLAMA_REF),
    Rule("llamacpp.howto", "llamacpp", "how to convert and serve an adapter that passed", _LLAMA_SRC, _LLAMA_REF),
    Rule("ollama.adapter-unsupported", "ollama", "ADAPTER in a Modelfile fails with 'LoRA adapters are no longer supported'",
         _OLLAMA_SRC, _OLLAMA_REF),
)
_RULE_BY_ID = {r.id: r for r in RULES}
KNOWN_TARGETS = ("vllm", "llamacpp", "ollama")


def _finding(rule_id: str, status: str, message: str, fix: str = "") -> Finding:
    rule = _RULE_BY_ID[rule_id]
    return Finding(rule_id, rule.target, status, message, fix, f"{rule.source_url} ({rule.source_ref}, checked {rule.checked})")


# --------------------------------------------------------------------------- base comparison
@dataclass
class ShapeProblem:
    kind: str  # "vocab", "linear" or "full"
    text: str


def shape_problems(summary: AdapterSummary, base: BaseSummary) -> tuple[list[ShapeProblem], list[str]]:
    """Compare adapter shapes with the base. Returns (problems, modules missing in base)."""
    problems: list[ShapeProblem] = []
    missing: list[str] = []
    for name, info in summary.modules.items():
        base_shape = base.shape_of(name)
        if base_shape is None or len(base_shape) != 2:
            missing.append(name)
            continue
        if info.a_shape is None or info.b_shape is None or len(info.a_shape) != 2 or len(info.b_shape) != 2:
            continue
        if info.kind == "embedding":
            if info.a_shape[1] != base_shape[0]:
                problems.append(ShapeProblem(
                    "vocab",
                    f"embedding LoRA size({info.a_shape[1]}) must be consistent with the base model's "
                    f"vocabulary size({base_shape[0]}) ({name})",
                ))
            if info.b_shape[0] != base_shape[1]:
                problems.append(ShapeProblem("linear", f"{name}: B has {info.b_shape[0]} rows, base embedding dim is {base_shape[1]}"))
        else:
            if info.a_shape[1] != base_shape[1]:
                problems.append(ShapeProblem("linear", f"{name}: A expects {info.a_shape[1]} inputs, base has {base_shape[1]}"))
            if info.b_shape[0] != base_shape[0]:
                problems.append(ShapeProblem("linear", f"{name}: B gives {info.b_shape[0]} outputs, base has {base_shape[0]}"))
    for name in summary.full_tensors:
        base_shape = base.shapes.get(name)
        shape = summary.tensors[name][1]
        if base_shape is not None and tuple(base_shape) != tuple(shape):
            problems.append(ShapeProblem("full", f"{name}: adapter shape {list(shape)} != base shape {list(base_shape)}"))
    return problems, missing


# --------------------------------------------------------------------------- common
def _common(summary: AdapterSummary, base: BaseSummary | None) -> list[Finding]:
    out: list[Finding] = []
    if summary.weights_format == "bin":
        out.append(_finding(
            "common.weights-format", "unknown",
            "adapter_model.bin is a pickle file; tensors cannot be read safely without torch, so only adapter_config.json was checked.",
            "Re-save as safetensors: model.save_pretrained(dir, safe_serialization=True)",
        ))
        return out
    bad = [f"{m} (A and B ranks differ)" for m in summary.rank_mismatch]
    bad += [f"{m} (only one of lora_A / lora_B present)" for m in summary.unpaired]
    if bad:
        out.append(_finding("common.pairing", "fail", "broken LoRA pairs: " + "; ".join(bad[:5]),
                            "The file is incomplete or corrupt; re-run save_pretrained or copy the adapter again."))
    else:
        out.append(_finding("common.pairing", "pass", f"{len(summary.modules)} module(s) have matching A/B tensors"))
    if summary.unmatched_targets:
        out.append(_finding(
            "common.unmatched-target", "warn",
            "target_modules entries with no tensor in the file: " + ", ".join(summary.unmatched_targets)
            + ". PEFT skips a target the model does not have without an error, so those layers were never trained.",
            "Check the module names with `print(model)` before training and fix target_modules.",
        ))
    else:
        out.append(_finding("common.unmatched-target", "pass", "every target_modules entry has tensors"))
    if base is not None:
        _, missing = shape_problems(summary, base)
        if missing:
            out.append(_finding(
                "common.base-module-missing", "warn",
                f"{len(missing)} adapter module(s) not found in the base, e.g. {missing[0]}. "
                "Is this the base model the adapter was trained on?",
                "Pass the matching base with --base <folder|hf:repo>.",
            ))
        else:
            out.append(_finding("common.base-module-missing", "pass", "all adapter modules exist in the base"))
    return out


# --------------------------------------------------------------------------- vLLM
def _next_allowed(rank: int) -> int | None:
    return next((v for v in VLLM_MAX_LORA_RANKS if v >= rank), None)


def _vllm(summary: AdapterSummary, base: BaseSummary | None, max_lora_rank: int) -> list[Finding]:
    cfg = summary.config
    out: list[Finding] = []

    # rank: vLLM checks r and rank_pattern values; we also look at the tensors.
    candidates = [v for v in (summary.config_max_rank, summary.max_rank) if v is not None]
    rank = max(candidates) if candidates else None
    if rank is None:
        out.append(_finding("vllm.rank", "unknown", "no rank found in the config or tensors"))
    elif rank > max_lora_rank:
        need = _next_allowed(rank)
        if need is None:
            out.append(_finding(
                "vllm.rank", "fail",
                f"LoRA rank {rank} is greater than max_lora_rank {max_lora_rank}, and {rank} is above the largest allowed value (512; allowed: "
                + ", ".join(map(str, VLLM_MAX_LORA_RANKS)) + ").",
                "No --max-lora-rank value fits. Retrain with r <= 512, or merge the adapter into the base model and serve the merged model.",
            ))
        else:
            out.append(_finding(
                "vllm.rank", "fail",
                f"LoRA rank {rank} is greater than max_lora_rank {max_lora_rank}.",
                fixes.vllm_serve(summary, need),
            ))
    else:
        out.append(_finding("vllm.rank", "pass", f"max rank {rank} <= max_lora_rank {max_lora_rank}"))

    # DoRA
    if cfg.get("use_dora") or summary.magnitude_tensors:
        out.append(_finding("vllm.dora", "fail", "vLLM does not yet support DoRA.",
                            "Merge the adapter into the base model (merge_and_unload) and serve the merged model."))
    else:
        out.append(_finding("vllm.dora", "pass", "no DoRA"))

    # bias
    bias = cfg.get("bias", "none")
    if bias != "none":
        if summary.bias_tensors:
            fix = "Retrain with bias='none', or merge the adapter into the base model and serve the merged model."
        else:
            fix = ("The file has no bias tensors, so only the config is wrong: set \"bias\": \"none\" in "
                   "adapter_config.json (a copy is safest) and the adapter loads.")
        out.append(_finding("vllm.bias", "fail", f"Adapter bias is not supported (bias={bias!r}).", fix))
    else:
        out.append(_finding("vllm.bias", "pass", "bias is 'none'"))
    if cfg.get("lora_bias"):
        out.append(_finding("vllm.lora-bias", "fail", "vLLM does not support LoRA bias (lora_bias).",
                            "Retrain with lora_bias=False, or merge the adapter and serve the merged model."))
    else:
        out.append(_finding("vllm.lora-bias", "pass", "lora_bias is off"))

    # modules_to_save / whole-module tensors
    allowed = ("classifier", "score")
    unsupported = [m for m in summary.modules_to_save if m not in allowed]
    for name in summary.full_tensors:
        module = name.rsplit(".", 1)[0]
        if module.rsplit(".", 1)[-1] not in allowed and module not in unsupported:
            unsupported.append(module)
    if unsupported:
        out.append(_finding(
            "vllm.modules-to-save", "fail",
            f"vLLM only supports modules_to_save being either None or [\"classifier\", \"score\"]. Unsupported: {unsupported}",
            "Drop the extra modules (you lose the changed embedding/head weights, e.g. new tokens), "
            "or merge the adapter into the base model and serve the merged model.",
        ))
    else:
        out.append(_finding("vllm.modules-to-save", "pass", "no unsupported modules_to_save"))

    # init_lora_weights
    init = cfg.get("init_lora_weights")
    if isinstance(init, str) and init.lower() not in _VLLM_OK_INITS:
        out.append(_finding(
            "vllm.init-weights", "fail",
            f"vLLM does not support init_lora_weights={init!r}. Inits such as PiSSA, OLoRA, CorDA and LoftQ modify the base model weights.",
            "Convert to a regular LoRA adapter with PEFT's path_initial_model_for_weight_conversion, "
            "or set init_lora_weights to true if the served base already contains the modified weights.",
        ))
    else:
        out.append(_finding("vllm.init-weights", "pass", "init_lora_weights is supported"))

    variants = []
    if cfg.get("alora_invocation_tokens"):
        variants.append("Activated LoRA (aLoRA)")
    if cfg.get("layer_replication"):
        variants.append("layer_replication")
    if cfg.get("use_bdlora"):
        variants.append("block-diagonal LoRA (BD-LoRA)")
    if cfg.get("use_qalora"):
        variants.append("QALoRA")
    if variants:
        out.append(_finding("vllm.variants", "fail", "vLLM does not support: " + ", ".join(variants) + ".",
                            "Merge the adapter into the base model and serve the merged model."))
    else:
        out.append(_finding("vllm.variants", "pass", "no unsupported adapter variant"))

    # base-dependent checks
    vocab_fail = None
    problems: list[ShapeProblem] = []
    if base is not None:
        problems, _ = shape_problems(summary, base)
    embed_modules = [m for m in summary.modules.values() if m.kind == "embedding"]
    if base is None and embed_modules:
        out.append(_finding("vllm.embedding-vocab", "unknown",
                            "the adapter has embedding LoRA tensors, but no base was given to compare vocabulary sizes.",
                            "Re-run with --base <folder|hf:repo>."))
    else:
        vocab_fail = [p for p in problems if p.kind == "vocab"]
        full_vocab = [p for p in problems if p.kind == "full"]
        both = vocab_fail + full_vocab
        if both:
            out.append(_finding(
                "vllm.embedding-vocab", "fail", "; ".join(p.text for p in both[:3]),
                "The adapter was saved after the vocabulary grew (new tokens). Serve a base resized to the same vocabulary "
                "(save the resized base after training), or merge the adapter into that resized base.",
            ))
        else:
            out.append(_finding("vllm.embedding-vocab", "pass", "no vocabulary size conflict"))
    if base is not None:
        linear = [p for p in problems if p.kind == "linear"]
        if linear:
            out.append(_finding("vllm.shape", "fail", "; ".join(p.text for p in linear[:3]),
                                "This is probably not the base the adapter was trained on; pass the right --base."))
        else:
            out.append(_finding("vllm.shape", "pass", "A/B shapes fit the base"))

    # module names vLLM may not know
    unknown_modules = sorted({n.rsplit(".", 1)[-1] for n in summary.modules} - _VLLM_KNOWN_MODULES)
    if unknown_modules:
        out.append(_finding(
            "vllm.unknown-module", "warn",
            "module names outside the Llama/Mistral/Qwen/Gemma table: " + ", ".join(unknown_modules)
            + ". vLLM accepts only the modules the served model class lists; this could not be confirmed.",
            "Try loading it once with vLLM; if it fails with 'Please verify that the loaded LoRA module is correct', "
            "retrain with target_modules limited to modules the model supports.",
        ))
    else:
        out.append(_finding("vllm.unknown-module", "pass", "all module names are in the known table"))
    return out


# --------------------------------------------------------------------------- llama.cpp
def _is_norm(name: str) -> bool:
    return "_layernorm" in name or ".norm" in name


def _llamacpp(summary: AdapterSummary, base: BaseSummary | None) -> list[Finding]:
    out: list[Finding] = []
    if summary.weights_format == "bin":
        out.append(_finding("llamacpp.unexpected-tensor", "unknown",
                            "tensors were not read (adapter_model.bin), so the converter outcome cannot be predicted.",
                            "Re-save as safetensors: model.save_pretrained(dir, safe_serialization=True)"))
        return out
    if summary.magnitude_tensors:
        out.append(_finding(
            "llamacpp.dora", "fail",
            f"Unexpected name '{summary.magnitude_tensors[0]}': Not a lora_A or lora_B tensor (DoRA magnitude vectors).",
            "Merge the adapter into the base model (merge_and_unload), then convert the merged model to GGUF.",
        ))
    else:
        out.append(_finding("llamacpp.dora", "pass", "no DoRA magnitude tensors"))
    unexpected = [n for n in summary.full_tensors + summary.bias_tensors if not _is_norm(n)]
    if unexpected:
        shown = ", ".join(f"'{n}'" for n in unexpected[:3])
        out.append(_finding(
            "llamacpp.unexpected-tensor", "fail",
            f"Unexpected name {shown}: Not a lora_A or lora_B tensor.",
            "Remove whole-module tensors (modules_to_save such as embed_tokens / lm_head, bias) before converting, "
            "or merge the adapter into the base model and convert the merged model.",
        ))
    else:
        out.append(_finding("llamacpp.unexpected-tensor", "pass", "only lora_A / lora_B tensors"))
    if base is None:
        out.append(_finding("llamacpp.shape", "unknown",
                            "no base model to compare shapes with; a wrong base fails at load time.",
                            "Re-run with --base <folder|hf:repo>."))
    else:
        problems, _ = shape_problems(summary, base)
        problems = [p for p in problems if p.kind != "full"]
        if problems:
            out.append(_finding(
                "llamacpp.shape", "fail",
                "tensor has incorrect shape (hint: maybe wrong base model?): " + "; ".join(p.text for p in problems[:3]),
                "Convert and serve with the exact base the adapter was trained on (vocabulary included).",
            ))
        else:
            out.append(_finding("llamacpp.shape", "pass", "adapter shapes fit the base"))
    if not any(f.status == "fail" for f in out):
        out.append(_finding("llamacpp.howto", "pass", "convert to GGUF, then serve with --lora.", fixes.llamacpp_convert(summary)))
    return out


# --------------------------------------------------------------------------- Ollama
def _ollama(summary: AdapterSummary) -> list[Finding]:
    return [_finding(
        "ollama.adapter-unsupported", "warn",
        "A Modelfile ADAPTER line failed with 'Error: LoRA adapters are no longer supported' on Ollama 0.40.15 "
        "(checked 2026-11-10; the version where this started is unknown, and other versions may differ).",
        fixes.merge_hint(summary),
    )]


# --------------------------------------------------------------------------- entry point
def evaluate(
    summary: AdapterSummary,
    base: BaseSummary | None,
    targets: tuple[str, ...],
    max_lora_rank: int,
) -> list[Finding]:
    """Run the common rules plus the rules of every requested target."""
    findings = _common(summary, base)
    if "vllm" in targets:
        findings += _vllm(summary, base, max_lora_rank)
    if "llamacpp" in targets:
        findings += _llamacpp(summary, base)
    if "ollama" in targets:
        findings += _ollama(summary)
    return findings


def rules_table() -> list[dict[str, Any]]:
    return [r.to_dict() for r in RULES]
