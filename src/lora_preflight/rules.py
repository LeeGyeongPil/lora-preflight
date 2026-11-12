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
from .i18n import J, M, tr
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
    Rule("common.truncated", "common", "adapter_model.safetensors must be at least as large as its header requires (8 + header length + last data_offsets end)",
         "https://github.com/huggingface/safetensors#format", "safetensors format spec; safetensors 0.8.0 load_file rejects short files (run locally)"),
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


def _finding(
    rule_id: str,
    status: str,
    key: str,
    params: dict[str, Any] | None = None,
    fix_key: str = "",
    fix_params: dict[str, Any] | None = None,
    fix_raw: str = "",
) -> Finding:
    """Build a finding. The English text is made from the message table; the keys travel along for the text report."""
    rule = _RULE_BY_ID[rule_id]
    params = params or {}
    fix_params = fix_params or {}
    message = tr("en", key, **params)
    fix = tr("en", fix_key, **fix_params) if fix_key else fix_raw
    return Finding(
        rule_id, rule.target, status, message, fix, f"{rule.source_url} ({rule.source_ref}, checked {rule.checked})",
        key=key, params=params, fix_key=fix_key, fix_params=fix_params,
    )


# --------------------------------------------------------------------------- base comparison
@dataclass
class ShapeProblem:
    kind: str  # "vocab", "linear" or "full"
    key: str  # message-table key
    params: dict[str, Any]

    @property
    def text(self) -> str:
        """The problem in English."""
        return tr("en", self.key, **self.params)

    def message(self) -> M:
        """The problem as a nested message, shown in the current language."""
        return M(self.key, **self.params)


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
                problems.append(ShapeProblem("vocab", "sp.vocab", {"a": info.a_shape[1], "b": base_shape[0], "name": name}))
            if info.b_shape[0] != base_shape[1]:
                problems.append(ShapeProblem("linear", "sp.emb_dim", {"name": name, "rows": info.b_shape[0], "dim": base_shape[1]}))
        else:
            if info.a_shape[1] != base_shape[1]:
                problems.append(ShapeProblem("linear", "sp.lin_in", {"name": name, "a": info.a_shape[1], "b": base_shape[1]}))
            if info.b_shape[0] != base_shape[0]:
                problems.append(ShapeProblem("linear", "sp.lin_out", {"name": name, "a": info.b_shape[0], "b": base_shape[0]}))
    for name in summary.full_tensors:
        base_shape = base.shapes.get(name)
        shape = summary.tensors[name][1]
        if base_shape is not None and tuple(base_shape) != tuple(shape):
            problems.append(ShapeProblem("full", "sp.full", {"name": name, "a": str(list(shape)), "b": str(list(base_shape))}))
    return problems, missing


def _joined(problems: list[ShapeProblem]) -> J:
    return J("; ", [p.message() for p in problems[:3]])


# --------------------------------------------------------------------------- common
def _common(summary: AdapterSummary, base: BaseSummary | None) -> list[Finding]:
    out: list[Finding] = []
    if summary.weights_format == "bin":
        out.append(_finding("common.weights-format", "unknown", "msg.bin_pickle", fix_key="fix.resave"))
        return out
    if summary.truncated is not None:
        need, have = summary.truncated
        out.append(_finding("common.truncated", "fail", "msg.truncated", {"need": need, "have": have}, "fix.truncated"))
    else:
        out.append(_finding("common.truncated", "pass", "msg.truncated_ok"))
    bad: list[M] = [M("pair.rank", name=m) for m in summary.rank_mismatch]
    bad += [M("pair.single", name=m) for m in summary.unpaired]
    if bad:
        out.append(_finding("common.pairing", "fail", "msg.pairing_bad", {"detail": J("; ", bad[:5])}, "fix.pairing"))
    else:
        out.append(_finding("common.pairing", "pass", "msg.pairing_ok", {"n": len(summary.modules)}))
    if summary.unmatched_targets:
        out.append(_finding("common.unmatched-target", "warn", "msg.unmatched",
                            {"names": ", ".join(summary.unmatched_targets)}, "fix.unmatched"))
    else:
        out.append(_finding("common.unmatched-target", "pass", "msg.unmatched_ok"))
    if base is not None:
        _, missing = shape_problems(summary, base)
        if missing:
            out.append(_finding("common.base-module-missing", "warn", "msg.base_missing",
                                {"n": len(missing), "example": missing[0]}, "fix.base_missing"))
        else:
            out.append(_finding("common.base-module-missing", "pass", "msg.base_missing_ok"))
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
        out.append(_finding("vllm.rank", "unknown", "msg.rank_unknown"))
    elif rank > max_lora_rank:
        need = _next_allowed(rank)
        if need is None:
            out.append(_finding(
                "vllm.rank", "fail", "msg.rank_too_big",
                {"rank": rank, "max": max_lora_rank, "allowed": ", ".join(map(str, VLLM_MAX_LORA_RANKS))},
                "fix.rank_too_big",
            ))
        else:
            out.append(_finding("vllm.rank", "fail", "msg.rank_over", {"rank": rank, "max": max_lora_rank},
                                fix_raw=fixes.vllm_serve(summary, need)))
    else:
        out.append(_finding("vllm.rank", "pass", "msg.rank_ok", {"rank": rank, "max": max_lora_rank}))

    # DoRA
    if cfg.get("use_dora") or summary.magnitude_tensors:
        out.append(_finding("vllm.dora", "fail", "msg.dora", fix_key="fix.merge_unload"))
    else:
        out.append(_finding("vllm.dora", "pass", "msg.dora_ok"))

    # bias
    bias = cfg.get("bias", "none")
    if bias != "none":
        fix_key = "fix.bias_retrain" if summary.bias_tensors else "fix.bias_config"
        out.append(_finding("vllm.bias", "fail", "msg.bias", {"bias": repr(bias)}, fix_key))
    else:
        out.append(_finding("vllm.bias", "pass", "msg.bias_ok"))
    if cfg.get("lora_bias"):
        out.append(_finding("vllm.lora-bias", "fail", "msg.lora_bias", fix_key="fix.lora_bias"))
    else:
        out.append(_finding("vllm.lora-bias", "pass", "msg.lora_bias_ok"))

    # modules_to_save / whole-module tensors
    allowed = ("classifier", "score")
    unsupported = [m for m in summary.modules_to_save if m not in allowed]
    for name in summary.full_tensors:
        module = name.rsplit(".", 1)[0]
        if module.rsplit(".", 1)[-1] not in allowed and module not in unsupported:
            unsupported.append(module)
    if unsupported:
        out.append(_finding("vllm.modules-to-save", "fail", "msg.mts", {"names": str(unsupported)}, "fix.mts"))
    else:
        out.append(_finding("vllm.modules-to-save", "pass", "msg.mts_ok"))

    # init_lora_weights
    init = cfg.get("init_lora_weights")
    if isinstance(init, str) and init.lower() not in _VLLM_OK_INITS:
        out.append(_finding("vllm.init-weights", "fail", "msg.init", {"init": repr(init)}, "fix.init"))
    else:
        out.append(_finding("vllm.init-weights", "pass", "msg.init_ok"))

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
        out.append(_finding("vllm.variants", "fail", "msg.variants", {"names": ", ".join(variants)}, "fix.merge_serve"))
    else:
        out.append(_finding("vllm.variants", "pass", "msg.variants_ok"))

    # base-dependent checks
    problems: list[ShapeProblem] = []
    if base is not None:
        problems, _ = shape_problems(summary, base)
    embed_modules = [m for m in summary.modules.values() if m.kind == "embedding"]
    if base is None and embed_modules:
        out.append(_finding("vllm.embedding-vocab", "unknown", "msg.embed_unknown", fix_key="fix.rerun_base"))
    else:
        both = [p for p in problems if p.kind in ("vocab", "full")]
        if both:
            out.append(_finding("vllm.embedding-vocab", "fail", "msg.joined", {"detail": _joined(both)}, "fix.vocab"))
        else:
            out.append(_finding("vllm.embedding-vocab", "pass", "msg.vocab_ok"))
    if base is not None:
        linear = [p for p in problems if p.kind == "linear"]
        if linear:
            out.append(_finding("vllm.shape", "fail", "msg.joined", {"detail": _joined(linear)}, "fix.shape"))
        else:
            out.append(_finding("vllm.shape", "pass", "msg.shape_ok"))

    # module names vLLM may not know
    unknown_modules = sorted({n.rsplit(".", 1)[-1] for n in summary.modules} - _VLLM_KNOWN_MODULES)
    if unknown_modules:
        out.append(_finding("vllm.unknown-module", "warn", "msg.unknown_module",
                            {"names": ", ".join(unknown_modules)}, "fix.unknown_module"))
    else:
        out.append(_finding("vllm.unknown-module", "pass", "msg.unknown_module_ok"))
    return out


# --------------------------------------------------------------------------- llama.cpp
def _is_norm(name: str) -> bool:
    return "_layernorm" in name or ".norm" in name


def _llamacpp(summary: AdapterSummary, base: BaseSummary | None) -> list[Finding]:
    out: list[Finding] = []
    if summary.weights_format == "bin":
        out.append(_finding("llamacpp.unexpected-tensor", "unknown", "msg.llama_unread", fix_key="fix.resave"))
        return out
    if summary.magnitude_tensors:
        out.append(_finding("llamacpp.dora", "fail", "msg.llama_dora", {"name": summary.magnitude_tensors[0]},
                            "fix.merge_convert"))
    else:
        out.append(_finding("llamacpp.dora", "pass", "msg.llama_dora_ok"))
    unexpected = [n for n in summary.full_tensors + summary.bias_tensors if not _is_norm(n)]
    if unexpected:
        shown = ", ".join(f"'{n}'" for n in unexpected[:3])
        out.append(_finding("llamacpp.unexpected-tensor", "fail", "msg.llama_unexpected", {"names": shown},
                            "fix.llama_unexpected"))
    else:
        out.append(_finding("llamacpp.unexpected-tensor", "pass", "msg.llama_unexpected_ok"))
    if base is None:
        out.append(_finding("llamacpp.shape", "unknown", "msg.llama_shape_unknown", fix_key="fix.rerun_base"))
    else:
        problems, _ = shape_problems(summary, base)
        problems = [p for p in problems if p.kind != "full"]
        if problems:
            out.append(_finding("llamacpp.shape", "fail", "msg.llama_shape", {"detail": _joined(problems)},
                                "fix.llama_shape"))
        else:
            out.append(_finding("llamacpp.shape", "pass", "msg.llama_shape_ok"))
    if not any(f.status == "fail" for f in out):
        out.append(_finding("llamacpp.howto", "pass", "msg.llama_howto", fix_raw=fixes.llamacpp_convert(summary)))
    return out


# --------------------------------------------------------------------------- Ollama
def _ollama(summary: AdapterSummary) -> list[Finding]:
    return [_finding("ollama.adapter-unsupported", "warn", "msg.ollama",
                     fix_key="fix.merge_hint", fix_params={"cmd": fixes.merge_for_ollama(summary)})]


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
