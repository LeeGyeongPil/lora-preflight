"""Describe a PEFT LoRA adapter from adapter_config.json and its tensor names.

The tensor names are the source of truth: a config can say ``r=8`` while
``rank_pattern`` makes one layer rank 64, or say ``bias="all"`` while the file
holds no bias tensor. Everything here is derived from names and shapes only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import AdapterFormatError, NotFoundError, SourceError
from .safetensors_header import Header
from .sources import Source

CONFIG_FILE = "adapter_config.json"
SAFETENSORS_FILE = "adapter_model.safetensors"
BIN_FILE = "adapter_model.bin"

_PREFIX = "base_model.model."
_LORA_LINEAR = re.compile(r"^(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")
_LORA_EMBED = re.compile(r"^(?P<module>.+)\.lora_embedding_(?P<ab>[AB])$")
_MAGNITUDE = re.compile(r"^(?P<module>.+)\.lora_magnitude_vector(\.weight)?$")
_BASE_LAYER = re.compile(r"^(?P<module>.+)\.base_layer\.(weight|bias)$")
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def strip_prefix(name: str) -> str:
    """Drop the ``base_model.model.`` prefix PEFT puts on every saved key."""
    return name[len(_PREFIX):] if name.startswith(_PREFIX) else name


@dataclass
class ModuleInfo:
    """One adapted module (a linear layer or an embedding)."""

    name: str
    kind: str  # "linear" or "embedding"
    rank: int | None = None
    a_shape: tuple[int, ...] | None = None
    b_shape: tuple[int, ...] | None = None
    scale: float | None = None


@dataclass
class AdapterSummary:
    """Everything the rules need to know about an adapter."""

    source: str
    config: dict[str, Any]
    weights_format: str  # "safetensors", "bin" (tensors unreadable) or "none"
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    tensors: dict[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)  # stripped name -> (dtype, shape)
    full_tensors: list[str] = field(default_factory=list)  # whole-module weights (modules_to_save style)
    bias_tensors: list[str] = field(default_factory=list)
    magnitude_tensors: list[str] = field(default_factory=list)
    base_layer_tensors: list[str] = field(default_factory=list)
    unpaired: list[str] = field(default_factory=list)  # modules with only A or only B
    rank_mismatch: list[str] = field(default_factory=list)  # modules whose A rank != B rank
    unmatched_targets: list[str] = field(default_factory=list)
    # (required, actual) bytes when adapter_model.safetensors is shorter than its header requires.
    truncated: tuple[int, int] | None = None

    # --- config accessors -------------------------------------------------
    @property
    def r(self) -> int | None:
        value = self.config.get("r")
        return value if isinstance(value, int) else None

    @property
    def lora_alpha(self) -> float | None:
        value = self.config.get("lora_alpha")
        return value if isinstance(value, (int, float)) else None

    @property
    def rank_pattern(self) -> dict[str, int]:
        return dict(self.config.get("rank_pattern") or {})

    @property
    def base_model_name(self) -> str | None:
        value = self.config.get("base_model_name_or_path")
        return value if isinstance(value, str) and value else None

    @property
    def modules_to_save(self) -> list[str]:
        return [str(m) for m in (self.config.get("modules_to_save") or [])]

    # --- tensor-derived values --------------------------------------------
    @property
    def tensor_ranks(self) -> list[int]:
        return [m.rank for m in self.modules.values() if m.rank is not None]

    @property
    def max_rank(self) -> int | None:
        """Largest rank found in the tensors (None when tensors are unreadable)."""
        ranks = self.tensor_ranks
        return max(ranks) if ranks else None

    @property
    def config_max_rank(self) -> int | None:
        """Largest rank the config declares: ``r`` and every ``rank_pattern`` value."""
        values = [v for v in self.rank_pattern.values() if isinstance(v, int)]
        if self.r is not None:
            values.append(self.r)
        return max(values) if values else None

    @property
    def rank_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for rank in self.tensor_ranks:
            hist[rank] = hist.get(rank, 0) + 1
        return dict(sorted(hist.items()))

    @property
    def scales(self) -> list[float]:
        return sorted({round(m.scale, 6) for m in self.modules.values() if m.scale is not None})

    @property
    def layers(self) -> list[int]:
        found: set[int] = set()
        for name in self.modules:
            match = _LAYER_INDEX.search(name + ".")
            if match:
                found.add(int(match.group(1)))
        return sorted(found)

    @property
    def module_types(self) -> dict[str, int]:
        """Count of adapted modules per last name component (q_proj, v_proj, ...)."""
        counts: dict[str, int] = {}
        for name in self.modules:
            key = name.rsplit(".", 1)[-1]
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "weights_format": self.weights_format,
            "peft_version": self.config.get("peft_version"),
            "base_model_name_or_path": self.base_model_name,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "use_rslora": bool(self.config.get("use_rslora")),
            "use_dora": bool(self.config.get("use_dora")),
            "bias": self.config.get("bias", "none"),
            "target_modules": self.config.get("target_modules"),
            "modules_to_save": self.modules_to_save,
            "rank_pattern": self.rank_pattern,
            "max_rank_config": self.config_max_rank,
            "max_rank_tensors": self.max_rank,
            "rank_histogram": {str(k): v for k, v in self.rank_histogram.items()},
            "scales": self.scales,
            "module_count": len(self.modules),
            "layers": self.layers,
            "module_types": self.module_types,
            "full_tensors": self.full_tensors,
            "bias_tensors": self.bias_tensors,
            "magnitude_tensors": self.magnitude_tensors,
            "unmatched_targets": self.unmatched_targets,
        }


def match_pattern(pattern: dict[str, Any], name: str) -> Any:
    """PEFT's rule: a key matches if it is a suffix of the module name at a "." boundary."""
    for key, value in pattern.items():
        try:
            if re.match(rf"(.*\.)?({key})$", name):
                return value
        except re.error:
            continue
    return None


def scale_for(config: dict[str, Any], module: str, rank: int) -> float:
    """Effective LoRA scale of one module: alpha/r, or alpha/sqrt(r) for rsLoRA."""
    alpha = match_pattern(config.get("alpha_pattern") or {}, module)
    if alpha is None:
        alpha = config.get("lora_alpha", 8)
    alpha = float(alpha)
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank


def _classify(header: Header) -> AdapterSummary:
    summary = AdapterSummary(source="", config={}, weights_format="safetensors")
    a_shapes: dict[str, tuple[int, ...]] = {}
    b_shapes: dict[str, tuple[int, ...]] = {}
    kinds: dict[str, str] = {}
    for raw_name, info in header.tensors.items():
        name = strip_prefix(raw_name)
        summary.tensors[name] = (info.dtype, info.shape)
        for pattern, kind in ((_LORA_LINEAR, "linear"), (_LORA_EMBED, "embedding")):
            match = pattern.match(name)
            if match:
                module = match.group("module")
                kinds[module] = kind
                (a_shapes if match.group("ab") == "A" else b_shapes)[module] = info.shape
                break
        else:
            if _MAGNITUDE.match(name):
                summary.magnitude_tensors.append(name)
            elif _BASE_LAYER.match(name):
                summary.base_layer_tensors.append(name)
            elif name.endswith(".bias"):
                summary.bias_tensors.append(name)
            else:
                summary.full_tensors.append(name)
    for module in sorted(set(a_shapes) | set(b_shapes)):
        info = ModuleInfo(
            name=module,
            kind=kinds[module],
            a_shape=a_shapes.get(module),
            b_shape=b_shapes.get(module),
        )
        a, b = info.a_shape, info.b_shape
        if a is None or b is None:
            summary.unpaired.append(module)
        else:
            # Linear: A is [r, in], B is [out, r]. Embedding: A is [r, num], B is [dim, r].
            if len(a) == 2 and len(b) == 2:
                info.rank = a[0]
                if a[0] != b[1]:
                    summary.rank_mismatch.append(module)
        summary.modules[module] = info
    return summary


def _unmatched_targets(summary: AdapterSummary) -> list[str]:
    targets = summary.config.get("target_modules")
    if not isinstance(targets, list) or not summary.modules:
        return []
    names = list(summary.modules)
    missing = []
    for target in targets:
        if not isinstance(target, str):
            continue
        if not any(n == target or n.endswith("." + target) for n in names):
            missing.append(target)
    return missing


def load_adapter(source: Source) -> AdapterSummary:
    """Read config and tensor header from ``source`` and build the summary."""
    try:
        config = json.loads(source.read_text(CONFIG_FILE))
    except NotFoundError as exc:
        raise SourceError.coded("err.no_config", where=source.describe(), file=CONFIG_FILE) from exc
    except json.JSONDecodeError as exc:
        raise AdapterFormatError.coded(
            "err.config_json", path=f"{source.describe()}/{CONFIG_FILE}", exc=str(exc)
        ) from exc
    if not isinstance(config, dict):
        raise AdapterFormatError.coded("err.config_obj", path=f"{source.describe()}/{CONFIG_FILE}")
    peft_type = config.get("peft_type")
    if peft_type is not None and str(peft_type).upper() != "LORA":
        raise AdapterFormatError.coded(
            "err.peft_type", path=f"{source.describe()}/{CONFIG_FILE}", peft_type=repr(peft_type)
        )

    try:
        header = source.read_header(SAFETENSORS_FILE)
    except NotFoundError:
        has_bin = source.exists(BIN_FILE)
        if not has_bin:
            raise SourceError.coded(
                "err.no_weights", where=source.describe(), st=SAFETENSORS_FILE, bin=BIN_FILE
            ) from None
        summary = AdapterSummary(source=source.describe(), config=config, weights_format="bin")
        return summary

    summary = _classify(header)
    summary.source = source.describe()
    summary.config = config
    if header.is_truncated and header.file_size is not None:
        summary.truncated = (header.required_size, header.file_size)
    for module, info in summary.modules.items():
        if info.rank:
            info.scale = scale_for(config, module, info.rank)
    summary.unmatched_targets = _unmatched_targets(summary)
    return summary
