"""Describe the base model an adapter was trained on (names and shapes only)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import NotFoundError, SourceError
from .sources import Source, parse_location

SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"


@dataclass
class BaseSummary:
    """Shapes of every base tensor plus a few facts from config.json."""

    source: str
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    architectures: list[str] = field(default_factory=list)
    num_layers: int | None = None

    @property
    def vocab_size(self) -> int | None:
        """Row count of the input embedding, if the base has one under a known name."""
        for name in ("model.embed_tokens.weight", "embed_tokens.weight",
                     "transformer.wte.weight", "model.decoder.embed_tokens.weight"):
            shape = self.shapes.get(name)
            if shape and len(shape) == 2:
                return shape[0]
        return None

    def shape_of(self, module: str) -> tuple[int, ...] | None:
        return self.shapes.get(module + ".weight")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "tensor_count": len(self.shapes),
            "architectures": self.architectures,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
        }


def load_base(source: Source) -> BaseSummary:
    """Read the base tensor header(s): a single file or a sharded index."""
    summary = BaseSummary(source=source.describe())
    try:
        index = json.loads(source.read_text(INDEX_FILE))
        files = sorted({str(v) for v in index.get("weight_map", {}).values()})
        if not files:
            raise SourceError(f"{source.describe()}/{INDEX_FILE} has an empty weight_map.")
    except NotFoundError:
        files = [SINGLE_FILE]
    except json.JSONDecodeError as exc:
        raise SourceError(f"{source.describe()}/{INDEX_FILE} is not valid JSON ({exc}).") from exc
    for name in files:
        try:
            header = source.read_header(name)
        except NotFoundError as exc:
            raise SourceError(
                f"{source.describe()} has no {name}. The base must be safetensors "
                f"({SINGLE_FILE} or {INDEX_FILE} with shards)."
            ) from exc
        if header.is_truncated:
            raise SourceError(
                f"Base file is truncated: {source.describe()}/{name} has {header.file_size} bytes but "
                f"its header requires {header.required_size}. Copy or download the base model again."
            )
        for tensor_name, info in header.tensors.items():
            summary.shapes[tensor_name] = info.shape
    try:
        config = json.loads(source.read_text(CONFIG_FILE))
        archs = config.get("architectures") or []
        summary.architectures = [str(a) for a in archs]
        layers = config.get("num_hidden_layers")
        summary.num_layers = layers if isinstance(layers, int) else None
    except (NotFoundError, json.JSONDecodeError, AttributeError):
        pass  # config.json is optional; shapes are what the rules need
    return summary


def resolve_base(
    base: str | Path | None, adapter_base_name: str | None, opener: Any = None
) -> BaseSummary | None:
    """Pick the base to compare against.

    An explicit ``base`` always wins. Otherwise the adapter's own
    ``base_model_name_or_path`` is used only if it is a local folder; a Hub id is
    never fetched implicitly (pass ``--base hf:<id>`` to opt in to network use).
    """
    if base is not None:
        return load_base(parse_location(base, opener=opener))
    if adapter_base_name and not adapter_base_name.startswith("hf:") and os.path.isdir(adapter_base_name):
        try:
            return load_base(parse_location(adapter_base_name))
        except SourceError:
            return None
    return None
