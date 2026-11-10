"""The one function CI scripts and the CLI both call: :func:`check`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .adapter import load_adapter
from .base import resolve_base
from .errors import PreflightError
from .report import Report, worst
from .rules import KNOWN_TARGETS, VLLM_DEFAULT_MAX_LORA_RANK, VLLM_MAX_LORA_RANKS, evaluate
from .sources import parse_location

_ALIASES = {"llama.cpp": "llamacpp", "llama-cpp": "llamacpp", "llama_cpp": "llamacpp"}


def normalize_targets(targets: Iterable[str] | str) -> tuple[str, ...]:
    """Accept ``"all"``, ``"vllm,llamacpp"`` or an iterable; return known names in a fixed order."""
    if isinstance(targets, str):
        targets = [t for t in targets.split(",") if t.strip()]
    names: list[str] = []
    for raw in targets:
        name = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name == "all":
            return KNOWN_TARGETS
        if name not in KNOWN_TARGETS:
            raise PreflightError(
                f"unknown target '{raw}'. Choose from: {', '.join(KNOWN_TARGETS)}, or 'all'."
            )
        if name not in names:
            names.append(name)
    if not names:
        raise PreflightError(f"no target given. Choose from: {', '.join(KNOWN_TARGETS)}, or 'all'.")
    return tuple(t for t in KNOWN_TARGETS if t in names)


def check(
    adapter: str | Path,
    base: str | Path | None = None,
    targets: Iterable[str] | str = KNOWN_TARGETS,
    max_lora_rank: int = VLLM_DEFAULT_MAX_LORA_RANK,
    opener: Any = None,
) -> Report:
    """Check a LoRA adapter against the serving runtimes in ``targets``.

    ``adapter`` and ``base`` are a local folder or ``hf:<owner>/<repo>[@revision]``.
    Without ``base``, the adapter's ``base_model_name_or_path`` is used only when it
    is a local folder; shape rules that need a base report ``unknown`` otherwise.
    ``opener`` replaces the HTTP opener (for tests and proxies).

    Raises :class:`PreflightError` subclasses for bad input.
    """
    chosen = normalize_targets(targets)
    if max_lora_rank not in VLLM_MAX_LORA_RANKS:
        raise PreflightError(
            f"max_lora_rank {max_lora_rank} is not a value vLLM accepts. "
            f"Use one of: {', '.join(map(str, VLLM_MAX_LORA_RANKS))}."
        )
    source = parse_location(adapter, opener=opener)
    summary = load_adapter(source)
    base_summary = resolve_base(base, summary.base_model_name, opener=opener)
    findings = evaluate(summary, base_summary, chosen, max_lora_rank)
    verdicts = {
        target: worst([f.status for f in findings if f.target in (target, "common")])
        for target in chosen
    }
    return Report(
        adapter=summary,
        findings=findings,
        verdicts=verdicts,
        base=base_summary,
        max_lora_rank=max_lora_rank,
        tool_version=__version__,
    )

