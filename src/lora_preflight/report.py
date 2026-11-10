"""Findings, the report object, and plain-text / JSON rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapter import AdapterSummary
from .base import BaseSummary

STATUSES = ("pass", "warn", "fail", "unknown")
# Higher number = worse. "unknown" sits between warn and pass: it never fails a
# run, but it is shown so a missing base model is not mistaken for a pass.
_SEVERITY = {"pass": 0, "unknown": 1, "warn": 2, "fail": 3}

FOOTER = (
    "These verdicts are rule-based predictions from files only; no runtime was started. "
    "Rule sources and check dates: `lora-preflight rules`."
)


@dataclass
class Finding:
    """One rule outcome for one target runtime (or ``common``)."""

    rule_id: str
    target: str
    status: str
    message: str
    fix: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "target": self.target,
            "status": self.status,
            "message": self.message,
            "fix": self.fix,
            "source": self.source,
        }


def worst(statuses: list[str]) -> str:
    """The most severe status in ``statuses`` (``pass`` for an empty list)."""
    return max(statuses, key=lambda s: _SEVERITY[s], default="pass")


@dataclass
class Report:
    """Result of :func:`lora_preflight.check`."""

    adapter: AdapterSummary
    findings: list[Finding] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)
    base: BaseSummary | None = None
    max_lora_rank: int = 16
    tool_version: str = ""

    def exit_code(self, strict: bool = False) -> int:
        """0 when nothing blocks serving, 1 on FAIL (and on WARN with ``strict``)."""
        values = set(self.verdicts.values())
        if "fail" in values:
            return 1
        if strict and "warn" in values:
            return 1
        return 0

    def findings_for(self, target: str) -> list[Finding]:
        return [f for f in self.findings if f.target == target]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_version": self.tool_version,
            "adapter": self.adapter.to_dict(),
            "base": self.base.to_dict() if self.base else None,
            "max_lora_rank": self.max_lora_rank,
            "findings": [f.to_dict() for f in self.findings],
            "verdicts": dict(self.verdicts),
        }


def _fmt_scale(values: list[float]) -> str:
    return ", ".join(f"{v:g}" for v in values) if values else "n/a"


def _fmt_layers(layers: list[int], total: int | None) -> str:
    if not layers:
        return "none"
    span = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])
    return f"{len(layers)} of {total} ({span})" if total else f"{len(layers)} ({span})"


def summary_lines(report: Report) -> list[str]:
    """The 'adapter anatomy' block: rank, scale and coverage from tensors."""
    a = report.adapter
    lines = [f"adapter: {a.source}"]
    lines.append(f"base:    {report.base.source if report.base else '(not compared; pass --base <folder|hf:repo>)'}")
    cfg = [f"r={a.r}", f"alpha={a.lora_alpha:g}" if a.lora_alpha is not None else "alpha=?"]
    if a.config.get("use_rslora"):
        cfg.append("rsLoRA")
    if a.config.get("use_dora"):
        cfg.append("DoRA")
    if a.config.get("peft_version"):
        cfg.append(f"PEFT {a.config['peft_version']}")
    lines.append("config:  " + " ".join(cfg))
    if a.weights_format == "bin":
        lines.append("tensors: adapter_model.bin is a pickle; tensors were not read (config-only check)")
        return lines
    hist = ", ".join(f"r={r} x{n}" for r, n in a.rank_histogram.items()) or "none"
    lines.append(f"ranks:   {hist}  (config says r={a.r}, rank_pattern={a.rank_pattern or '{}'})")
    lines.append(f"scale:   {_fmt_scale(a.scales)}")
    types = ", ".join(f"{k} x{v}" for k, v in a.module_types.items()) or "none"
    total = report.base.num_layers if report.base else None
    lines.append(f"modules: {len(a.modules)} ({types}); layers {_fmt_layers(a.layers, total)}")
    extras = []
    if a.full_tensors:
        extras.append(f"{len(a.full_tensors)} whole-module tensor(s)")
    if a.magnitude_tensors:
        extras.append(f"{len(a.magnitude_tensors)} DoRA magnitude tensor(s)")
    if a.bias_tensors:
        extras.append(f"{len(a.bias_tensors)} bias tensor(s)")
    if extras:
        lines.append("extra:   " + ", ".join(extras))
    return lines


def render_text(report: Report) -> str:
    """Plain text, no colour, safe to paste into an issue or a CI log."""
    out = [f"lora-preflight {report.tool_version}".strip()]
    out += summary_lines(report)
    out.append("")
    out.append("verdicts")
    for target, verdict in report.verdicts.items():
        out.append(f"  {target:<9} {verdict.upper()}")
    shown = [f for f in report.findings if f.status != "pass"]
    passed = len(report.findings) - len(shown)
    out.append("")
    out.append(f"findings ({len(shown)} to look at, {passed} rule(s) passed)")
    for f in shown:
        out.append(f"  [{f.status.upper()}] {f.rule_id}: {f.message}")
        for i, line in enumerate(f.fix.splitlines()):
            out.append(f"      fix: {line}" if i == 0 else f"           {line}")
    for f in report.findings:
        if f.rule_id.endswith(".howto") and f.status == "pass":
            out.append(f"  [HOW] {f.rule_id}: {f.message}")
            for line in f.fix.splitlines():
                out.append(f"      {line}")
    out.append("")
    out.append(FOOTER)
    return "\n".join(out) + "\n"
