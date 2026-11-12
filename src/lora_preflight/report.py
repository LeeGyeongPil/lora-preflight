"""Findings, the report object, and plain-text / JSON rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapter import AdapterSummary
from .base import BaseSummary
from .i18n import t

STATUSES = ("pass", "warn", "fail", "unknown")
# Higher number = worse. "unknown" sits between warn and pass: it never fails a
# run, but it is shown so a missing base model is not mistaken for a pass.
_SEVERITY = {"pass": 0, "unknown": 1, "warn": 2, "fail": 3}


@dataclass
class Finding:
    """One rule outcome for one target runtime (or ``common``).

    ``message`` and ``fix`` are always English (they are what JSON carries).
    ``key``/``params`` and ``fix_key``/``fix_params`` name the same text in the
    message table so the text report can show it in the chosen language; they
    are not serialized.
    """

    rule_id: str
    target: str
    status: str
    message: str
    fix: str = ""
    source: str = ""
    key: str = field(default="", compare=False, repr=False)
    params: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    fix_key: str = field(default="", compare=False, repr=False)
    fix_params: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "target": self.target,
            "status": self.status,
            "message": self.message,
            "fix": self.fix,
            "source": self.source,
        }

    def message_text(self) -> str:
        """The message in the current screen language."""
        return t(self.key, **self.params) if self.key else self.message

    def fix_text(self) -> str:
        """The fix in the current screen language (copy-paste commands stay as they are)."""
        return t(self.fix_key, **self.fix_params) if self.fix_key else self.fix


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
    return ", ".join(f"{v:g}" for v in values) if values else t("rep.na")


def _fmt_layers(layers: list[int], total: int | None) -> str:
    if not layers:
        return t("rep.none")
    span = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])
    if total:
        return t("rep.layers_of", n=len(layers), total=total, span=span)
    return t("rep.layers", n=len(layers), span=span)


def summary_lines(report: Report) -> list[str]:
    """The 'adapter anatomy' block: rank, scale and coverage from tensors."""
    a = report.adapter
    lines = [t("rep.header_adapter", value=a.source)]
    lines.append(t("rep.header_base", value=report.base.source if report.base else t("rep.base_none")))
    cfg = [f"r={a.r}", f"alpha={a.lora_alpha:g}" if a.lora_alpha is not None else "alpha=?"]
    if a.config.get("use_rslora"):
        cfg.append("rsLoRA")
    if a.config.get("use_dora"):
        cfg.append("DoRA")
    if a.config.get("peft_version"):
        cfg.append(f"PEFT {a.config['peft_version']}")
    lines.append(t("rep.config", value=" ".join(cfg)))
    if a.weights_format == "bin":
        lines.append(t("rep.bin"))
        return lines
    hist = ", ".join(f"r={r} x{n}" for r, n in a.rank_histogram.items()) or t("rep.none")
    lines.append(t("rep.ranks", hist=hist, r=a.r, pattern=a.rank_pattern or "{}"))
    lines.append(t("rep.scale", value=_fmt_scale(a.scales)))
    types = ", ".join(f"{k} x{v}" for k, v in a.module_types.items()) or t("rep.none")
    total = report.base.num_layers if report.base else None
    lines.append(t("rep.modules", n=len(a.modules), types=types, layers=_fmt_layers(a.layers, total)))
    extras = []
    if a.full_tensors:
        extras.append(t("rep.extra_full", n=len(a.full_tensors)))
    if a.magnitude_tensors:
        extras.append(t("rep.extra_magnitude", n=len(a.magnitude_tensors)))
    if a.bias_tensors:
        extras.append(t("rep.extra_bias", n=len(a.bias_tensors)))
    if extras:
        lines.append(t("rep.extra", value=", ".join(extras)))
    return lines


def render_text(report: Report) -> str:
    """Plain text, no colour, safe to paste into an issue or a CI log.

    The words are in the current screen language (:func:`lora_preflight.i18n.set_lang`);
    JSON output (``Report.to_dict``) is always English.
    """
    out = [f"lora-preflight {report.tool_version}".strip()]
    out += summary_lines(report)
    out.append("")
    out.append(t("rep.verdicts"))
    for target, verdict in report.verdicts.items():
        out.append(f"  {target:<9} {verdict.upper()}")
    shown = [f for f in report.findings if f.status != "pass"]
    passed = len(report.findings) - len(shown)
    out.append("")
    out.append(t("rep.findings", shown=len(shown), passed=passed))
    for f in shown:
        out.append(f"  [{f.status.upper()}] {f.rule_id}: {f.message_text()}")
        for i, line in enumerate(f.fix_text().splitlines()):
            out.append(t("rep.fix_first", line=line) if i == 0 else f"           {line}")
    for f in report.findings:
        if f.rule_id.endswith(".howto") and f.status == "pass":
            out.append(t("rep.how", rule_id=f.rule_id, message=f.message_text()))
            for line in f.fix_text().splitlines():
                out.append(f"      {line}")
    out.append("")
    out.append(t("rep.footer"))
    return "\n".join(out) + "\n"


