"""Command line entry point: ``lora-preflight``."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from . import __version__
from .checker import check
from .errors import PreflightError
from .report import render_text
from .rules import KNOWN_TARGETS, VLLM_DEFAULT_MAX_LORA_RANK, rules_table
from .synth import VARIANTS, make_adapter, make_base

EXIT_OK, EXIT_FAIL, EXIT_INPUT = 0, 1, 2

DEMO_BANNER = (
    "PRACTICE MODE: these are synthetic LoRA adapters (zero-filled weights, tensor names and shapes "
    "as PEFT 0.21.2 saved them). They are not real training results."
)

USAGE = """\
usage:
  lora-preflight <adapter> [--base <folder|hf:repo>] [--target vllm,llamacpp,ollama|all]
                 [--max-lora-rank N] [--json] [--strict]
  lora-preflight demo [--keep <dir>]
  lora-preflight rules [--json]
  lora-preflight --version

<adapter> is a folder (adapter_config.json + adapter_model.safetensors) or hf:<owner>/<repo>[@revision].
Exit codes: 0 no FAIL, 1 FAIL (or WARN with --strict), 2 bad input.
"""


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-preflight",
        description="Check a PEFT LoRA adapter against vLLM, llama.cpp and Ollama before serving it.",
        epilog="Other commands: `lora-preflight demo`, `lora-preflight rules`.",
    )
    parser.add_argument("adapter", help="adapter folder, or hf:<owner>/<repo>[@revision]")
    parser.add_argument("--base", help="base model folder or hf:<owner>/<repo> (default: the adapter's local base path, if it exists)")
    parser.add_argument("--target", default="all", help="comma list of vllm,llamacpp,ollama (default: all)")
    parser.add_argument("--max-lora-rank", type=int, default=VLLM_DEFAULT_MAX_LORA_RANK,
                        help="the --max-lora-rank you plan to give vLLM (default 16)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--strict", action="store_true", help="exit 1 on WARN as well as FAIL")
    return parser


def _run_check(argv: Sequence[str]) -> int:
    args = _check_parser().parse_args(argv)
    report = check(args.adapter, base=args.base, targets=args.target, max_lora_rank=args.max_lora_rank)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_text(report), end="")
    return report.exit_code(strict=args.strict)


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def _run_demo(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="lora-preflight demo", description="Check 7 synthetic practice adapters.")
    parser.add_argument("--keep", metavar="DIR", help="write the practice files to DIR and keep them")
    args = parser.parse_args(argv)
    if args.keep:
        Path(args.keep).mkdir(parents=True, exist_ok=True)
        return _demo_in(Path(args.keep).resolve())
    with tempfile.TemporaryDirectory(prefix="lora-preflight-demo-") as tmp:
        return _demo_in(Path(tmp).resolve())


def _demo_in(root: Path) -> int:
    base = make_base(root / "base")
    print(DEMO_BANNER)
    print()
    print(f"{'adapter':<18}" + "".join(f"{t:<10}" for t in KNOWN_TARGETS) + "first problem")
    fixes: list[str] = []
    for variant in VARIANTS:
        folder = make_adapter(root / variant, variant)  # base_model_name_or_path stays a readable placeholder
        report = check(folder, base=base)
        cells = "".join(f"{report.verdicts[t].upper():<10}" for t in KNOWN_TARGETS)
        bad = [f for f in report.findings if f.status == "fail"]
        print(f"{variant:<18}{cells}{bad[0].rule_id if bad else '-'}")
        for f in bad:
            if f.target == "vllm" and f.fix:
                fixes.append(f"  {variant}: {f.rule_id}: {_first_line(f.fix)}")
    print()
    print("vLLM fixes")
    for line in fixes:
        print(line.replace(str(root), "<demo-dir>"))
    print()
    print("Details for one adapter: lora-preflight demo --keep ./demo, then lora-preflight ./demo/dora --base ./demo/base")
    return EXIT_OK


def _run_rules(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="lora-preflight rules", description="Show every rule and where it comes from.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    table = rules_table()
    if args.json:
        print(json.dumps(table, indent=2, ensure_ascii=False))
        return EXIT_OK
    for row in table:
        print(f"{row['id']} [{row['target']}] {row['title']}")
        print(f"    source: {row['source_url']}")
        print(f"    basis:  {row['source_ref']}; checked {row['checked']}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return EXIT_OK if args else EXIT_INPUT
    if args[0] == "--version":
        print(f"lora-preflight {__version__}")
        return EXIT_OK
    try:
        if args[0] == "demo":
            return _run_demo(args[1:])
        if args[0] == "rules":
            return _run_rules(args[1:])
        return _run_check(args)
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except SystemExit as exc:  # argparse exits with 2 on bad flags, 0 on --help
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(main())
