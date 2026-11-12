"""Command line entry point: ``lora-preflight``."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Sequence

from . import __version__, i18n
from .checker import check
from .errors import PreflightError
from .i18n import t
from .report import render_text
from .rules import KNOWN_TARGETS, VLLM_DEFAULT_MAX_LORA_RANK, rules_table
from .synth import VARIANTS, make_adapter, make_base

EXIT_OK, EXIT_FAIL, EXIT_INPUT = 0, 1, 2


class _PreParseError(Exception):
    """The --lang pre-parser met something it cannot read; the real parser will report it."""


class _QuietParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # never print, never exit
        raise _PreParseError(message)


def _split_lang(args: list[str]) -> tuple[str | None, list[str]]:
    """Take ``--lang`` out of ``args`` so it works before or after any command.

    Errors here are ignored on purpose: whatever this cannot read stays in the
    argument list and the real parser reports it, once.
    """
    pre = _QuietParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--lang")
    try:
        found, rest = pre.parse_known_args(args)
    except _PreParseError:
        return None, args
    return found.lang, rest


def _add_lang(parser: argparse.ArgumentParser) -> None:
    # Already consumed by _split_lang; declared so --help documents it.
    parser.add_argument("--lang", choices=i18n.LANGS, help=t("arg.lang"))


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-preflight",
        description=t("arg.check.desc"),
        epilog=t("arg.check.epilog"),
    )
    parser.add_argument("adapter", help=t("arg.adapter"))
    parser.add_argument("--base", help=t("arg.base"))
    parser.add_argument("--target", default="all", help=t("arg.target"))
    parser.add_argument("--max-lora-rank", type=int, default=VLLM_DEFAULT_MAX_LORA_RANK, help=t("arg.max_rank"))
    parser.add_argument("--json", action="store_true", help=t("arg.json"))
    parser.add_argument("--strict", action="store_true", help=t("arg.strict"))
    _add_lang(parser)
    return parser


def _run_check(argv: Sequence[str]) -> int:
    args = _check_parser().parse_args(argv)
    report = check(args.adapter, base=args.base, targets=args.target, max_lora_rank=args.max_lora_rank)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))  # machine output: always English
    else:
        print(render_text(report), end="")
    return report.exit_code(strict=args.strict)


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def _run_demo(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="lora-preflight demo", description=t("arg.demo.desc"))
    parser.add_argument("--keep", metavar="DIR", help=t("arg.keep"))
    _add_lang(parser)
    args = parser.parse_args(argv)
    if args.keep:
        Path(args.keep).mkdir(parents=True, exist_ok=True)
        return _demo_in(Path(args.keep).resolve())
    with tempfile.TemporaryDirectory(prefix="lora-preflight-demo-") as tmp:
        return _demo_in(Path(tmp).resolve())


def _pad(text: str, width: int) -> str:
    """Left-align ``text`` in ``width`` terminal columns (Hangul and other wide characters count as 2)."""
    cols = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)
    return text + " " * max(width - cols, 0)


def _demo_in(root: Path) -> int:
    base = make_base(root / "base")
    print(t("demo.banner"))
    print()
    print(_pad(t("demo.col_adapter"), 18) + "".join(f"{name:<10}" for name in KNOWN_TARGETS) + t("demo.col_first"))
    fixes: list[str] = []
    for variant in VARIANTS:
        folder = make_adapter(root / variant, variant)  # base_model_name_or_path stays a readable placeholder
        report = check(folder, base=base)
        cells = "".join(f"{report.verdicts[name].upper():<10}" for name in KNOWN_TARGETS)
        bad = [f for f in report.findings if f.status == "fail"]
        print(f"{variant:<18}{cells}{bad[0].rule_id if bad else '-'}")
        for f in bad:
            if f.target == "vllm" and f.fix:
                fixes.append(f"  {variant}: {f.rule_id}: {_first_line(f.fix_text())}")
    print()
    print(t("demo.fixes"))
    for line in fixes:
        print(line.replace(str(root), "<demo-dir>"))
    print()
    print(t("demo.details"))
    return EXIT_OK


def _run_rules(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="lora-preflight rules", description=t("arg.rules.desc"))
    parser.add_argument("--json", action="store_true", help=t("arg.rules_json"))
    _add_lang(parser)
    args = parser.parse_args(argv)
    table = rules_table()
    if args.json:
        print(json.dumps(table, indent=2, ensure_ascii=False))  # machine output: always English
        return EXIT_OK
    for row in table:
        print(f"{row['id']} [{row['target']}] {t('rule.' + row['id'])}")
        print(t("rules.source", url=row["source_url"]))
        print(t("rules.basis", ref=row["source_ref"], date=row["checked"]))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    lang_arg, args = _split_lang(args)
    if lang_arg is not None and lang_arg.lower() not in i18n.LANGS:
        i18n.set_lang(i18n.resolve_lang(None))
        print(t("cli.error", message=t("cli.bad_lang", value=repr(lang_arg))), file=sys.stderr)
        return EXIT_INPUT
    previous_lang = i18n.get_lang()
    i18n.set_lang(i18n.resolve_lang(lang_arg.lower() if lang_arg is not None else None))
    original_gettext = argparse._  # type: ignore[attr-defined]
    argparse._ = i18n.argparse_gettext  # type: ignore[attr-defined]  # argparse's own words follow the screen language
    try:
        return _dispatch(args)
    finally:
        argparse._ = original_gettext  # type: ignore[attr-defined]
        i18n.set_lang(previous_lang)


def _dispatch(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print(t("usage.text"), end="")
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
        text = t(exc.code, **exc.params) if exc.code else str(exc)
        print(t("cli.error", message=text), file=sys.stderr)
        return EXIT_INPUT
    except SystemExit as exc:  # argparse exits with 2 on bad flags, 0 on --help
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(main())
