"""Screen language: message tables, language choice, and that Korean really shows up."""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from lora_preflight import i18n
from lora_preflight.cli import main
from lora_preflight.errors import PreflightError
from lora_preflight.rules import RULES

HANGUL = re.compile(r"[가-힣]")


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


# --------------------------------------------------------------------------- tables
def test_both_languages_have_the_same_keys_and_placeholders() -> None:
    en, ko = i18n.MESSAGES["en"], i18n.MESSAGES["ko"]
    assert set(en) == set(ko)
    for key in en:
        if key.startswith("argparse."):
            continue  # argparse's own %-style phrases
        assert _placeholders(en[key]) == _placeholders(ko[key]), key


def test_every_rule_has_a_title_in_both_languages() -> None:
    for rule in RULES:
        for lang in i18n.LANGS:
            assert f"rule.{rule.id}" in i18n.MESSAGES[lang]


def test_korean_text_has_no_particle_placeholders() -> None:
    for key, text in i18n.MESSAGES["ko"].items():
        assert "이(가)" not in text and "을(를)" not in text and "은(는)" not in text, key


def test_korean_table_is_korean() -> None:
    translated = [k for k, v in i18n.MESSAGES["ko"].items() if HANGUL.search(v)]
    assert len(translated) > 0.9 * len(i18n.MESSAGES["ko"])


# --------------------------------------------------------------------------- choosing a language
def test_priority_flag_then_tool_variable_then_locale_variables() -> None:
    env = {"LORA_PREFLIGHT_LANG": "ko", "LC_ALL": "en_US.UTF-8", "LC_MESSAGES": "en_US", "LANG": "en_US"}
    assert i18n.resolve_lang("en", env) == "en"  # --lang beats everything
    assert i18n.resolve_lang(None, env) == "ko"  # tool variable beats locale
    del env["LORA_PREFLIGHT_LANG"]
    assert i18n.resolve_lang(None, env) == "en"  # LC_ALL is English here
    assert i18n.resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US"}) == "ko"
    assert i18n.resolve_lang(None, {"LC_MESSAGES": "ko_KR", "LANG": "en_US"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"


@pytest.mark.parametrize("env", [{}, {"LANG": "C"}, {"LANG": "POSIX"}, {"LANG": ""}, {"LANG": "fr_FR.UTF-8"}, {"LORA_PREFLIGHT_LANG": "xx"}])
def test_anything_but_korean_is_english(env: dict[str, str]) -> None:
    assert i18n.resolve_lang(None, env) == "en"


def test_empty_values_fall_through_to_the_next_variable() -> None:
    assert i18n.resolve_lang(None, {"LORA_PREFLIGHT_LANG": "", "LC_ALL": "", "LANG": "ko_KR.UTF-8"}) == "ko"


def test_environment_variables_pick_the_language(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")
    code, out, _ = run(capsys, "rules")
    assert code == 0 and HANGUL.search(out)
    monkeypatch.setenv("LANG", "C")
    _, out, _ = run(capsys, "rules")
    assert not HANGUL.search(out)
    monkeypatch.setenv("LORA_PREFLIGHT_LANG", "ko")
    _, out, _ = run(capsys, "rules")
    assert HANGUL.search(out)
    _, out, _ = run(capsys, "rules", "--lang", "en")  # the flag wins over the variable
    assert not HANGUL.search(out)


def test_set_lang_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="choose from en, ko"):
        i18n.set_lang("fr")


# --------------------------------------------------------------------------- screens in Korean
def test_text_report_in_korean_keeps_ids_statuses_and_commands(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, str(world["r128_all"]), "--target", "vllm", "--lang", "ko")
    assert code == 1
    assert re.search(r"판정", out) and re.search(r"점검 결과", out)
    assert "[FAIL] vllm.rank:" in out and "VLLM" not in out
    assert re.search(r"LoRA 랭크가 max_lora_rank 보다 큽니다: 128 > 16", out)
    assert "vllm serve" in out and "--max-lora-rank 128" in out  # copy-paste command is untouched
    assert "rule-based predictions" not in out and re.search(r"규칙으로 예측", out)


def test_howto_and_ollama_lines_in_korean(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, str(world["plain_r8"]), "--lang", "ko")
    assert code == 0
    assert re.search(r"\[방법\] llamacpp.howto: GGUF 로 변환", out)
    assert "convert_lora_to_gguf.py" in out
    assert re.search(r"\[WARN\] ollama.adapter-unsupported:", out) and "merge_and_unload()" in out


def test_demo_in_korean(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "demo", "--lang", "ko")
    assert code == 0
    assert re.search(r"^연습 모드:", out) and re.search(r"vLLM 수정 방법", out)
    assert "dora" in out and "--max-lora-rank 128" in out
    header = next(line for line in out.splitlines() if "첫 문제" in line)
    assert header.startswith("어댑터") and "adapter" not in header  # first column header is translated


def test_demo_header_in_english(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = run(capsys, "demo", "--lang", "en")
    assert any(line.startswith("adapter ") and "first problem" in line for line in out.splitlines())


def test_usage_lists_lang_on_every_command() -> None:
    for lang in i18n.LANGS:
        text = i18n.tr(lang, "usage.text")
        for line in text.splitlines():
            if line.lstrip().startswith(("lora-preflight demo", "lora-preflight rules")):
                assert "[--lang en|ko]" in line


def test_korean_templates_have_no_particle_right_after_a_placeholder() -> None:
    # A fixed particle (가/이/을/를/은/는/와/과/로) after a value is wrong for half of all values.
    bad = re.compile(r"\}\s?(?:가|이|을|를|은|는|와|과|으로|로)(?![가-힣])")
    offenders = [k for k, v in i18n.MESSAGES["ko"].items() if bad.search(v)]
    assert offenders == []


def test_rules_in_korean_and_english(capsys: pytest.CaptureFixture[str]) -> None:
    _, ko, _ = run(capsys, "rules", "--lang", "ko")
    assert re.search(r"vllm.dora \[vllm\] DoRA 는 지원하지 않습니다", ko)
    assert re.search(r"출처: https://", ko) and re.search(r"확인일 2026-11-10", ko)
    _, en, _ = run(capsys, "rules", "--lang", "en")
    assert "vllm.dora [vllm] DoRA is not supported" in en and "checked 2026-11-10" in en


def test_usage_and_help_in_korean(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--lang", "ko")
    assert code == 2 and re.search(r"^사용법:", out)
    code, out, _ = run(capsys, "x", "--help", "--lang", "ko")
    assert code == 0
    assert re.search(r"사용법: lora-preflight", out) and re.search(r"옵션:", out) and re.search(r"위치 인자:", out)
    assert "--lang {en,ko}" in out


def test_argparse_errors_in_korean_keep_exit_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "--lang", "ko", "--max-lora-rank", "abc", "x")
    assert code == 2 and re.search(r"오류: .*abc", err)
    code, _, err = run(capsys, "--lang", "ko")  # no adapter, but usage text is not an error screen
    assert code == 2
    code, _, err = run(capsys, "--lang", "ko", "--bogus")
    assert code == 2 and re.search(r"다음 인자가 필요합니다", err)
    code, _, err = run(capsys, "--lang", "en", "--max-lora-rank", "abc", "x")
    assert code == 2 and "invalid int value" in err and not HANGUL.search(err)


def test_library_errors_are_localized_by_code(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, str(world["plain_r8"] / "missing"), "--lang", "ko")
    assert code == 2 and err.startswith("오류:") and re.search(r"폴더가 아닙니다", err)
    code, _, err = run(capsys, str(world["plain_r8"]), "--target", "tpu", "--lang", "ko")
    assert code == 2 and re.search(r"알 수 없는 target: 'tpu'", err)
    code, _, err = run(capsys, str(world["plain_r8"]), "--max-lora-rank", "100", "--lang", "ko")
    assert code == 2 and re.search(r"받지 않는 max_lora_rank 값: 100", err) and "1, 8, 16" in err
    code, _, err = run(capsys, str(world["plain_r8"] / "missing"), "--lang", "en")
    assert code == 2 and err.startswith("error:") and not HANGUL.search(err)


def test_broken_safetensors_header_error_in_korean(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
    (folder / "adapter_model.safetensors").write_bytes(b"abc")
    code, _, err = run(capsys, str(folder), "--lang", "ko")
    assert code == 2 and re.search(r"8바이트보다 짧아", err)
    code, _, err = run(capsys, str(folder), "--lang", "en")
    assert code == 2 and "shorter than 8 bytes" in err


def test_error_text_for_plain_exceptions_and_str_stays_english() -> None:
    assert str(PreflightError("plain message")) == "plain message"  # old call style still works
    exc = PreflightError.coded("err.target_none", known="vllm")
    assert str(exc) == "no target given. Choose from: vllm, or 'all'." and exc.code == "err.target_none"
    assert exc.params == {"known": "vllm"}


# --------------------------------------------------------------------------- --lang itself
def test_bad_lang_value_reports_once_with_exit_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "--lang", "xx", "rules")
    assert code == 2 and out == ""
    assert err.count("xx") == 1 and err.count("\n") == 1 and err.startswith("error:")


def test_lang_without_a_value_reports_once(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "rules", "--lang")
    assert code == 2
    assert err.count("error:") == 1 and err.count("--lang") >= 1


def test_lang_works_before_and_after_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    _, before, _ = run(capsys, "--lang", "ko", "rules")
    _, after, _ = run(capsys, "rules", "--lang", "ko")
    _, equals, _ = run(capsys, "rules", "--lang=ko")
    assert before == after == equals and HANGUL.search(before)
    code, out, _ = run(capsys, "--lang", "ko", "--version")
    assert code == 0 and out.startswith("lora-preflight ")


def test_language_does_not_leak_after_main_returns(capsys: pytest.CaptureFixture[str]) -> None:
    run(capsys, "rules", "--lang", "ko")
    assert i18n.get_lang() == "en"


# --------------------------------------------------------------------------- machine output
def test_json_output_is_identical_in_both_languages(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("dora", "r128_all", "plain_r8", "mts_embed_lmhead"):
        code_en, en, _ = run(capsys, str(world[name]), "--base", str(world["base"]), "--json", "--lang", "en")
        code_ko, ko, _ = run(capsys, str(world[name]), "--base", str(world["base"]), "--json", "--lang", "ko")
        assert code_en == code_ko and en == ko, name
        assert not HANGUL.search(en)
    _, en, _ = run(capsys, "rules", "--json", "--lang", "en")
    _, ko, _ = run(capsys, "rules", "--json", "--lang", "ko")
    assert en == ko


def test_exit_codes_do_not_depend_on_language(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    for lang in ("en", "ko"):
        assert run(capsys, str(world["plain_r8"]), "--lang", lang)[0] == 0
        assert run(capsys, str(world["plain_r8"]), "--strict", "--lang", lang)[0] == 1
        assert run(capsys, str(world["dora"]), "--lang", lang)[0] == 1
        assert run(capsys, str(world["plain_r8"] / "nope"), "--lang", lang)[0] == 2
