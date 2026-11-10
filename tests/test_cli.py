import json
from pathlib import Path

import pytest

from lora_preflight import __version__
from lora_preflight.cli import main
from lora_preflight.synth import VARIANTS


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_exit_codes_and_json_shape(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, str(world["plain_r8"]), "--target", "vllm,llamacpp", "--json")
    assert code == 0
    data = json.loads(out)
    assert {"adapter", "findings", "verdicts"} <= set(data)
    assert data["verdicts"] == {"vllm": "pass", "llamacpp": "pass"}
    assert all({"rule_id", "target", "status", "message", "fix", "source"} <= set(f) for f in data["findings"])

    code, out, _ = run(capsys, str(world["dora"]))
    assert code == 1
    assert "lora_magnitude_vector" in out and "DoRA" in out

    code, _, err = run(capsys, str(world["plain_r8"] / "missing"))
    assert code == 2
    assert err.startswith("error:")

    code, _, err = run(capsys, str(world["plain_r8"]), "--target", "tpu")
    assert code == 2 and "unknown target" in err

    code, _, err = run(capsys, str(world["plain_r8"]), "--max-lora-rank", "100")
    assert code == 2 and "1, 8, 16" in err


def test_strict_and_target_filter(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, str(world["plain_r8"]))
    assert code == 0  # ollama only warns
    assert "ollama" in out
    code, _, _ = run(capsys, str(world["plain_r8"]), "--strict")
    assert code == 1
    code, out, _ = run(capsys, str(world["plain_r8"]), "--strict", "--target", "vllm,llamacpp")
    assert code == 0
    assert "ollama" not in out.split("findings")[0]
    # raising the rank limit turns the rank failure into a pass
    code, _, _ = run(capsys, str(world["r128_all"]), "--target", "vllm")
    assert code == 1
    code, _, _ = run(capsys, str(world["r128_all"]), "--target", "vllm", "--max-lora-rank", "128")
    assert code == 0


def test_text_report_shows_anatomy_and_fix(world: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, str(world["rank_pattern"]), "--target", "vllm")
    assert code == 1
    assert "r=8 x2, r=64 x2" in out  # ranks found in tensors
    assert "--max-lora-rank 64" in out
    assert "rule-based predictions" in out


def test_rules_lists_sources(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "rules", "--json")
    assert code == 0
    rows = json.loads(out)
    assert rows and all(r["source_url"].startswith("https://") and r["checked"] for r in rows)
    assert {r["target"] for r in rows} >= {"vllm", "llamacpp", "ollama", "common"}
    code, text, _ = run(capsys, "rules")
    assert code == 0 and "checked 2026-11-10" in text


def test_demo_runs_offline_and_says_it_is_practice(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "demo")
    assert code == 0
    lines = out.splitlines()
    assert "synthetic" in lines[0] and "PRACTICE" in lines[0]
    rows = {line.split()[0]: line for line in lines if line.split() and line.split()[0] in VARIANTS}
    assert set(rows) == set(VARIANTS)
    assert "FAIL" in rows["dora"].split()[1]  # first verdict column is vllm
    assert "--max-lora-rank 128" in out


def test_demo_keep_writes_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "kept"
    code, _, _ = run(capsys, "demo", "--keep", str(target))
    assert code == 0
    assert (target / "dora" / "adapter_config.json").is_file()
    code, out, _ = run(capsys, str(target / "dora"), "--base", str(target / "base"))
    assert code == 1 and "lora_magnitude_vector" in out


def test_version_and_usage(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--version")
    assert code == 0 and __version__ in out
    code, out, _ = run(capsys)
    assert code == 2 and "usage" in out
