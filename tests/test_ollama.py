from pathlib import Path

from lora_preflight import check
from lora_preflight.synth import VARIANTS


def test_adapter_import_points_to_merge(world: dict[str, Path]) -> None:
    for variant in VARIANTS:
        report = check(world[variant], base=world["base"], targets="ollama")
        f = next(x for x in report.findings if x.rule_id == "ollama.adapter-unsupported")
        assert f.status == "warn", variant
        assert "merge_and_unload" in f.fix
        assert "FROM" in f.fix
        assert "0.40.15" in f.message
        assert report.verdicts == {"ollama": "warn"}


def test_merge_command_uses_quoted_python_strings(world: dict[str, Path]) -> None:
    report = check(world["plain_r8"], base=world["base"], targets="ollama")
    fix = report.findings[-1].fix
    assert "AutoModelForCausalLM.from_pretrained('" in fix
    assert "PeftModel.from_pretrained(" in fix
