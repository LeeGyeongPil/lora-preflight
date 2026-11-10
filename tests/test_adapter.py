import json
from pathlib import Path

import pytest

from lora_preflight import check
from lora_preflight.adapter import load_adapter, scale_for
from lora_preflight.errors import AdapterFormatError, SourceError
from lora_preflight.sources import LocalSource
from lora_preflight.synth import make_adapter


def test_rank_pattern_reveals_hidden_rank(world: dict[str, Path]) -> None:
    summary = load_adapter(LocalSource(world["rank_pattern"]))
    assert summary.r == 8
    assert summary.max_rank == 64
    assert summary.rank_histogram == {8: 2, 64: 2}


def test_effective_scale_plain_and_rslora(world: dict[str, Path], tmp_path: Path) -> None:
    plain = load_adapter(LocalSource(world["plain_r8"]))
    assert plain.scales == [2.0]
    # rsLoRA divides alpha by sqrt(r): alpha 16, r 64 -> 2.0 (plain LoRA would give 0.25)
    assert scale_for({"lora_alpha": 16, "use_rslora": True}, "m", 64) == 2.0
    assert scale_for({"lora_alpha": 16, "use_rslora": False}, "m", 64) == 0.25
    folder = make_adapter(tmp_path / "rs", "rank_pattern")
    config = json.loads((folder / "adapter_config.json").read_text())
    config["use_rslora"] = True
    config["alpha_pattern"] = {"v_proj": 64}
    (folder / "adapter_config.json").write_text(json.dumps(config))
    summary = load_adapter(LocalSource(folder))
    scales = {m.name.rsplit(".", 1)[-1]: m.scale for m in summary.modules.values()}
    assert scales["v_proj"] == 8.0  # alpha_pattern 64 / sqrt(64)
    assert scales["q_proj"] == pytest.approx(16 / 8 ** 0.5)


def test_unmatched_target_module_warns(tmp_path: Path) -> None:
    folder = make_adapter(tmp_path / "a", "plain_r8")
    config = json.loads((folder / "adapter_config.json").read_text())
    config["target_modules"] = ["q_proj", "v_proj", "gate_proj"]
    (folder / "adapter_config.json").write_text(json.dumps(config))
    report = check(folder, targets="vllm")
    finding = next(f for f in report.findings if f.rule_id == "common.unmatched-target")
    assert finding.status == "warn"
    assert "gate_proj" in finding.message
    assert report.adapter.unmatched_targets == ["gate_proj"]


def test_summary_counts_layers_and_types(world: dict[str, Path]) -> None:
    summary = load_adapter(LocalSource(world["plain_r8"]))
    assert summary.layers == [0, 1]
    assert summary.module_types == {"q_proj": 2, "v_proj": 2}
    dora = load_adapter(LocalSource(world["dora"]))
    assert len(dora.magnitude_tensors) == 4
    mts = load_adapter(LocalSource(world["mts_embed_lmhead"]))
    assert sorted(mts.full_tensors) == ["lm_head.weight", "model.embed_tokens.weight"]


def test_missing_config_gives_actionable_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(SourceError, match="adapter_config.json"):
        load_adapter(LocalSource(tmp_path / "empty"))
    with pytest.raises(SourceError, match="not a folder"):
        LocalSource(tmp_path / "nope")


def test_non_lora_peft_type_is_rejected(tmp_path: Path) -> None:
    folder = make_adapter(tmp_path / "a", "plain_r8")
    config = json.loads((folder / "adapter_config.json").read_text())
    config["peft_type"] = "PREFIX_TUNING"
    (folder / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(AdapterFormatError, match="PREFIX_TUNING"):
        load_adapter(LocalSource(folder))


def test_bin_only_adapter_is_config_only(tmp_path: Path) -> None:
    folder = make_adapter(tmp_path / "a", "plain_r8")
    (folder / "adapter_model.safetensors").unlink()
    (folder / "adapter_model.bin").write_bytes(b"\x80\x02pickle")
    report = check(folder)
    assert report.adapter.weights_format == "bin"
    assert any(f.rule_id == "common.weights-format" and f.status == "unknown" for f in report.findings)
    assert report.exit_code() == 0
    (folder / "adapter_model.bin").unlink()
    with pytest.raises(SourceError, match="neither"):
        check(folder)
