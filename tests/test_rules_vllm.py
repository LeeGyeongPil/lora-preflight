import json
from pathlib import Path

import pytest

from lora_preflight import check
from lora_preflight.errors import PreflightError
from lora_preflight.synth import make_adapter, make_base


def finding(report, rule_id):
    return next(f for f in report.findings if f.rule_id == rule_id)


def test_rank_over_default_suggests_next_allowed_value(world: dict[str, Path], tmp_path: Path) -> None:
    report = check(world["r128_all"], base=world["base"], targets="vllm")
    f = finding(report, "vllm.rank")
    assert f.status == "fail"
    assert "--max-lora-rank 128" in f.fix
    assert "rank 128" in f.message

    ok = check(world["r128_all"], base=world["base"], targets="vllm", max_lora_rank=128)
    assert finding(ok, "vllm.rank").status == "pass"
    assert ok.verdicts["vllm"] == "pass"

    # rank 100 is not an allowed value itself; the next allowed one is 128
    folder = make_adapter(tmp_path / "r100", "plain_r8")
    config = json.loads((folder / "adapter_config.json").read_text())
    config["r"] = 100
    (folder / "adapter_config.json").write_text(json.dumps(config))
    assert "--max-lora-rank 128" in finding(check(folder, targets="vllm"), "vllm.rank").fix

    # rank 600 exceeds every allowed value
    config["r"] = 600
    (folder / "adapter_config.json").write_text(json.dumps(config))
    big = finding(check(folder, targets="vllm"), "vllm.rank")
    assert big.status == "fail"
    assert "vllm serve" not in big.fix
    assert "512" in big.message


def test_hidden_rank_in_rank_pattern_is_caught(world: dict[str, Path]) -> None:
    f = finding(check(world["rank_pattern"], targets="vllm"), "vllm.rank")
    assert f.status == "fail"
    assert "--max-lora-rank 64" in f.fix


def test_invalid_max_lora_rank_is_input_error(world: dict[str, Path]) -> None:
    with pytest.raises(PreflightError, match="1, 8, 16"):
        check(world["plain_r8"], max_lora_rank=100)


def test_config_feature_rejections(world: dict[str, Path], tmp_path: Path) -> None:
    cases = {
        "dora": "vllm.dora",
        "bias_all": "vllm.bias",
        "mts_embed_lmhead": "vllm.modules-to-save",
    }
    for variant, rule in cases.items():
        report = check(world[variant], base=world["base"], targets="vllm", max_lora_rank=16)
        assert finding(report, rule).status == "fail", variant

    def with_config(name: str, **changes: object):
        folder = make_adapter(tmp_path / name, "plain_r8")
        config = json.loads((folder / "adapter_config.json").read_text())
        config.update(changes)
        (folder / "adapter_config.json").write_text(json.dumps(config))
        return check(folder, targets="vllm")

    assert finding(with_config("lb", lora_bias=True), "vllm.lora-bias").status == "fail"
    assert finding(with_config("pissa", init_lora_weights="pissa"), "vllm.init-weights").status == "fail"
    assert finding(with_config("gauss", init_lora_weights="gaussian"), "vllm.init-weights").status == "pass"
    assert finding(with_config("score", modules_to_save=["score"]), "vllm.modules-to-save").status == "pass"
    assert finding(with_config("clf", modules_to_save=["classifier", "score"]), "vllm.modules-to-save").status == "pass"
    assert finding(with_config("lr", layer_replication=[[0, 1]]), "vllm.variants").status == "fail"
    assert finding(with_config("alora", alora_invocation_tokens=[1]), "vllm.variants").status == "fail"
    assert finding(with_config("ok"), "vllm.variants").status == "pass"


def test_bias_config_only_fix_mentions_config_edit(world: dict[str, Path]) -> None:
    f = finding(check(world["bias_all"], targets="vllm"), "vllm.bias")
    assert "adapter_config.json" in f.fix  # no bias tensors in the file: editing the config is enough


def test_embedding_vocab_mismatch_against_base(tmp_path: Path) -> None:
    base = make_base(tmp_path / "base", vocab=128)
    adapter = make_adapter(tmp_path / "ad", "lora_on_embed", vocab=133)
    report = check(adapter, base=base, targets="vllm")
    f = finding(report, "vllm.embedding-vocab")
    assert f.status == "fail"
    assert "133" in f.message and "128" in f.message
    assert report.verdicts["vllm"] == "fail"
    # same vocabulary is fine
    same = make_adapter(tmp_path / "same", "lora_on_embed", vocab=128)
    assert finding(check(same, base=base, targets="vllm"), "vllm.embedding-vocab").status == "pass"


def test_extended_vocab_whole_module_tensors(tmp_path: Path) -> None:
    base = make_base(tmp_path / "base", vocab=128)
    adapter = make_adapter(tmp_path / "ad", "mts_embed_lmhead", vocab=133)
    report = check(adapter, base=base, targets="vllm")
    assert finding(report, "vllm.embedding-vocab").status == "fail"
    assert finding(report, "vllm.modules-to-save").status == "fail"


def test_without_base_embedding_vocab_is_unknown_with_hint(tmp_path: Path) -> None:
    # default base name is a Hub-style id, which is never fetched implicitly
    report = check(make_adapter(tmp_path / "ad", "lora_on_embed"), targets="vllm")
    f = finding(report, "vllm.embedding-vocab")
    assert f.status == "unknown"
    assert "--base" in f.fix


def test_unknown_module_suffix_warns(tmp_path: Path) -> None:
    folder = make_adapter(tmp_path / "ad", "plain_r8")
    # rename q_proj tensors to a module name outside the known table
    import json as _json
    from lora_preflight.synth import write_safetensors
    from lora_preflight.safetensors_header import read_header_file
    header = read_header_file(str(folder / "adapter_model.safetensors"))
    tensors = {n.replace("q_proj", "qkv_proj"): (t.dtype, t.shape) for n, t in header.tensors.items()}
    write_safetensors(folder / "adapter_model.safetensors", tensors)
    f = finding(check(folder, targets="vllm"), "vllm.unknown-module")
    assert f.status == "warn"
    assert "qkv_proj" in f.message
    assert _json  # silence unused warning in minimal editors
