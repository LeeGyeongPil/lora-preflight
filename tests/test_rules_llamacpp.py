from pathlib import Path

from lora_preflight import check
from lora_preflight.synth import make_adapter, make_base

# Outcome of running llama.cpp's convert_lora_to_gguf.py (master 45c8ea8374, 2026-11-10)
# on these synthetic adapters: True = conversion succeeded.
RECORDED_CONVERSION = {
    "plain_r8": True,
    "r128_all": True,
    "bias_all": True,
    "rank_pattern": True,
    "lora_on_embed": True,
    "dora": False,
    "mts_embed_lmhead": False,
}


def verdict(world: dict[str, Path], variant: str) -> str:
    return check(world[variant], base=world["base"], targets="llamacpp").verdicts["llamacpp"]


def test_matches_recorded_conversion_results(world: dict[str, Path]) -> None:
    for variant, converted in RECORDED_CONVERSION.items():
        expected = "pass" if converted else "fail"
        assert verdict(world, variant) == expected, variant
    dora = check(world["dora"], base=world["base"], targets="llamacpp")
    assert any("lora_magnitude_vector" in f.message for f in dora.findings if f.status == "fail")
    mts = check(world["mts_embed_lmhead"], base=world["base"], targets="llamacpp")
    assert any("lm_head.weight" in f.message for f in mts.findings if f.status == "fail")


def test_passing_adapter_gets_convert_command(world: dict[str, Path]) -> None:
    report = check(world["plain_r8"], base=world["base"], targets="llamacpp")
    howto = next(f for f in report.findings if f.rule_id == "llamacpp.howto")
    assert "convert_lora_to_gguf.py" in howto.fix and "--lora" in howto.fix


def test_shape_mismatch_and_missing_base(tmp_path: Path) -> None:
    base = make_base(tmp_path / "base", vocab=128)
    adapter = make_adapter(tmp_path / "ad", "lora_on_embed", vocab=133)
    report = check(adapter, base=base, targets="llamacpp")
    shape = next(f for f in report.findings if f.rule_id == "llamacpp.shape")
    assert shape.status == "fail"
    assert "133" in shape.message and "128" in shape.message

    # no --base, and the config's base is a Hub id that is not fetched: unknown + hint
    plain = make_adapter(tmp_path / "plain", "plain_r8")
    unknown = check(plain, targets="llamacpp")
    shape = next(f for f in unknown.findings if f.rule_id == "llamacpp.shape")
    assert shape.status == "unknown"
    assert "--base" in shape.fix
    assert unknown.verdicts["llamacpp"] == "unknown"
    assert unknown.exit_code() == 0


def test_wrong_base_is_flagged_as_linear_mismatch(tmp_path: Path) -> None:
    # an adapter whose A matrices expect 64 inputs against a base with a different hidden size
    import json
    base = make_base(tmp_path / "base")
    adapter = make_adapter(tmp_path / "ad", "plain_r8")
    from lora_preflight.safetensors_header import read_header_file
    from lora_preflight.synth import write_safetensors
    header = read_header_file(str(base / "model.safetensors"))
    tensors = {n: (t.dtype, (t.shape[0], 48) if "q_proj" in n else t.shape) for n, t in header.tensors.items()}
    write_safetensors(base / "model.safetensors", tensors)
    report = check(adapter, base=base, targets="llamacpp,vllm")
    assert report.verdicts == {"vllm": "fail", "llamacpp": "fail"}
    assert json  # keep import used
