"""A safetensors file shorter than its header requires must be reported, not passed."""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lora_preflight import check, synth
from lora_preflight.cli import main
from lora_preflight.errors import SourceError
from lora_preflight.rules import rules_table


def truncate_half(path: Path) -> int:
    size = path.stat().st_size // 2
    with open(path, "r+b") as handle:
        handle.truncate(size)
    return size


class Resp:
    def __init__(self, body: bytes, status: int, headers: dict[str, str]) -> None:
        self.body, self.status, self.headers = body, status, headers

    def read(self, size: int = -1) -> bytes:
        return self.body if size is None or size < 0 else self.body[:size]

    def close(self) -> None:
        pass


class RangeOpener:
    """Serves files with Range support; ``total`` controls the Content-Range total."""

    def __init__(self, files: dict[str, bytes], total: str | None = "real") -> None:
        self.files, self.total = files, total

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> Resp:
        filename = request.full_url.rsplit("/", 1)[1]
        if filename not in self.files:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        data = self.files[filename]
        start, end = (int(x) for x in request.get_header("Range").split("=")[1].split("-"))
        body = data[start : end + 1]
        headers: dict[str, str] = {}
        if self.total == "real":
            headers["Content-Range"] = f"bytes {start}-{start + len(body) - 1}/{len(data)}"
        elif self.total is not None:
            headers["Content-Range"] = f"bytes {start}-{start + len(body) - 1}/{self.total}"
        return Resp(body, 206, headers)


def files_of(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


def test_truncated_local_adapter_fails_every_target(world: dict[str, Path]) -> None:
    truncate_half(world["plain_r8"] / "adapter_model.safetensors")
    report = check(world["plain_r8"], base=world["base"], targets="all")
    assert set(report.verdicts.values()) == {"fail"}
    hit = [f for f in report.findings if f.rule_id == "common.truncated"]
    assert len(hit) == 1 and hit[0].status == "fail"
    assert "adapter_model.safetensors" in hit[0].message


def test_truncated_local_adapter_exits_1(world: dict[str, Path]) -> None:
    truncate_half(world["plain_r8"] / "adapter_model.safetensors")
    assert main([str(world["plain_r8"]), "--target", "vllm,llamacpp"]) == 1


def test_complete_adapters_still_pass_truncation_rule(world: dict[str, Path]) -> None:
    for name in world:
        if name == "base":
            continue
        report = check(world[name], base=world["base"], targets="all")
        rows = [f for f in report.findings if f.rule_id == "common.truncated"]
        assert [f.status for f in rows] == ["pass"], name


def test_plain_adapter_verdicts_unchanged(world: dict[str, Path]) -> None:
    report = check(world["plain_r8"], base=world["base"], targets="vllm,llamacpp")
    assert report.verdicts == {"vllm": "pass", "llamacpp": "pass"}


def test_rule_is_listed() -> None:
    assert "common.truncated" in {r["id"] for r in rules_table()}


def test_hub_content_range_smaller_than_required_fails(world: dict[str, Path]) -> None:
    files = files_of(world["plain_r8"])
    files["adapter_model.safetensors"] = files["adapter_model.safetensors"][:-100]
    report = check("hf:me/adapter", targets="vllm", opener=RangeOpener(files))
    assert report.verdicts["vllm"] == "fail"
    assert any(f.rule_id == "common.truncated" and f.status == "fail" for f in report.findings)


def test_hub_complete_file_passes(world: dict[str, Path]) -> None:
    report = check("hf:me/adapter", targets="vllm", opener=RangeOpener(files_of(world["plain_r8"])))
    assert report.verdicts["vllm"] == "pass"


@pytest.mark.parametrize("total", [None, "*"])
def test_hub_unknown_total_skips_check(world: dict[str, Path], total: str | None) -> None:
    files = files_of(world["plain_r8"])
    files["adapter_model.safetensors"] = files["adapter_model.safetensors"][:-100]
    report = check("hf:me/adapter", targets="vllm", opener=RangeOpener(files, total=total))
    assert report.verdicts["vllm"] == "pass"  # unknown size: no false alarm
    assert [f.status for f in report.findings if f.rule_id == "common.truncated"] == ["pass"]


def test_truncated_single_base_is_source_error(world: dict[str, Path]) -> None:
    truncate_half(world["base"] / "model.safetensors")
    with pytest.raises(SourceError, match="Base file is truncated"):
        check(world["plain_r8"], base=world["base"], targets="all")
    assert main([str(world["plain_r8"]), "--base", str(world["base"])]) == 2


def test_truncated_base_shard_is_source_error(tmp_path: Path, world: dict[str, Path]) -> None:
    base = tmp_path / "sharded"
    base.mkdir()
    shard = base / "model-00001-of-00001.safetensors"
    shard.write_bytes((world["base"] / "model.safetensors").read_bytes())
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard.name}})
    )
    truncate_half(shard)
    with pytest.raises(SourceError, match="model-00001-of-00001.safetensors"):
        check(world["plain_r8"], base=base, targets="all")


def test_demo_still_has_seven_adapters(tmp_path: Path) -> None:
    _, adapters = synth.make_all(tmp_path / "w")
    assert len(adapters) == 7
