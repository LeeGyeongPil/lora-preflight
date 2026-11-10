import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lora_preflight import check
from lora_preflight.base import load_base
from lora_preflight.errors import NotFoundError, SourceError
from lora_preflight.safetensors_header import read_header_file
from lora_preflight.sources import HubSource, parse_location
from lora_preflight.synth import write_safetensors

ZERO = bytes(1)


class FakeResponse:
    def __init__(self, body: bytes, status: int) -> None:
        self.body, self.status, self.closed = body, status, False

    def read(self, size: int = -1) -> bytes:
        return self.body if size is None or size < 0 else self.body[:size]

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Serves in-memory repos by URL, honouring (or ignoring) the Range header."""

    def __init__(self, repos: dict[str, dict[str, bytes]], honor_range: bool = True) -> None:
        self.repos, self.honor_range = repos, honor_range
        self.requests: list[urllib.request.Request] = []
        self.sent = 0

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        path = request.full_url.split("://", 1)[1].split("/", 1)[1]  # owner/repo/resolve/rev/file
        owner, name, _, _, filename = path.split("/", 4)
        files = self.repos.get(f"{owner}/{name}", {})
        if filename not in files:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        data = files[filename]
        rng = request.get_header("Range")
        if rng and self.honor_range:
            start, end = (int(x) for x in rng.split("=")[1].split("-"))
            body = data[start : end + 1]
            self.sent += len(body)
            return FakeResponse(body, 206)
        self.sent += len(data)
        return FakeResponse(data, 200)


def folder_files(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


def test_reads_only_header_bytes_with_range(world: dict[str, Path]) -> None:
    adapter_files = folder_files(world["r128_all"])
    adapter_files["adapter_model.safetensors"] += ZERO * 5_000_000  # pretend the weights are large
    base_files = folder_files(world["base"])
    base_files["model.safetensors"] += ZERO * 9_000_000
    opener = FakeOpener({"me/adapter": adapter_files, "me/base": base_files})
    report = check("hf:me/adapter", base="hf:me/base", targets="vllm", opener=opener)
    assert report.verdicts["vllm"] == "fail"  # rank 128 > 16, found without downloading weights
    assert report.adapter.source == "hf:me/adapter@main"
    assert opener.requests
    assert all(r.get_header("Range") for r in opener.requests)
    total = len(adapter_files["adapter_model.safetensors"]) + len(base_files["model.safetensors"])
    assert opener.sent < total // 100


def test_range_ignored_by_server_still_reads_small_prefix(world: dict[str, Path]) -> None:
    opener = FakeOpener({"me/adapter": folder_files(world["plain_r8"])}, honor_range=False)
    header = HubSource("me/adapter", opener=opener).read_header("adapter_model.safetensors")
    assert len(header.tensors) == 8


def test_token_not_forwarded_on_redirect() -> None:
    hub = HubSource("me/private", token="hf_secret", opener=FakeOpener({}))
    request = hub.build_request("adapter_config.json", 0, 99)
    assert "Authorization" not in request.headers  # not among the normal (redirect-forwarded) headers
    assert request.unredirected_hdrs.get("Authorization") == "Bearer hf_secret"
    # What urllib builds for a redirect to another host must not carry the token.
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example.invalid/blob"  # type: ignore[arg-type]
    )
    assert redirected is not None
    assert "authorization" not in {k.lower() for k, _ in redirected.header_items()}


def test_sharded_base_index(tmp_path: Path, world: dict[str, Path]) -> None:
    full = read_header_file(str(world["base"] / "model.safetensors"))
    names = sorted(full.tensors)
    half = len(names) // 2
    shards = {"model-00001.safetensors": names[:half], "model-00002.safetensors": names[half:]}
    files: dict[str, bytes] = {}
    weight_map: dict[str, str] = {}
    for shard, members in shards.items():
        path = tmp_path / shard
        write_safetensors(path, {n: (full.tensors[n].dtype, full.tensors[n].shape) for n in members})
        files[shard] = path.read_bytes()
        weight_map.update({n: shard for n in members})
    files["model.safetensors.index.json"] = json.dumps({"weight_map": weight_map}).encode()
    base = load_base(HubSource("me/base", opener=FakeOpener({"me/base": files})))
    assert set(base.shapes) == set(names)
    assert base.vocab_size == 128


def test_missing_file_is_source_error() -> None:
    hub = HubSource("me/adapter", opener=FakeOpener({}))
    with pytest.raises(NotFoundError, match="adapter_config.json"):
        hub.read_text("adapter_config.json")
    with pytest.raises(SourceError, match="adapter_config.json"):
        check("hf:me/adapter", opener=FakeOpener({}))


def test_auth_error_mentions_hf_token() -> None:
    class Denied:
        def open(self, request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    with pytest.raises(SourceError, match="HF_TOKEN"):
        HubSource("me/private", opener=Denied()).read_text("adapter_config.json")


def test_bad_locations_are_rejected() -> None:
    with pytest.raises(SourceError, match="Hub repo id"):
        parse_location("hf:not-a-repo-id")
    with pytest.raises(SourceError, match="not a folder"):
        parse_location("/definitely/not/here")
