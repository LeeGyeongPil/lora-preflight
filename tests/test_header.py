import struct
from pathlib import Path

import pytest

from lora_preflight.errors import AdapterFormatError
from lora_preflight.safetensors_header import read_header_file
from lora_preflight.synth import write_safetensors


def test_roundtrip_matches_written_tensors(tmp_path: Path) -> None:
    written = {
        "a.weight": ("F32", (4, 3)),
        "b.weight": ("F16", (2,)),
        "c": ("BF16", (5, 1, 2)),
    }
    path = tmp_path / "x.safetensors"
    write_safetensors(path, written)
    header = read_header_file(str(path))
    assert {n: (t.dtype, t.shape) for n, t in header.tensors.items()} == written
    assert header.metadata == {"format": "pt"}


def test_corrupt_header_raises_format_error(tmp_path: Path) -> None:
    too_long = tmp_path / "too_long.safetensors"
    too_long.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(AdapterFormatError, match="too_long.safetensors"):
        read_header_file(str(too_long))

    bad_json = tmp_path / "bad_json.safetensors"
    bad_json.write_bytes(struct.pack("<Q", 5) + b"{oops")
    with pytest.raises(AdapterFormatError, match="bad_json.safetensors"):
        read_header_file(str(bad_json))

    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(AdapterFormatError, match="truncated.safetensors"):
        read_header_file(str(truncated))

    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"abc")
    with pytest.raises(AdapterFormatError, match="tiny.safetensors"):
        read_header_file(str(tiny))


def test_entry_without_shape_is_rejected(tmp_path: Path) -> None:
    body = b'{"w":{"dtype":"F32"}}'
    path = tmp_path / "noshape.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(AdapterFormatError, match="'w'"):
        read_header_file(str(path))
