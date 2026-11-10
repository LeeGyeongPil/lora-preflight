"""Read the header of a safetensors file using only the standard library.

Layout of a safetensors file:

    [8 bytes: little-endian u64 N][N bytes: JSON header][tensor data ...]

The JSON header maps tensor names to ``{"dtype", "shape", "data_offsets"}`` and
may carry a ``__metadata__`` entry. Reading only the first ``8 + N`` bytes is
enough to learn every tensor name, shape and dtype, which is why a remote file
can be inspected with two small HTTP Range requests.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Callable

from .errors import AdapterFormatError

# The reference implementation refuses headers larger than 100 MB.
MAX_HEADER_BYTES = 100 * 1024 * 1024

DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

# (offset, length) -> bytes. Local files and HTTP Range requests both fit this shape.
ReadAt = Callable[[int, int], bytes]


@dataclass(frozen=True)
class TensorInfo:
    """Name, dtype and shape of one tensor (no data)."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass
class Header:
    """Parsed safetensors header."""

    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    # Bytes the header says the file must have: 8 + header length + largest data_offsets end.
    required_size: int = 0
    # Actual file size, or None when it is unknown (remote server gave no total length).
    file_size: int | None = None

    @property
    def is_truncated(self) -> bool:
        """True only when the real size is known and smaller than the header requires."""
        return self.file_size is not None and self.file_size < self.required_size


def parse_length(prefix: bytes, label: str) -> int:
    """Decode the 8-byte header length and check it is plausible."""
    if len(prefix) < 8:
        raise AdapterFormatError(
            f"{label}: file is shorter than 8 bytes, so it is not a safetensors file."
        )
    (length,) = struct.unpack("<Q", prefix[:8])
    if length > MAX_HEADER_BYTES:
        raise AdapterFormatError(
            f"{label}: header length {length} is larger than {MAX_HEADER_BYTES} bytes. "
            "The file is truncated, corrupt, or not a safetensors file "
            "(for example a pickle .bin or a Git LFS pointer)."
        )
    return length


def parse_header_json(
    body: bytes, length: int, label: str, file_size: int | None = None
) -> Header:
    """Decode the JSON part of the header into a :class:`Header`."""
    if len(body) < length:
        raise AdapterFormatError(
            f"{label}: header says {length} bytes but only {len(body)} are available. "
            "The file is truncated; copy or download it again."
        )
    try:
        raw = json.loads(body[:length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterFormatError(f"{label}: header is not valid JSON ({exc}).") from exc
    if not isinstance(raw, dict):
        raise AdapterFormatError(f"{label}: header JSON must be an object.")
    header = Header(required_size=8 + length, file_size=file_size)
    for name, entry in raw.items():
        if name == "__metadata__":
            if isinstance(entry, dict):
                header.metadata = {str(k): str(v) for k, v in entry.items()}
            continue
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(x) for x in entry["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterFormatError(
                f"{label}: tensor {name!r} has no valid dtype/shape entry."
            ) from exc
        header.tensors[name] = TensorInfo(name, dtype, shape)
        # data_offsets only feed the size check; a missing or odd entry is ignored here.
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in offsets)
        ):
            header.required_size = max(header.required_size, 8 + length + offsets[1])
    return header


def read_header(read_at: ReadAt, label: str, file_size: int | None = None) -> Header:
    """Read a header through ``read_at`` (two reads: length, then JSON)."""
    length = parse_length(read_at(0, 8), label)
    body = read_at(8, length) if length else b""
    return parse_header_json(body, length, label, file_size)


def read_header_file(path: str, label: str | None = None) -> Header:
    """Read the header of a local safetensors file."""
    name = label or str(path)

    def read_at(offset: int, size: int) -> bytes:
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read(size)

    return read_header(read_at, name, file_size=os.path.getsize(path))
