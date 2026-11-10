"""Where adapter and base files come from: a local folder or the Hugging Face Hub.

Both sources expose the same three calls (``read_text``, ``read_header``,
``describe``). The Hub source never downloads a whole file; it asks for byte
ranges, so only the safetensors header (tens of KB) crosses the network.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .errors import NotFoundError, SourceError
from .safetensors_header import Header, parse_header_json, parse_length, read_header_file

DEFAULT_ENDPOINT = "https://huggingface.co"
# Config files are read with one ranged request of at most this many bytes.
MAX_TEXT_BYTES = 1024 * 1024
HTTP_TIMEOUT = 60  # seconds; generous because the Hub can be slow

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_REV_RE = re.compile(r"^[\w][\w./-]*$")


class Source(Protocol):
    """Minimal interface the rest of the package needs."""

    def describe(self) -> str: ...
    def read_text(self, name: str) -> str: ...
    def read_header(self, name: str) -> Header: ...
    def exists(self, name: str) -> bool: ...


class LocalSource:
    """A folder on disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise SourceError(
                f"{self.root} is not a folder. Pass the folder that contains "
                "adapter_config.json, or use hf:<owner>/<repo> for a Hub repo."
            )

    def describe(self) -> str:
        return str(self.root)

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.is_file():
            raise NotFoundError(f"{path} does not exist.")
        return path

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def read_text(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")

    def read_header(self, name: str) -> Header:
        path = self._path(name)
        return read_header_file(str(path), label=str(path))


class HubSource:
    """A Hugging Face Hub repo, read with HTTP Range requests.

    ``opener`` is anything with ``open(request, timeout=...)`` (a
    ``urllib.request.OpenerDirector`` by default); tests inject a fake one.
    """

    def __init__(
        self,
        repo: str,
        revision: str = "main",
        token: str | None = None,
        endpoint: str | None = None,
        opener: Any = None,
    ) -> None:
        if not _REPO_RE.match(repo):
            raise SourceError(
                f"'{repo}' is not a Hub repo id. Use hf:<owner>/<repo>, e.g. hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0."
            )
        if not _REV_RE.match(revision):
            raise SourceError(f"'{revision}' is not a valid revision name.")
        self.repo = repo
        self.revision = revision
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self.opener = opener if opener is not None else urllib.request.build_opener()

    def describe(self) -> str:
        return f"hf:{self.repo}@{self.revision}"

    def url(self, name: str) -> str:
        rev = urllib.parse.quote(self.revision, safe="")
        return f"{self.endpoint}/{self.repo}/resolve/{rev}/{urllib.parse.quote(name)}"

    def build_request(self, name: str, start: int, end: int) -> urllib.request.Request:
        """Build a ranged GET. The token goes in an *unredirected* header so it
        is dropped if the Hub redirects to a CDN on another host."""
        request = urllib.request.Request(self.url(name), method="GET")
        request.add_header("Range", f"bytes={start}-{end}")
        request.add_header("User-Agent", "lora-preflight")
        if self.token:
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return request

    def _fetch(self, name: str, start: int, size: int) -> bytes:
        """Return ``size`` bytes of ``name`` starting at ``start``.

        Servers that ignore Range answer 200 with the whole body; in that case
        we read only what we need from the stream and close it.
        """
        end = start + size - 1
        request = self.build_request(name, start, end)
        try:
            response = self.opener.open(request, timeout=HTTP_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotFoundError(f"{self.describe()}: {name} not found (HTTP 404).") from exc
            if exc.code in (401, 403):
                raise SourceError(
                    f"{self.describe()}: HTTP {exc.code} for {name}. The repo is private or gated; "
                    "set the HF_TOKEN environment variable to a token that can read it."
                ) from exc
            raise SourceError(f"{self.describe()}: HTTP {exc.code} for {name}.") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise SourceError(
                f"{self.describe()}: network error while reading {name} ({exc}). "
                "Check your connection, or point lora-preflight at a local folder."
            ) from exc
        try:
            status = getattr(response, "status", 200)
            if status == 206:
                return response.read(size)
            data = response.read(end + 1)
            return data[start : end + 1]
        finally:
            response.close()

    def exists(self, name: str) -> bool:
        try:
            self._fetch(name, 0, 1)
        except NotFoundError:
            return False
        return True

    def read_text(self, name: str) -> str:
        data = self._fetch(name, 0, MAX_TEXT_BYTES)
        return data.decode("utf-8", errors="replace")

    def read_header(self, name: str) -> Header:
        label = f"{self.describe()}/{name}"
        length = parse_length(self._fetch(name, 0, 8), label)
        body = self._fetch(name, 8, length) if length else b""
        return parse_header_json(body, length, label)


def parse_location(spec: str | Path, opener: Any = None) -> Source:
    """Turn ``./folder`` or ``hf:owner/repo[@revision]`` into a source."""
    text = str(spec)
    if text.startswith("hf:"):
        repo, _, revision = text[3:].partition("@")
        return HubSource(repo, revision or "main", opener=opener)
    return LocalSource(text)
