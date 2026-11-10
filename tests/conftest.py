"""Shared fixtures: a socket guard and synthetic base/adapter folders."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from lora_preflight import synth


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to open a network connection fails the test."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    """base/ plus one folder per synthetic variant, under a temporary folder."""
    base, adapters = synth.make_all(tmp_path / "world")
    return {"base": base, **adapters}
