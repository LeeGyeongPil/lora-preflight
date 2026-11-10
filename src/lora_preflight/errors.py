"""Exception types raised by lora-preflight."""

from __future__ import annotations


class PreflightError(Exception):
    """Base class for every error this package raises on purpose."""


class AdapterFormatError(PreflightError):
    """A config or safetensors file exists but its content is broken."""


class SourceError(PreflightError):
    """A file could not be read (missing path, HTTP error, bad location)."""


class NotFoundError(SourceError):
    """The requested file does not exist in the source."""
