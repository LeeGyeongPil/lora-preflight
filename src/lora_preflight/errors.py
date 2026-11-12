"""Exception types raised by lora-preflight."""

from __future__ import annotations

from typing import Any, Mapping

from . import i18n


class PreflightError(Exception):
    """Base class for every error this package raises on purpose.

    ``str(exc)`` is always English. ``code`` and ``params`` (optional) name the
    message in :mod:`lora_preflight.i18n`, so the command line can show it in
    the chosen language.
    """

    def __init__(self, message: str, code: str | None = None, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.params = dict(params) if params else {}

    @classmethod
    def coded(cls, code: str, /, **params: Any) -> "PreflightError":
        """Build the error from a message-table key; the English text becomes ``str(exc)``."""
        return cls(i18n.tr("en", code, **params), code=code, params=params)


class AdapterFormatError(PreflightError):
    """A config or safetensors file exists but its content is broken."""


class SourceError(PreflightError):
    """A file could not be read (missing path, HTTP error, bad location)."""


class NotFoundError(SourceError):
    """The requested file does not exist in the source."""
