"""lora-preflight: check a PEFT LoRA adapter against vLLM, llama.cpp and Ollama.

Public API::

    from lora_preflight import check
    report = check("./out", base="./base-model", targets="vllm,llamacpp")
    print(report.verdicts)          # {"vllm": "pass", "llamacpp": "fail"}
    for f in report.findings:
        print(f.rule_id, f.status, f.fix)

Everything else in the package is internal and may change between releases.
"""

__version__ = "0.1.0"

from .checker import check  # noqa: E402
from .errors import AdapterFormatError, NotFoundError, PreflightError, SourceError  # noqa: E402
from .report import Finding, Report  # noqa: E402

__all__ = [
    "__version__",
    "check",
    "Report",
    "Finding",
    "PreflightError",
    "AdapterFormatError",
    "SourceError",
    "NotFoundError",
]
