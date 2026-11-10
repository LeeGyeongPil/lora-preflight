"""Copy-paste commands attached to findings."""

from __future__ import annotations

import shlex

from .adapter import AdapterSummary

BASE_PLACEHOLDER = "<base-model>"


def adapter_ref(summary: AdapterSummary) -> str:
    """How a runtime should refer to the adapter: a Hub id or a (quoted) path."""
    if summary.source.startswith("hf:"):
        return summary.source[3:].partition("@")[0]
    return shlex.quote(summary.source)


def base_ref(summary: AdapterSummary) -> str:
    name = summary.base_model_name
    return shlex.quote(name) if name else BASE_PLACEHOLDER


def vllm_serve(summary: AdapterSummary, max_lora_rank: int) -> str:
    return (
        f"vllm serve {base_ref(summary)} --enable-lora "
        f"--lora-modules adapter={adapter_ref(summary)} --max-lora-rank {max_lora_rank}"
    )


def llamacpp_convert(summary: AdapterSummary) -> str:
    return (
        f"python convert_lora_to_gguf.py --base {base_ref(summary)} "
        f"--outfile adapter.gguf {adapter_ref(summary)}\n"
        "llama-server -m base.gguf --lora adapter.gguf"
    )


def merge_for_ollama(summary: AdapterSummary) -> str:
    base = repr(summary.base_model_name or BASE_PLACEHOLDER)  # Python string literal inside the shell command
    adapter = repr(adapter_ref(summary).strip("'"))
    return (
        "python -c \"from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained({base}), {adapter}).merge_and_unload(); "
        f"m.save_pretrained('merged'); AutoTokenizer.from_pretrained({base}).save_pretrained('merged')\"\n"
        "printf 'FROM ./merged\\n' > Modelfile\n"
        "ollama create my-merged-model -f Modelfile"
    )


def merge_hint(summary: AdapterSummary) -> str:
    return (
        "Merge the adapter into the base (`merge_and_unload()`), save it, and serve the merged model. "
        "Example:\n" + merge_for_ollama(summary)
    )
