# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-10

### Added
- `lora-preflight <adapter>` command: reads `adapter_config.json` and the safetensors header (no weights) and predicts whether vLLM, llama.cpp and Ollama will accept the adapter, with a copy-paste fix for each problem.
- Adapter anatomy summary: real ranks from tensor shapes (including `rank_pattern`), effective scale (alpha/r, alpha/sqrt(r) for rsLoRA), layer coverage, unmatched `target_modules`.
- `hf:<owner>/<repo>[@revision]` inputs, read with HTTP Range requests so only headers are fetched; `HF_TOKEN` and `HF_ENDPOINT` are honoured and the token is never forwarded across redirects.
- `--base`, `--target`, `--max-lora-rank`, `--json`, `--strict` options and exit codes 0 / 1 / 2.
- `lora-preflight demo` (synthetic practice adapters) and `lora-preflight rules` (rule table with sources and check dates).
- Python API: `lora_preflight.check()`, `Report`, `Finding`, `PreflightError` and subclasses.
