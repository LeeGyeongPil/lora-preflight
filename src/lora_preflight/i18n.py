"""Screen language: one message table per language, and the rules that pick one.

Only text meant for people is translated. JSON output, rule IDs, status words
(PASS/WARN/FAIL/UNKNOWN), file and module names, source URLs and copy-paste
commands are never translated.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

LANGS = ("en", "ko")
DEFAULT_LANG = "en"
ENV_VAR = "LORA_PREFLIGHT_LANG"


class M:
    """A nested message: a key plus parameters, resolved in whatever language is shown."""

    def __init__(self, key: str, **params: Any) -> None:
        self.key = key
        self.params = params


class J:
    """Several nested messages (or plain strings) joined with ``sep``."""

    def __init__(self, sep: str, items: list[Any]) -> None:
        self.sep = sep
        self.items = items


MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # ---- command line
        "usage.text": (
            "usage:\n"
            "  lora-preflight <adapter> [--base <folder|hf:repo>] [--target vllm,llamacpp,ollama|all]\n"
            "                 [--max-lora-rank N] [--json] [--strict] [--lang en|ko]\n"
            "  lora-preflight demo [--keep <dir>] [--lang en|ko]\n"
            "  lora-preflight rules [--json] [--lang en|ko]\n"
            "  lora-preflight --version\n"
            "\n"
            "<adapter> is a folder (adapter_config.json + adapter_model.safetensors) or hf:<owner>/<repo>[@revision].\n"
            "Exit codes: 0 no FAIL, 1 FAIL (or WARN with --strict), 2 bad input.\n"
            "Language: --lang en|ko, or LORA_PREFLIGHT_LANG, or LC_ALL / LC_MESSAGES / LANG (anything but ko is English).\n"
        ),
        "cli.error": "error: {message}",
        "cli.bad_lang": "invalid --lang value {value}; choose en or ko",
        "arg.check.desc": "Check a PEFT LoRA adapter against vLLM, llama.cpp and Ollama before serving it.",
        "arg.check.epilog": "Other commands: `lora-preflight demo`, `lora-preflight rules`.",
        "arg.adapter": "adapter folder, or hf:<owner>/<repo>[@revision]",
        "arg.base": "base model folder or hf:<owner>/<repo> (default: the adapter's local base path, if it exists)",
        "arg.target": "comma list of vllm,llamacpp,ollama (default: all)",
        "arg.max_rank": "the --max-lora-rank you plan to give vLLM (default 16)",
        "arg.json": "print the report as JSON",
        "arg.strict": "exit 1 on WARN as well as FAIL",
        "arg.lang": "screen language: en or ko (default: LORA_PREFLIGHT_LANG, then LC_ALL / LC_MESSAGES / LANG)",
        "arg.demo.desc": "Check 7 synthetic practice adapters.",
        "arg.keep": "write the practice files to DIR and keep them",
        "arg.rules.desc": "Show every rule and where it comes from.",
        "arg.rules_json": "print the rule table as JSON",
        # ---- demo / rules output
        "demo.banner": (
            "PRACTICE MODE: these are synthetic LoRA adapters (zero-filled weights, tensor names and shapes "
            "as PEFT 0.21.2 saved them). They are not real training results."
        ),
        "demo.col_adapter": "adapter",
        "demo.col_first": "first problem",
        "demo.fixes": "vLLM fixes",
        "demo.details": "Details for one adapter: lora-preflight demo --keep ./demo, then lora-preflight ./demo/dora --base ./demo/base",
        "rules.source": "    source: {url}",
        "rules.basis": "    basis:  {ref}; checked {date}",
        # ---- text report
        "rep.header_adapter": "adapter: {value}",
        "rep.header_base": "base:    {value}",
        "rep.base_none": "(not compared; pass --base <folder|hf:repo>)",
        "rep.config": "config:  {value}",
        "rep.bin": "tensors: adapter_model.bin is a pickle; tensors were not read (config-only check)",
        "rep.ranks": "ranks:   {hist}  (config says r={r}, rank_pattern={pattern})",
        "rep.scale": "scale:   {value}",
        "rep.modules": "modules: {n} ({types}); layers {layers}",
        "rep.extra": "extra:   {value}",
        "rep.extra_full": "{n} whole-module tensor(s)",
        "rep.extra_magnitude": "{n} DoRA magnitude tensor(s)",
        "rep.extra_bias": "{n} bias tensor(s)",
        "rep.none": "none",
        "rep.na": "n/a",
        "rep.layers_of": "{n} of {total} ({span})",
        "rep.layers": "{n} ({span})",
        "rep.verdicts": "verdicts",
        "rep.findings": "findings ({shown} to look at, {passed} rule(s) passed)",
        "rep.fix_first": "      fix: {line}",
        "rep.how": "  [HOW] {rule_id}: {message}",
        "rep.footer": (
            "These verdicts are rule-based predictions from files only; no runtime was started. "
            "Rule sources and check dates: `lora-preflight rules`."
        ),
        # ---- rule titles
        "rule.common.weights-format": "adapter_model.bin cannot be read without torch; only config is checked",
        "rule.common.truncated": "adapter_model.safetensors must be at least as large as its header requires (8 + header length + last data_offsets end)",
        "rule.common.pairing": "every LoRA module needs matching A and B tensors of the same rank",
        "rule.common.unmatched-target": "a target_modules entry that matches no tensor was silently skipped in training",
        "rule.common.base-module-missing": "adapter modules that do not exist in the base model",
        "rule.vllm.rank": "max rank (r and rank_pattern) must be <= --max-lora-rank (default 16; allowed 1,8,16,32,64,128,256,320,512)",
        "rule.vllm.dora": "DoRA is not supported",
        "rule.vllm.bias": "bias other than 'none' is not supported",
        "rule.vllm.lora-bias": "lora_bias is not supported",
        "rule.vllm.modules-to-save": "modules_to_save / whole-module tensors only allowed for classifier and score",
        "rule.vllm.init-weights": "init_lora_weights such as PiSSA, OLoRA, CorDA, LoftQ are not supported",
        "rule.vllm.variants": "aLoRA, layer_replication, BD-LoRA and QALoRA are not supported",
        "rule.vllm.embedding-vocab": "embedding LoRA vocabulary size must equal the base vocabulary size",
        "rule.vllm.shape": "A/B shapes and whole-module tensors must fit the base model",
        "rule.vllm.unknown-module": "module names outside the Llama/Mistral/Qwen/Gemma table cannot be confirmed",
        "rule.llamacpp.dora": "convert_lora_to_gguf.py stops on lora_magnitude_vector (DoRA)",
        "rule.llamacpp.unexpected-tensor": "tensors that are not lora_A / lora_B (whole modules, bias) stop the converter",
        "rule.llamacpp.shape": "adapter shapes must match the base model ('maybe wrong base model?')",
        "rule.llamacpp.howto": "how to convert and serve an adapter that passed",
        "rule.ollama.adapter-unsupported": "ADAPTER in a Modelfile fails with 'LoRA adapters are no longer supported'",
        # ---- shape problems
        "sp.vocab": "embedding LoRA size({a}) must be consistent with the base model's vocabulary size({b}) ({name})",
        "sp.emb_dim": "{name}: B has {rows} rows, base embedding dim is {dim}",
        "sp.lin_in": "{name}: A expects {a} inputs, base has {b}",
        "sp.lin_out": "{name}: B gives {a} outputs, base has {b}",
        "sp.full": "{name}: adapter shape {a} != base shape {b}",
        "pair.rank": "{name} (A and B ranks differ)",
        "pair.single": "{name} (only one of lora_A / lora_B present)",
        # ---- finding messages
        "msg.bin_pickle": "adapter_model.bin is a pickle file; tensors cannot be read safely without torch, so only adapter_config.json was checked.",
        "msg.truncated": "adapter_model.safetensors is truncated: the header requires {need} bytes but the file has {have}. Every runtime fails to load it.",
        "msg.truncated_ok": "file size matches what the header requires",
        "msg.pairing_bad": "broken LoRA pairs: {detail}",
        "msg.pairing_ok": "{n} module(s) have matching A/B tensors",
        "msg.unmatched": "target_modules entries with no tensor in the file: {names}. PEFT skips a target the model does not have without an error, so those layers were never trained.",
        "msg.unmatched_ok": "every target_modules entry has tensors",
        "msg.base_missing": "{n} adapter module(s) not found in the base, e.g. {example}. Is this the base model the adapter was trained on?",
        "msg.base_missing_ok": "all adapter modules exist in the base",
        "msg.rank_unknown": "no rank found in the config or tensors",
        "msg.rank_too_big": "LoRA rank {rank} is greater than max_lora_rank {max}, and {rank} is above the largest allowed value (512; allowed: {allowed}).",
        "msg.rank_over": "LoRA rank {rank} is greater than max_lora_rank {max}.",
        "msg.rank_ok": "max rank {rank} <= max_lora_rank {max}",
        "msg.dora": "vLLM does not yet support DoRA.",
        "msg.dora_ok": "no DoRA",
        "msg.bias": "Adapter bias is not supported (bias={bias}).",
        "msg.bias_ok": "bias is 'none'",
        "msg.lora_bias": "vLLM does not support LoRA bias (lora_bias).",
        "msg.lora_bias_ok": "lora_bias is off",
        "msg.mts": "vLLM only supports modules_to_save being either None or [\"classifier\", \"score\"]. Unsupported: {names}",
        "msg.mts_ok": "no unsupported modules_to_save",
        "msg.init": "vLLM does not support init_lora_weights={init}. Inits such as PiSSA, OLoRA, CorDA and LoftQ modify the base model weights.",
        "msg.init_ok": "init_lora_weights is supported",
        "msg.variants": "vLLM does not support: {names}.",
        "msg.variants_ok": "no unsupported adapter variant",
        "msg.embed_unknown": "the adapter has embedding LoRA tensors, but no base was given to compare vocabulary sizes.",
        "msg.joined": "{detail}",
        "msg.vocab_ok": "no vocabulary size conflict",
        "msg.shape_ok": "A/B shapes fit the base",
        "msg.unknown_module": "module names outside the Llama/Mistral/Qwen/Gemma table: {names}. vLLM accepts only the modules the served model class lists; this could not be confirmed.",
        "msg.unknown_module_ok": "all module names are in the known table",
        "msg.llama_unread": "tensors were not read (adapter_model.bin), so the converter outcome cannot be predicted.",
        "msg.llama_dora": "Unexpected name '{name}': Not a lora_A or lora_B tensor (DoRA magnitude vectors).",
        "msg.llama_dora_ok": "no DoRA magnitude tensors",
        "msg.llama_unexpected": "Unexpected name {names}: Not a lora_A or lora_B tensor.",
        "msg.llama_unexpected_ok": "only lora_A / lora_B tensors",
        "msg.llama_shape_unknown": "no base model to compare shapes with; a wrong base fails at load time.",
        "msg.llama_shape": "tensor has incorrect shape (hint: maybe wrong base model?): {detail}",
        "msg.llama_shape_ok": "adapter shapes fit the base",
        "msg.llama_howto": "convert to GGUF, then serve with --lora.",
        "msg.ollama": "A Modelfile ADAPTER line failed with 'Error: LoRA adapters are no longer supported' on Ollama 0.40.15 (checked 2026-11-10; the version where this started is unknown, and other versions may differ).",
        # ---- fixes
        "fix.resave": "Re-save as safetensors: model.save_pretrained(dir, safe_serialization=True)",
        "fix.truncated": "The copy or upload was cut off; re-save the adapter or copy/download it again.",
        "fix.pairing": "The file is incomplete or corrupt; re-run save_pretrained or copy the adapter again.",
        "fix.unmatched": "Check the module names with `print(model)` before training and fix target_modules.",
        "fix.base_missing": "Pass the matching base with --base <folder|hf:repo>.",
        "fix.rank_too_big": "No --max-lora-rank value fits. Retrain with r <= 512, or merge the adapter into the base model and serve the merged model.",
        "fix.merge_unload": "Merge the adapter into the base model (merge_and_unload) and serve the merged model.",
        "fix.merge_serve": "Merge the adapter into the base model and serve the merged model.",
        "fix.bias_retrain": "Retrain with bias='none', or merge the adapter into the base model and serve the merged model.",
        "fix.bias_config": "The file has no bias tensors, so only the config is wrong: set \"bias\": \"none\" in adapter_config.json (a copy is safest) and the adapter loads.",
        "fix.lora_bias": "Retrain with lora_bias=False, or merge the adapter and serve the merged model.",
        "fix.mts": "Drop the extra modules (you lose the changed embedding/head weights, e.g. new tokens), or merge the adapter into the base model and serve the merged model.",
        "fix.init": "Convert to a regular LoRA adapter with PEFT's path_initial_model_for_weight_conversion, or set init_lora_weights to true if the served base already contains the modified weights.",
        "fix.rerun_base": "Re-run with --base <folder|hf:repo>.",
        "fix.vocab": "The adapter was saved after the vocabulary grew (new tokens). Serve a base resized to the same vocabulary (save the resized base after training), or merge the adapter into that resized base.",
        "fix.shape": "This is probably not the base the adapter was trained on; pass the right --base.",
        "fix.unknown_module": "Try loading it once with vLLM; if it fails with 'Please verify that the loaded LoRA module is correct', retrain with target_modules limited to modules the model supports.",
        "fix.merge_convert": "Merge the adapter into the base model (merge_and_unload), then convert the merged model to GGUF.",
        "fix.llama_unexpected": "Remove whole-module tensors (modules_to_save such as embed_tokens / lm_head, bias) before converting, or merge the adapter into the base model and convert the merged model.",
        "fix.llama_shape": "Convert and serve with the exact base the adapter was trained on (vocabulary included).",
        "fix.merge_hint": "Merge the adapter into the base (`merge_and_unload()`), save it, and serve the merged model. Example:\n{cmd}",
        # ---- errors (also used as PreflightError text)
        "err.not_folder": "{root} is not a folder. Pass the folder that contains adapter_config.json, or use hf:<owner>/<repo> for a Hub repo.",
        "err.not_exist": "{path} does not exist.",
        "err.bad_repo": "'{repo}' is not a Hub repo id. Use hf:<owner>/<repo>, e.g. hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0.",
        "err.bad_rev": "'{revision}' is not a valid revision name.",
        "err.http404": "{where}: {name} not found (HTTP 404).",
        "err.http_auth": "{where}: HTTP {code} for {name}. The repo is private or gated; set the HF_TOKEN environment variable to a token that can read it.",
        "err.http": "{where}: HTTP {code} for {name}.",
        "err.network": "{where}: network error while reading {name} ({exc}). Check your connection, or point lora-preflight at a local folder.",
        "err.no_config": "{where} has no {file}. Point lora-preflight at the folder that `model.save_pretrained(...)` wrote.",
        "err.config_json": "{path} is not valid JSON ({exc}). Re-save the adapter.",
        "err.config_obj": "{path} must be a JSON object.",
        "err.peft_type": "{path} has peft_type={peft_type}; lora-preflight only checks LORA adapters.",
        "err.no_weights": "{where} has neither {st} nor {bin}. Point lora-preflight at the folder that holds the adapter weights.",
        "err.base_empty": "{path} has an empty weight_map.",
        "err.base_json": "{path} is not valid JSON ({exc}).",
        "err.base_nofile": "{where} has no {name}. The base must be safetensors ({single} or {index} with shards).",
        "err.base_truncated": "Base file is truncated: {path} has {have} bytes but its header requires {need}. Copy or download the base model again.",
        "err.target_unknown": "unknown target '{raw}'. Choose from: {known}, or 'all'.",
        "err.target_none": "no target given. Choose from: {known}, or 'all'.",
        "err.max_rank": "max_lora_rank {rank} is not a value vLLM accepts. Use one of: {allowed}.",
        "err.hdr_short": "{label}: file is shorter than 8 bytes, so it is not a safetensors file.",
        "err.hdr_len": "{label}: header length {length} is larger than {max} bytes. The file is truncated, corrupt, or not a safetensors file (for example a pickle .bin or a Git LFS pointer).",
        "err.hdr_trunc": "{label}: header says {length} bytes but only {have} are available. The file is truncated; copy or download it again.",
        "err.hdr_json": "{label}: header is not valid JSON ({exc}).",
        "err.hdr_obj": "{label}: header JSON must be an object.",
        "err.hdr_tensor": "{label}: tensor {name} has no valid dtype/shape entry.",
        # ---- argparse built-ins (keys are argparse's own message ids)
        "argparse.usage: ": "usage: ",
        "argparse.options": "options",
        "argparse.optional arguments": "options",
        "argparse.positional arguments": "positional arguments",
        "argparse.show this help message and exit": "show this help message and exit",
    },
    "ko": {
        # ---- command line
        "usage.text": (
            "사용법:\n"
            "  lora-preflight <adapter> [--base <folder|hf:repo>] [--target vllm,llamacpp,ollama|all]\n"
            "                 [--max-lora-rank N] [--json] [--strict] [--lang en|ko]\n"
            "  lora-preflight demo [--keep <dir>] [--lang en|ko]\n"
            "  lora-preflight rules [--json] [--lang en|ko]\n"
            "  lora-preflight --version\n"
            "\n"
            "<adapter> 는 폴더(adapter_config.json + adapter_model.safetensors) 또는 hf:<owner>/<repo>[@revision] 입니다.\n"
            "종료 코드: 0 FAIL 없음, 1 FAIL 있음(--strict 에서는 WARN 포함), 2 잘못된 입력.\n"
            "화면 언어: --lang en|ko, 또는 LORA_PREFLIGHT_LANG, 또는 LC_ALL / LC_MESSAGES / LANG (ko 로 시작하지 않으면 영어).\n"
        ),
        "cli.error": "오류: {message}",
        "cli.bad_lang": "--lang 값이 올바르지 않습니다: {value}. en 또는 ko 를 고르세요.",
        "arg.check.desc": "PEFT LoRA 어댑터를 서빙하기 전에 vLLM, llama.cpp, Ollama 기준으로 점검합니다.",
        "arg.check.epilog": "다른 명령: `lora-preflight demo`, `lora-preflight rules`.",
        "arg.adapter": "어댑터 폴더 또는 hf:<owner>/<repo>[@revision]",
        "arg.base": "베이스 모델 폴더 또는 hf:<owner>/<repo> (기본값: 어댑터에 적힌 로컬 베이스 경로가 있으면 그것)",
        "arg.target": "vllm,llamacpp,ollama 중 쉼표로 나열 (기본값: all)",
        "arg.max_rank": "vLLM 에 줄 예정인 --max-lora-rank 값 (기본값 16)",
        "arg.json": "보고서를 JSON 으로 출력합니다",
        "arg.strict": "FAIL 뿐 아니라 WARN 에서도 종료 코드 1 을 냅니다",
        "arg.lang": "화면 언어: en 또는 ko (기본값: LORA_PREFLIGHT_LANG, 그다음 LC_ALL / LC_MESSAGES / LANG)",
        "arg.demo.desc": "합성 연습용 어댑터 7개를 점검합니다.",
        "arg.keep": "연습용 파일을 DIR 에 쓰고 남깁니다",
        "arg.rules.desc": "모든 규칙과 그 출처를 보여 줍니다.",
        "arg.rules_json": "규칙 표를 JSON 으로 출력합니다",
        # ---- demo / rules output
        "demo.banner": (
            "연습 모드: 합성 LoRA 어댑터입니다(가중치는 0으로 채웠고, 텐서 이름과 모양은 PEFT 0.21.2 가 저장한 형태입니다). "
            "실제 학습 결과가 아닙니다."
        ),
        "demo.col_adapter": "어댑터",
        "demo.col_first": "첫 문제",
        "demo.fixes": "vLLM 수정 방법",
        "demo.details": "어댑터 하나의 자세한 결과: lora-preflight demo --keep ./demo 실행 후 lora-preflight ./demo/dora --base ./demo/base",
        "rules.source": "    출처: {url}",
        "rules.basis": "    근거:  {ref}; 확인일 {date}",
        # ---- text report
        "rep.header_adapter": "어댑터: {value}",
        "rep.header_base": "베이스:  {value}",
        "rep.base_none": "(비교하지 않음. --base <folder|hf:repo> 를 지정하세요)",
        "rep.config": "설정:    {value}",
        "rep.bin": "텐서:    adapter_model.bin 은 pickle 파일이라 텐서를 읽지 않았습니다(설정만 점검)",
        "rep.ranks": "랭크:    {hist}  (설정값 r={r}, rank_pattern={pattern})",
        "rep.scale": "스케일:  {value}",
        "rep.modules": "모듈:    {n}개 ({types}); 레이어 {layers}",
        "rep.extra": "추가:    {value}",
        "rep.extra_full": "모듈 전체 텐서 {n}개",
        "rep.extra_magnitude": "DoRA magnitude 텐서 {n}개",
        "rep.extra_bias": "bias 텐서 {n}개",
        "rep.none": "없음",
        "rep.na": "해당 없음",
        "rep.layers_of": "{n}/{total} ({span})",
        "rep.layers": "{n}개 ({span})",
        "rep.verdicts": "판정",
        "rep.findings": "점검 결과 (확인할 것 {shown}건, 통과한 규칙 {passed}개)",
        "rep.fix_first": "      수정: {line}",
        "rep.how": "  [방법] {rule_id}: {message}",
        "rep.footer": (
            "이 판정은 파일만 읽고 규칙으로 예측한 것이며 실제 런타임은 실행하지 않았습니다. "
            "규칙 출처와 확인일: `lora-preflight rules`."
        ),
        # ---- rule titles
        "rule.common.weights-format": "adapter_model.bin 은 torch 없이 읽을 수 없어 설정만 점검합니다",
        "rule.common.truncated": "adapter_model.safetensors 는 헤더가 요구하는 크기(8 + 헤더 길이 + 마지막 data_offsets 끝) 이상이어야 합니다",
        "rule.common.pairing": "모든 LoRA 모듈에는 같은 랭크의 A, B 텐서 쌍이 있어야 합니다",
        "rule.common.unmatched-target": "어떤 텐서와도 맞지 않는 target_modules 항목은 학습에서 조용히 건너뛰어졌습니다",
        "rule.common.base-module-missing": "베이스 모델에 없는 어댑터 모듈",
        "rule.vllm.rank": "최대 랭크(r 과 rank_pattern)는 --max-lora-rank 이하여야 합니다 (기본값 16, 허용값 1,8,16,32,64,128,256,320,512)",
        "rule.vllm.dora": "DoRA 는 지원하지 않습니다",
        "rule.vllm.bias": "bias 가 'none' 이 아니면 지원하지 않습니다",
        "rule.vllm.lora-bias": "lora_bias 는 지원하지 않습니다",
        "rule.vllm.modules-to-save": "modules_to_save 와 모듈 전체 텐서는 classifier, score 만 허용됩니다",
        "rule.vllm.init-weights": "PiSSA, OLoRA, CorDA, LoftQ 같은 init_lora_weights 는 지원하지 않습니다",
        "rule.vllm.variants": "aLoRA, layer_replication, BD-LoRA, QALoRA 는 지원하지 않습니다",
        "rule.vllm.embedding-vocab": "임베딩 LoRA 의 어휘 크기는 베이스의 어휘 크기와 같아야 합니다",
        "rule.vllm.shape": "A/B 모양과 모듈 전체 텐서는 베이스 모델에 맞아야 합니다",
        "rule.vllm.unknown-module": "Llama/Mistral/Qwen/Gemma 표에 없는 모듈 이름은 확인할 수 없습니다",
        "rule.llamacpp.dora": "convert_lora_to_gguf.py 는 lora_magnitude_vector(DoRA)에서 멈춥니다",
        "rule.llamacpp.unexpected-tensor": "lora_A / lora_B 가 아닌 텐서(모듈 전체, bias)는 변환기를 멈추게 합니다",
        "rule.llamacpp.shape": "어댑터 모양은 베이스 모델과 맞아야 합니다 ('maybe wrong base model?')",
        "rule.llamacpp.howto": "통과한 어댑터를 변환하고 서빙하는 방법",
        "rule.ollama.adapter-unsupported": "Modelfile 의 ADAPTER 는 'LoRA adapters are no longer supported' 로 실패합니다",
        # ---- shape problems
        "sp.vocab": "임베딩 LoRA 크기({a})가 베이스 모델의 어휘 크기({b})와 같아야 합니다 ({name})",
        "sp.emb_dim": "{name}: B 의 행 수 {rows}, 베이스 임베딩 차원 {dim}",
        "sp.lin_in": "{name}: A 의 입력 {a}, 베이스의 입력 {b}",
        "sp.lin_out": "{name}: B 의 출력 {a}, 베이스의 출력 {b}",
        "sp.full": "{name}: 어댑터 모양 {a}, 베이스 모양 {b}",
        "pair.rank": "{name} (A 와 B 의 랭크가 다름)",
        "pair.single": "{name} (lora_A / lora_B 중 하나만 있음)",
        # ---- finding messages
        "msg.bin_pickle": "adapter_model.bin 은 pickle 파일이라 torch 없이는 텐서를 안전하게 읽을 수 없어 adapter_config.json 만 점검했습니다.",
        "msg.truncated": "adapter_model.safetensors 가 잘렸습니다. 헤더가 요구하는 크기는 {need} 바이트인데 파일은 {have} 바이트입니다. 어떤 런타임도 로드하지 못합니다.",
        "msg.truncated_ok": "파일 크기가 헤더의 요구와 일치합니다",
        "msg.pairing_bad": "LoRA 쌍이 깨졌습니다: {detail}",
        "msg.pairing_ok": "모듈 {n}개 모두 A/B 텐서 쌍이 맞습니다",
        "msg.unmatched": "파일에 텐서가 없는 target_modules 항목: {names}. PEFT 는 모델에 없는 대상을 오류 없이 건너뛰므로 해당 레이어는 학습되지 않았습니다.",
        "msg.unmatched_ok": "모든 target_modules 항목에 텐서가 있습니다",
        "msg.base_missing": "베이스에서 찾지 못한 어댑터 모듈 {n}개, 예: {example}. 어댑터를 학습한 베이스 모델이 맞습니까?",
        "msg.base_missing_ok": "모든 어댑터 모듈이 베이스에 있습니다",
        "msg.rank_unknown": "설정과 텐서에서 랭크를 찾지 못했습니다",
        "msg.rank_too_big": "LoRA 랭크가 max_lora_rank 보다 크고, 허용되는 최대값(512)도 넘습니다: {rank} > {max} (허용값: {allowed}).",
        "msg.rank_over": "LoRA 랭크가 max_lora_rank 보다 큽니다: {rank} > {max}.",
        "msg.rank_ok": "최대 랭크 {rank} <= max_lora_rank {max}",
        "msg.dora": "vLLM 은 아직 DoRA 를 지원하지 않습니다.",
        "msg.dora_ok": "DoRA 없음",
        "msg.bias": "어댑터 bias 는 지원하지 않습니다 (bias={bias}).",
        "msg.bias_ok": "bias 가 'none' 입니다",
        "msg.lora_bias": "vLLM 은 LoRA bias(lora_bias)를 지원하지 않습니다.",
        "msg.lora_bias_ok": "lora_bias 가 꺼져 있습니다",
        "msg.mts": "vLLM 은 modules_to_save 가 None 이거나 [\"classifier\", \"score\"] 일 때만 지원합니다. 지원되지 않는 항목: {names}",
        "msg.mts_ok": "지원되지 않는 modules_to_save 가 없습니다",
        "msg.init": "vLLM 은 이 init_lora_weights 값을 지원하지 않습니다 (init_lora_weights={init}). PiSSA, OLoRA, CorDA, LoftQ 같은 초기화는 베이스 모델 가중치를 바꿉니다.",
        "msg.init_ok": "init_lora_weights 가 지원됩니다",
        "msg.variants": "vLLM 이 지원하지 않는 변형: {names}.",
        "msg.variants_ok": "지원되지 않는 어댑터 변형이 없습니다",
        "msg.embed_unknown": "어댑터에 임베딩 LoRA 텐서가 있지만 어휘 크기를 비교할 베이스가 지정되지 않았습니다.",
        "msg.joined": "{detail}",
        "msg.vocab_ok": "어휘 크기 충돌이 없습니다",
        "msg.shape_ok": "A/B 모양이 베이스에 맞습니다",
        "msg.unknown_module": "Llama/Mistral/Qwen/Gemma 표에 없는 모듈 이름: {names}. vLLM 은 서빙하는 모델 클래스가 나열한 모듈만 받으므로 확인할 수 없었습니다.",
        "msg.unknown_module_ok": "모든 모듈 이름이 알려진 표에 있습니다",
        "msg.llama_unread": "텐서를 읽지 않았으므로(adapter_model.bin) 변환기 결과를 예측할 수 없습니다.",
        "msg.llama_dora": "예상 밖 이름 '{name}': lora_A 또는 lora_B 텐서가 아닙니다(DoRA magnitude 벡터).",
        "msg.llama_dora_ok": "DoRA magnitude 텐서가 없습니다",
        "msg.llama_unexpected": "예상 밖 이름 {names}: lora_A 또는 lora_B 텐서가 아닙니다.",
        "msg.llama_unexpected_ok": "lora_A / lora_B 텐서만 있습니다",
        "msg.llama_shape_unknown": "모양을 비교할 베이스 모델이 없습니다. 베이스가 틀리면 로드할 때 실패합니다.",
        "msg.llama_shape": "텐서 모양이 맞지 않습니다 (힌트: 베이스 모델이 다를 수 있습니다): {detail}",
        "msg.llama_shape_ok": "어댑터 모양이 베이스에 맞습니다",
        "msg.llama_howto": "GGUF 로 변환한 뒤 --lora 로 서빙합니다.",
        "msg.ollama": "Ollama 0.40.15 에서 Modelfile 의 ADAPTER 줄이 'Error: LoRA adapters are no longer supported' 로 실패했습니다 (2026-11-10 확인. 언제부터인지는 알 수 없고 다른 버전은 다를 수 있습니다).",
        # ---- fixes
        "fix.resave": "safetensors 로 다시 저장하세요: model.save_pretrained(dir, safe_serialization=True)",
        "fix.truncated": "복사나 업로드가 중간에 끊겼습니다. 어댑터를 다시 저장하거나 다시 복사/다운로드하세요.",
        "fix.pairing": "파일이 불완전하거나 손상됐습니다. save_pretrained 를 다시 실행하거나 어댑터를 다시 복사하세요.",
        "fix.unmatched": "학습 전에 `print(model)` 로 모듈 이름을 확인하고 target_modules 를 고치세요.",
        "fix.base_missing": "맞는 베이스를 --base <folder|hf:repo> 로 지정하세요.",
        "fix.rank_too_big": "맞는 --max-lora-rank 값이 없습니다. r <= 512 로 다시 학습하거나, 어댑터를 베이스 모델에 병합해 병합 모델을 서빙하세요.",
        "fix.merge_unload": "어댑터를 베이스 모델에 병합(merge_and_unload)하고 병합한 모델을 서빙하세요.",
        "fix.merge_serve": "어댑터를 베이스 모델에 병합하고 병합한 모델을 서빙하세요.",
        "fix.bias_retrain": "bias='none' 으로 다시 학습하거나, 어댑터를 베이스 모델에 병합해 병합 모델을 서빙하세요.",
        "fix.bias_config": "파일에 bias 텐서가 없으므로 설정만 틀렸습니다. adapter_config.json 에서 \"bias\": \"none\" 으로 바꾸면(복사본이 안전합니다) 어댑터가 로드됩니다.",
        "fix.lora_bias": "lora_bias=False 로 다시 학습하거나, 어댑터를 병합해 병합 모델을 서빙하세요.",
        "fix.mts": "추가 모듈을 빼거나(바뀐 임베딩/헤드 가중치, 예를 들어 새 토큰을 잃습니다), 어댑터를 베이스 모델에 병합해 병합 모델을 서빙하세요.",
        "fix.init": "PEFT 의 path_initial_model_for_weight_conversion 으로 일반 LoRA 어댑터로 변환하세요. 서빙하는 베이스에 이미 바뀐 가중치가 들어 있다면 init_lora_weights 를 true 로 두세요.",
        "fix.rerun_base": "--base <folder|hf:repo> 를 붙여 다시 실행하세요.",
        "fix.vocab": "어휘가 늘어난(새 토큰) 뒤에 어댑터를 저장했습니다. 같은 어휘로 크기를 바꾼 베이스(학습 후 저장한 것)를 서빙하거나, 그 베이스에 어댑터를 병합하세요.",
        "fix.shape": "어댑터를 학습한 베이스가 아닐 가능성이 큽니다. 맞는 --base 를 지정하세요.",
        "fix.unknown_module": "vLLM 으로 한 번 로드해 보세요. 'Please verify that the loaded LoRA module is correct' 로 실패하면 모델이 지원하는 모듈만 target_modules 에 넣고 다시 학습하세요.",
        "fix.merge_convert": "어댑터를 베이스 모델에 병합(merge_and_unload)한 뒤, 병합한 모델을 GGUF 로 변환하세요.",
        "fix.llama_unexpected": "변환 전에 모듈 전체 텐서(embed_tokens / lm_head 같은 modules_to_save, bias)를 제거하거나, 어댑터를 베이스 모델에 병합해 병합 모델을 변환하세요.",
        "fix.llama_shape": "어댑터를 학습한 정확한 베이스(어휘 포함)로 변환하고 서빙하세요.",
        "fix.merge_hint": "어댑터를 베이스에 병합(`merge_and_unload()`)하고 저장한 뒤 병합한 모델을 서빙하세요. 예시:\n{cmd}",
        # ---- errors
        "err.not_folder": "폴더가 아닙니다: {root}. adapter_config.json 이 있는 폴더를 지정하거나, Hub 저장소는 hf:<owner>/<repo> 를 쓰세요.",
        "err.not_exist": "파일 없음: {path}",
        "err.bad_repo": "Hub 저장소 ID 가 아닙니다: '{repo}'. hf:<owner>/<repo> 형식으로 쓰세요. 예: hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0.",
        "err.bad_rev": "올바르지 않은 리비전 이름: '{revision}'",
        "err.http404": "{where}: {name} 파일을 찾을 수 없습니다 (HTTP 404).",
        "err.http_auth": "{where}: {name} 요청이 거부됐습니다 (HTTP {code}). 비공개이거나 접근 제한된 저장소입니다. 읽을 수 있는 토큰을 HF_TOKEN 환경변수에 지정하세요.",
        "err.http": "{where}: {name} 요청이 실패했습니다 (HTTP {code}).",
        "err.network": "{where}: 파일을 읽는 중 네트워크 오류가 났습니다 (파일: {name}, 원인: {exc}). 연결을 확인하거나 로컬 폴더를 지정하세요.",
        "err.no_config": "{where} 에 다음 파일이 없습니다: {file}. `model.save_pretrained(...)` 가 쓴 폴더를 지정하세요.",
        "err.config_json": "올바른 JSON 이 아닙니다: {path} ({exc}). 어댑터를 다시 저장하세요.",
        "err.config_obj": "JSON 객체여야 합니다: {path}",
        "err.peft_type": "{path} 의 peft_type={peft_type}. lora-preflight 는 LORA 어댑터만 점검합니다.",
        "err.no_weights": "{where} 에 {st} 도 {bin} 도 없습니다. 어댑터 가중치가 있는 폴더를 지정하세요.",
        "err.base_empty": "weight_map 이 비어 있습니다: {path}",
        "err.base_json": "올바른 JSON 이 아닙니다: {path} ({exc}).",
        "err.base_nofile": "{where} 에 다음 파일이 없습니다: {name}. 베이스는 safetensors 여야 합니다({single} 또는 샤드가 있는 {index}).",
        "err.base_truncated": "베이스 파일이 잘렸습니다: {path} (실제 {have} 바이트, 헤더가 요구하는 크기 {need} 바이트). 베이스 모델을 다시 복사하거나 다운로드하세요.",
        "err.target_unknown": "알 수 없는 target: '{raw}'. {known}, 'all' 중에서 고르세요.",
        "err.target_none": "target 이 지정되지 않았습니다. {known}, 'all' 중에서 고르세요.",
        "err.max_rank": "vLLM 이 받지 않는 max_lora_rank 값: {rank}. 다음 중 하나를 쓰세요: {allowed}.",
        "err.hdr_short": "{label}: 파일이 8바이트보다 짧아 safetensors 파일이 아닙니다.",
        "err.hdr_len": "{label}: 헤더 길이({length})가 {max} 바이트보다 큽니다. 파일이 잘렸거나 손상됐거나 safetensors 파일이 아닙니다(예: pickle .bin, Git LFS 포인터).",
        "err.hdr_trunc": "{label}: 헤더는 {length} 바이트라고 하는데 {have} 바이트만 읽을 수 있습니다. 파일이 잘렸으니 다시 복사하거나 다운로드하세요.",
        "err.hdr_json": "{label}: 헤더가 올바른 JSON 이 아닙니다 ({exc}).",
        "err.hdr_obj": "{label}: 헤더 JSON 은 객체여야 합니다.",
        "err.hdr_tensor": "{label}: 텐서 {name} 의 dtype/shape 항목이 올바르지 않습니다.",
        # ---- argparse built-ins
        "argparse.usage: ": "사용법: ",
        "argparse.options": "옵션",
        "argparse.optional arguments": "옵션",
        "argparse.positional arguments": "위치 인자",
        "argparse.show this help message and exit": "이 도움말을 보여 주고 종료합니다",
        "argparse.the following arguments are required: %s": "다음 인자가 필요합니다: %s",
        "argparse.unrecognized arguments: %s": "인식할 수 없는 인자: %s",
        "argparse.argument %(argument_name)s: %(message)s": "인자 %(argument_name)s: %(message)s",
        "argparse.expected one argument":"값이 하나 필요합니다",
        "argparse.invalid %(type)s value: %(value)r": "%(type)s 값이 올바르지 않습니다: %(value)r",
        "argparse.%(prog)s: error: %(message)s\n": "%(prog)s: 오류: %(message)s\n",
        "argparse.ambiguous option: %(option)s could match %(matches)s": "모호한 옵션: %(option)s 은 %(matches)s 와 맞을 수 있습니다",
    },
}

# argparse prints these through gettext; we map them for the shown language.
_ARGPARSE_PREFIX = "argparse."
for _key in MESSAGES["ko"]:
    if _key.startswith(_ARGPARSE_PREFIX):  # English keeps argparse's own wording
        MESSAGES["en"].setdefault(_key, _key[len(_ARGPARSE_PREFIX):])
_current = DEFAULT_LANG


def _norm(value: str | None) -> str | None:
    """'ko_KR.UTF-8' -> 'ko'; anything else that is set -> 'en'; unset/empty -> None."""
    if value is None or not value.strip():
        return None
    return "ko" if value.strip().lower().startswith("ko") else "en"


def resolve_lang(argv_lang: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Pick the screen language: --lang, then LORA_PREFLIGHT_LANG, LC_ALL, LC_MESSAGES, LANG, else en."""
    environ = os.environ if env is None else env
    if argv_lang is not None:
        return _norm(argv_lang) or DEFAULT_LANG
    for name in (ENV_VAR, "LC_ALL", "LC_MESSAGES", "LANG"):
        picked = _norm(environ.get(name))
        if picked:
            return picked
    return DEFAULT_LANG


def set_lang(lang: str) -> None:
    """Set the language used by :func:`t`."""
    global _current
    if lang not in LANGS:
        raise ValueError(f"unsupported language {lang!r}; choose from {', '.join(LANGS)}")
    _current = lang


def get_lang() -> str:
    return _current


def _value(lang: str, v: Any) -> Any:
    if isinstance(v, M):
        return tr(lang, v.key, **v.params)
    if isinstance(v, J):
        return v.sep.join(str(_value(lang, i)) for i in v.items)
    return v


def tr(lang: str, key: str, **params: Any) -> str:
    """Message ``key`` in ``lang`` with ``params`` filled in (nested M / J resolved in ``lang``)."""
    template = MESSAGES[lang][key]
    return template.format(**{k: _value(lang, v) for k, v in params.items()})


def t(key: str, **params: Any) -> str:
    """Message ``key`` in the current language."""
    return tr(_current, key, **params)


def argparse_gettext(message: str) -> str:
    """Replacement for argparse's gettext: translate its built-in phrases in the current language."""
    return MESSAGES[_current].get(_ARGPARSE_PREFIX + message, message)
