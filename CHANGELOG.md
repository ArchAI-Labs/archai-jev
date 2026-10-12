# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Kev-0.8B** (`jaredpalmer/kev-0.8b`) is a model of the registry (variants `bf16`, the default,
  and `f32`). It is published as a base model plus a LoRA adapter, so the library downloads the
  original files, checks them by SHA-256, converts them into a GGUF with a converter written in
  Rust (merge of the LoRA, the transformations of the Qwen3.5 family, the vocabulary), keeps the
  result in the cache and loads it. No hosted weights, no Python or PyTorch at runtime.
  The converter output equals the one of llama.cpp's official script tensor by tensor (the
  weights a LoRA adapted, the `bf16` and `f32` files: identical to the byte; only 13 of the 288
  values of `A_log` differ by one unit in the last place, because PyTorch's `exp` is not
  correctly rounded). See `docs/model-files.md` for where the files go and how big they are.
- A reader of `head.pt` files that never executes the pickle: an allow-list of three globals and
  of opcodes, limits on every size, an `IncompatibleModelError` naming the opcode or the global
  and its offset for anything else.
- The converter refuses, instead of guessing, a base model with a number of MTP layers other
  than one and any number of its config that does not fit the 32 bits of a GGUF integer.
- A verification record also for converted models: the second `from_pretrained` neither converts
  nor checks again; a converted file that no longer matches its record is rebuilt once from the
  verified originals.
- `demos/`: runnable, heavily commented scripts that need only the standard library and
  `archai_jev` (they are not part of the wheel). `000_truth_invaders.py` is a retro arcade game
  played by the model; `001` to `007` teach one concept each: the four tasks and typed answers,
  a grounding check, a duplicate finder, every error the library raises on purpose, calibration
  and `temperature=`, async (`aask_many`), and testing your own code with `MockScorer`.
- **Reuse of what questions share.** The fixed block of a task and, when the template puts it
  first, the state are computed once and kept in memory: inside one `ask` and between calls, per
  loaded model, limited in bytes (least recently used out first). `Jev.from_pretrained(prefix_cache_mib=...)`
  (default 128, environment variable `ARCHAI_JEV_PREFIX_CACHE_MIB`, `0` turns it off),
  `Jev.cache_info()` returning a `CacheInfo`, `Jev.clear_cache()`. See `docs/prefix-cache.md`.
  The key of a saved prefix includes the model revision, the GGUF hash, the template id and
  version, the tokenizer hash and the engine settings; a prefix that cannot be restored raises
  `InferenceError` and is dropped. No new dependency; the wheel grows by less than 1 %.
- **`Ctrl-C` stops a running call.** `ask` and `ask_many` look for an interrupt between two rows of
  the model (main thread, as Python delivers signals), stop with no partial result and leave the
  `Jev` usable. With the default model the `KeyboardInterrupt` arrived about 30 ms after the signal
  (before: when the whole call ended). A signal handler or a destructor that runs inside a call and
  calls `ask` raises `RuntimeError` instead of waiting for itself for ever.
- **Cancelling `aask`/`aask_many` stops the work.** A request that has not started never starts; one
  that is running stops at its next row (before, the result was thrown away and the model kept
  computing). They run on a four-thread executor of their own per `Jev` (`archai-jev-*` threads)
  instead of the event loop's default executor, so a busy default executor no longer slows them
  and a flood of `aask` no longer slows the program's `to_thread` calls.
- `ask_many(states, questions, *, return_exceptions=False)` and `aask_many(...)`: with `True` a
  state whose calculation fails gives its error in its place (`list[Answers | JevError]`, typed with
  `@overload`); invalid input still raises before any calculation. See `docs/concurrency.md`.
- A pool of inference contexts that share the weights exists in the Rust core (`SlotPool`,
  `LlamaBackend::with_max_concurrency`; not exposed in Python): bit-identical answers for 1, 2, 4
  and 8 contexts, and the rule that the threads of the running contexts never add up to more than the
  budget (llama.cpp's own thread pool stalls when they do). Measured gain on a 20-thread CPU: about
  1.1 to 1.2 times, which is why Python keeps one request at a time.

### Changed
- Prompts of the default model are now computed in segments (at the end of the task block and of
  the state) instead of in one pass. Answers stay within the tolerances of the model: on the golden
  prompts the largest difference from 0.1.0 was 0.48 on a raw logit and 0.033 on a probability,
  and no top answer changed. With the memory empty, warm or cleared the answers are identical bit
  for bit. `prefix_cache_mib=0` restores the 0.1.0 computation. A request with a single question
  about a text of a task never seen costs a little more than before (the prefix is saved).

## [0.1.0] - 2026-10-06

### Added
- `Jev.from_pretrained(...)`: loads the default demo model, a model of the registry or a local
  folder with a manifest; every file is verified (SHA-256), the weights, tokenizer and head are
  checked, and a self-check of the model's numbers runs on first use. `list_models()`.
- The default model `nickprock/archai-jev-qwen-1.5b` (Q8_0, about 1.6 GB, downloaded on first use),
  with four declared tasks (safety, intent, entailment, similarity) and a declared calibration
  (temperature 2.09, measured in-distribution only). Any other question raises
  `UnsupportedRequestError` listing the tasks.
- Inference engine on llama.cpp (CPU): the prompt is built and tokenized by the library (identical
  ids to the training formatters and to Kev's prompt: parity tested), the state is decoded once
  per request and each question runs on a copy of it, and the answer is read at the last token.
- New exceptions: `InferenceError`, and `UnsupportedRequestError` now has its real emissions: a
  prompt longer than the model's context, a question the model cannot answer, options that read the
  same to the model. Nothing is ever truncated.
- `ModelInfo` gains `dtype`, `source`, `calibration_source`, `tasks`, `notice`, `is_default`.
- `calibrated_softmax(logits, temperature=1.0)`: temperature-scaled softmax computed in Rust.
- Single abi3 wheel per platform supporting Python 3.10 and newer.

### Changed
- `calibrated_softmax` now rejects a non-finite `temperature` (`inf`) with `ValueError`.
