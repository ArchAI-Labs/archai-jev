# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **General Choice/yes-no models in the Decision Index format** (spec 010): a new family of models,
  `qwen2-keys`, answers any `Choice` or `YesNo` instead of a fixed set of tasks. The prompt is the
  one of the Decision Index `transformers` engine (the benchmark suite of Jev models) inside the
  ChatML chat template of Qwen2.5, and the score of an option is the sum of the log-probabilities
  of the tokens of its key (one token or several), normalised over the whole vocabulary, then a
  softmax across the options. A manifest declares `head.kind = "key_logprob"` with its limits
  (`max_options`, `max_key_tokens`) and `template = decision-index-chatml`; it has no closed set of
  tasks (`tasks: null`). A request the model cannot answer is refused before any computation, with
  the limit and the value received: `Score` questions, more options than the model scores, an
  option key of too many tokens (or none), two keys that are the same tokens, a key that starts
  another key (its score would always be at most the other's, so it could never be chosen: our
  measurement of the right answer on such a key was 0 of 95), a prompt that with its longest key
  does not fit the context. The prompt ids and the key ids are identical to the
  reference engine's on the 87 questions of its golden, and the engine's scores are within 0.003
  of its probabilities on the fp16 weights (0.06 on the published Q8_0). Questions about the same
  state share the system block and the state through the cache of prefixes.
- **See what a model accepts, and run the retrained model (v2)**: the registry now has a
  second revision of `nickprock/archai-jev-qwen-1.5b` (`revision="7730248d"`), the Decision Index
  model, with a declared calibration for its Q8_0 (temperature 1.7497, ECE 0.062 -> 0.009 on
  held-out data of its training tasks; it stays overconfident on tasks it did not see, which the
  `evidence` says). `ModelInfo.max_options` and
  `max_key_tokens` tell how many options a `Choice` may have and how long an option key may be
  (`None` when the model has no limit of its own); a request over a limit is refused with
  `UnsupportedRequestError`, as before. `list_models()` shows one entry per model name, at its
  current revision.
- **See which weights a model offers**: `ModelInfo.variants` (a tuple of the new `VariantInfo`:
  `dtype`, `size_bytes`, `calibrated`, `temperature`, `is_default`), from `list_models()` and from
  a loaded model, with no network. Each variant carries **its own** calibration: a quantized
  variant never inherits the temperature of another. When a variant has no declared calibration
  the error now says so and points to `Jev.calibrate()`; if another variant of the same model
  declares one, it says that temperature does not transfer. An unknown `dtype` lists the
  supported names (`"int8"` is pointed to `"q8_0"`), and a weights file whose tensor types do not
  fit its declared `dtype` is refused with the tensor, the expected type and the type found.
- **Weights formats and quantization guide** (`docs/quantization.md`): which `dtype` names a release accepts
  (`f32`, `bf16`, `q8_0`; `float32` and `bfloat16` as synonyms), why a variant enters the registry only after
  a published gate, and the result for the 4-bit candidate. The Q4_K_M of the default model passed the quality
  and size criteria (file 0.60 times the Q8_0, accuracy within 0.4 points) but its self-check numbers differ
  by up to 0.14 in probability between CPUs, so it is **not** offered; the K-quant names are refused with the
  list of valid names. The self-check of the default model was measured on Linux (x86-64 and ARM), macOS
  (Intel and Apple Silicon) and Windows: the largest difference from the reference engine is 0.04, against
  a tolerance of 0.1.
- **Measure and calibrate a model on your own data** (`archai_jev.evaluation`, plus
  `InvalidDatasetError`), all in Rust and with no new dependency:
  - `load_dataset(path)` reads the JSONL format of Kev (the TypeSafe request plus a `label` per
    question, optional `id` and `group`); `LabelledRecord` or `dict` records can be passed directly.
    Labels are exact, unknown fields are an error with the list of names, errors name the file
    and line, integers beyond 64 bits in a state are kept, and nothing is skipped.
  - `Jev.evaluate(dataset)` gives accuracy, ECE, Brier, log-loss and the reliability bins per
    kind of question and per question name, expected-level MAE, positive rate, optional 95 %
    bootstrap intervals (whole groups are resampled), and factual notes (overconfidence, small
    samples, no declared calibration, records that were also used to calibrate). All or nothing.
  - `Jev.calibrate(dataset)` fits one global temperature (absolute) and measures what it gains
    with a temperature fitted on the other folds (folds are made of whole groups);
    `Jev.with_temperature(...)` returns a **new** `Jev` that shares the model and says
    `calibrated=True`; `split_dataset` splits by group, deterministically.
  - `compare({label: jev}, dataset)` evaluates several models on the same data and fails, naming
    the model, if one cannot answer a record. No ranking.
  - Reports have `table()` (fixed columns, three decimals) and `to_dict()`.
- **Calibration tools** (`archai_jev.calibration`, plus `fit_temperature`, `TemperatureFit` and
  `CalibrationError` in `archai_jev`):
  - `fit_temperature(logits, labels, *, bounds=(0.01, 100.0))` finds the temperature that
    minimises the log-loss as the zero of a monotone derivative (the loss is convex in `1/T`),
    not a grid. When the best temperature is not inside `bounds` (every row uniform, labels that
    always match the top option, logits without information about the labels) it raises
    `CalibrationError` and returns nothing: a temperature at the edge would look calibrated
    without any basis.
  - `calibration_report`, `expected_calibration_error`, `brier_score`, `log_loss`,
    `log_loss_from_logits` and `reliability_curve`, with fixed definitions (top-label ECE, 10
    bins closed on the right; multi-class Brier in `[0, 2]`; natural log-loss without clipping,
    `inf` for a zero probability on the label).
  - The formulas of the answers as public functions, and batch versions: `calibrated_softmax_batch`,
    `yes_no_probability`, `choice_confidence`, `score_value`, `score_confidence`. They give the
    same bits as `ask` for the same probabilities. Unlike Kev, a row that does not sum to 1 (in
    particular a sum of 0) is an error, not "uniform" (`sum_tolerance`, at most 0.05, for
    rounded data).
  - Rows may have different lengths; lists, tuples and numpy arrays are accepted.
  `calibrated_softmax` is unchanged.
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
- **The self-check no longer depends on `prefix_cache_mib`**: the vectors of a model always run in
  the segments of the template's cuts, the same computation a default `Jev` uses, even when the cache
  is off. The two ways of computing a row (in one piece, or in segments) differ by a few hundredths
  of probability on a quantised model (measured on the v2: up to 0.04 for Q8_0, up to 0.09 for a
  4-bit file, at temperature 1), so a check that followed the cache setting could refuse a model
  that loads with the cache on. The answers with `prefix_cache_mib=0` are unchanged.
- **Option keys of several tokens are scored together** (v2 models): the keys that continue after
  their first token now run in groups of four in one decode instead of one decode each. A question with
  eight keys of three or four tokens costs about 1.3 times the prompt alone instead of 1.7, and one with
  twenty keys about 1.4 instead of 2.2. The probabilities are the same (within 5e-5 of the old path, and
  closer to the reference engine), the memory of the context grows by about 50 MiB.
- **The default model is now the Decision Index model (v2)**: `Jev.from_pretrained()` loads revision
  `7730248d` of `nickprock/archai-jev-qwen-1.5b`, which answers any `Choice` (up to 20 options, keys
  of up to 4 tokens, 512 tokens per prompt) and any `YesNo`, instead of four fixed tasks. A `YesNo` is scored as the
  Decision Index `noul` question; for a question about a long passage, a `Choice` with the keys
  `yes` and `no` was clearly better in our measurement (accuracy 0.76 against 0.63 on BoolQ, and
  within a few questions on five other tasks). `Score` is refused. The first release of the model
  stays available: `Jev.from_pretrained("nickprock/archai-jev-qwen-1.5b", revision="e4f3964a")`.
  The demos were rewritten for the generic model (`001_quickstart_four_tasks.py` is now
  `001_quickstart.py`).
- Prompts of the default model are now computed in segments (at the end of the task block and of
  the state) instead of in one pass. Answers stay within the tolerances of the model: on the golden
  prompts the largest difference from 0.1.0 was 0.48 on a raw logit and 0.033 on a probability,
  and no top answer changed. With the memory empty, warm or cleared the answers are identical bit
  for bit. `prefix_cache_mib=0` restores the 0.1.0 computation. A request with a single question
  about a text of a task never seen costs a little more than before (the prefix is saved).

### Fixed
- A prompt that was too long was refused with the message "state 0 + question N tokens", counting
  the whole prompt as the question; it now reports the length of the state and of the rest.

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
