# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
