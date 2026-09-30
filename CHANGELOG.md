# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `calibrated_softmax(logits, temperature=1.0)`: temperature-scaled softmax computed in Rust.
- Single abi3 wheel per platform supporting Python 3.10 and newer.

### Changed
- `calibrated_softmax` now rejects a non-finite `temperature` (`inf`) with `ValueError`.
