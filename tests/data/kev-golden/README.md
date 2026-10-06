# Kev golden (reduced)

Reference data for the parity tests of the prompt builder and, later, of the inference engine.

## What is here

| File | Rows | Content |
|---|---|---|
| `golden.jsonl` | 65 | requests and Kev-0.8B's answers (spike S2) |
| `golden_extra.jsonl` | 4 | the same for states of 300, 1,000 and 2,500 tokens (spike S4) |
| `selfcheck.jsonl` | 8 | the small subset used as self-check vectors of a model manifest |
| `PROVENANCE.json` | - | revisions, versions, SHA-256 of the originals and of these files |

Each row has the request (`request`), the packed prompt ids (`ids`), where the head reads
(`branch_start`, `branch_len`, `decide_pos`, `opt_pos`), and per question the raw head logits, the
logits after the temperature, the probabilities, the value and the confidence. Numbers are float32
values stored as JSON doubles.

Dropped from the originals (not used by any test): `prompt_text`, `xcheck`, `seconds`, `why`,
`state_text`, `answer`, `confidence_legacy_45923b7`, `instr_text`, `option_texts`.

## Where the data comes from

- **Inputs**: the requests were written by the authors of archai-jev.
- **Outputs**: computed by **Kev-0.8B** (`jaredpalmer/kev-0.8b`, revision `9a45d25e`, Apache-2.0)
  on top of `Qwen/Qwen3.5-0.8B-Base` (Apache-2.0), with Kev's own code (commit `67117d85`), in
  fp32 on CPU with the LoRA merged in fp32.
- The script that produced them needs PyTorch and Kev's code and is not part of this repository;
  `PROVENANCE.json` records the versions and the hashes.

## What the tests need besides these files

The `tokenizer.json` of `jaredpalmer/kev-0.8b` (19 MiB) is **not** in the repository. Point
`ARCHAI_JEV_TEST_TOKENIZER` at it; its SHA-256 is checked. With `ARCHAI_JEV_REQUIRE_ASSETS=1` (CI)
a missing file fails the test instead of skipping it.

## Requests the library refuses on purpose

Eight requests of the golden are valid for Kev and refused by archai-jev, which is stricter
(a bare number or `null` as the whole state, `null` instructions, numeric levels or descriptions,
unknown keys in the criteria of a yes/no question). The parity test lists them by id.
