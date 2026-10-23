# archai-jev

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/archai-jev?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/archai-jev)
[![PyPI](https://img.shields.io/pypi/v/archai-jev)](https://pypi.org/project/archai-jev/)
[![Python](https://img.shields.io/pypi/pyversions/archai-jev)](https://pypi.org/project/archai-jev/)
[![License](https://img.shields.io/github/license/ArchAI-Labs/archai-jev)](https://github.com/ArchAI-Labs/archai-jev/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/ArchAI-Labs/archai-jev)](https://github.com/ArchAI-Labs/archai-jev/stargazers)


[![CI](https://github.com/ArchAI-Labs/archai-jev/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArchAI-Labs/archai-jev/actions/workflows/CI.yml)
[![testpypi](https://github.com/ArchAI-Labs/archai-jev/actions/workflows/testpypi.yml/badge.svg?branch=main)](https://github.com/ArchAI-Labs/archai-jev/actions/workflows/testpypi.yml)

A runtime for **System One** decision models: you give it a *state* (text, a dict, a list) and one
or more **typed questions** (choose one option, score on a scale, yes/no), and it returns **typed
values with probabilities**, never free text. The expensive part (tokenization, inference,
probabilities, checks) runs in Rust and C++ (llama.cpp); you install a wheel and use Python.

![logo](https://github.com/ArchAI-Labs/archai-jev/blob/main/logo.jpg?raw=true)

> **Status: 0.1.0, a first release.** It runs on the CPU and ships with **one small general model**
> that answers any `Choice` or `YesNo` question, reliably on the kinds of task it was trained on
> (see [The default model](#the-default-model)). Running other checkpoints (for example Kev) is
> planned.

## Install

```bash
pip install archai-jev
```

Python 3.10 or newer. Wheels exist for Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows
(x86_64). On x86_64 the engine needs a CPU with AVX2, FMA, F16C and BMI2 (about 2013-2015 or newer);
on any other CPU it raises an `InferenceError` instead of crashing. The first call downloads the
default model (about **1.6 GB**) once and keeps it in a cache folder.

## A first example

```python
from archai_jev import Jev, Choice, YesNo

jev = Jev.from_pretrained()        # the default model; downloads and verifies it the first time
print(jev.model_info.notice)       # what the model is and what it is not

answers = jev.ask(
    state="I was charged twice for my March invoice.",
    questions={
        "team": Choice(
            "Which team should handle this message?",
            {
                "billing": "payments, invoices and refunds",
                "support": "bugs and how-to questions",
                "sales": "prices and new plans",
            },
        ),
    },
)
result = answers.choices["team"]
result.value            # "billing"
result.probabilities    # {"billing": 0.93..., "support": 0.04..., "sales": 0.01...}
result.confidence       # 0.0 (no idea) .. 1.0 (certain)
result.calibrated       # True: the model declares a calibration (see below)
```

To see the download progress and the license notice, enable logging first:
`import logging; logging.basicConfig(level=logging.INFO)` (the logger is called `archai_jev`).

A `YesNo` question is a statement or a question that is true or false of the state; its answer is
a single number, the probability of *yes*:

```python
music = YesNo("Is a person performing music?")
jev.ask("A man is playing a guitar on stage.", {"music": music}).yes_nos["music"].probability   # 0.97...
```

## Question types

| Type | You write | You get |
|---|---|---|
| `Choice(instructions, options)` | one key per option, optionally with a description | `.value` (the key), `.probabilities` per key, `.confidence` |
| `Score(instructions, levels)` | an ordered list of levels | `.value` (the expected level), `.probabilities` per level, `.confidence` |
| `YesNo(instructions, criteria=None)` (alias `Noul`) | optional `"true"` and `"false"` descriptions (the default model ignores them) | `.probability` of *yes* |

The default model answers `Choice` and `YesNo`; a `Score` question raises `UnsupportedRequestError`.

Answers are immutable and carry full-precision floats (nothing is rounded). `confidence` is how far
the distribution is from "no idea" (Choice) or how concentrated it is around its mode (Score); the
formulas are in the docstrings. A call with several questions is **all or nothing**.

## "Never wrong probabilities": what that means and what can be raised

If the library cannot guarantee that the probabilities are the model's own, it **raises an error**
instead of returning numbers. There is no switch to turn a check off.

| Exception | When |
|---|---|
| `IncompatibleModelError` | the model, its manifest, a file (SHA-256), the tokenizer or the calibration does not pass the checks at load time |
| `ModelVerificationError` | the model's self-check vectors gave other numbers than expected |
| `UnsupportedRequestError` | the model cannot answer *this* request faithfully: the prompt is longer than the model's declared context (**never truncated**), more options than the model scores, an option key that is too long, or a kind of question the model does not support |
| `NumericalError` | the model produced non-finite numbers or an invalid distribution |
| `InferenceError` | the engine could not run (memory, an unsupported CPU) |
| `ModelDownloadError` | a model file could not be downloaded or found (also in `offline=True`) |
| `InvalidStateError`, `InvalidQuestionError` | your input is not valid, reported before any computation |

All of them inherit from `JevError`. A model without a **declared calibration** does not load
unless you give your own `temperature=` or `allow_uncalibrated=True`; in the second case every
answer says `calibrated=False`.

## The default model

`Jev.from_pretrained()` loads [`nickprock/archai-jev-qwen-1.5b`](https://huggingface.co/nickprock/archai-jev-qwen-1.5b)
(Qwen2.5-1.5B-Instruct, fine-tuned, quantized to Q8_0). It is a **general** `Choice` / `YesNo` model: the
options and the question are yours, written in the request, in the prompt format of the
Decision Index benchmark suite (MIT). The score of an option
is the sum of the log-probabilities of the tokens of its key, then a softmax across the options.
English only.

What it accepts is declared by the model, not guessed (`jev.model_info`):

| Limit | Value | If you go over |
|---|---|---|
| options of a `Choice` | `max_options` = 20 | `UnsupportedRequestError`, naming the limit and your value |
| tokens of an option key | `max_key_tokens` = 4 | the same |
| tokens of the whole prompt | `max_context` = 512 | the same (never truncated) |
| `Score` questions | not supported | the same |
| a key that starts another key | refused | the same, naming both keys |

The limits come from our measurements (Q8_0, one machine): with keys that do not start with one
another, accuracy at 20 options is 0.88 to 0.98 and the expected calibration error at T = 1 is 0.02 to
0.07; beyond about 350 tokens of state the model loses accuracy and becomes overconfident, which is
why the context is 512 and not more.

**Keys must not start with one another.** An option scores the sum of the log-probabilities of the
tokens of its key, so a key whose tokens are the start of another key's (`1` and `10`, `A` and `AB`,
`option_1` and `option_10`) always scores at least as much as the longer one, which can then never win.
In our measurements the right answer on such a key was never chosen. Such a request is refused; use
letters, words, or numbers of the same width (`01`, `02`, ..., `10`).

**Know its limits.** It is reliable on the kinds of task it was trained on (safety, entailment,
similarity, topic and multiple-choice questions with few options) and **overconfident** on tasks it
never saw. Measured on held-out data (Q8_0, the declared temperature 1.7497 fitted on a separate
validation split): on the training tasks accuracy is 0.89 and the expected calibration error goes from
0.062 to 0.009; on other datasets of the same task families, accuracy is 0.69 and the error only
from 0.263 to 0.207; on families it never saw, accuracy is 0.79 and the error from 0.089 to 0.042
(out-of-distribution sets: the first 300 questions of each task). The declared temperature is
fitted on those training tasks (two to five options); with many options or other families it makes
the expected calibration error worse (on 48 options with well-formed keys, from 0.07 to 0.34 on one
set), so treat the probabilities there as uncalibrated. It was not trained to
recognise prompt injections: on the two short ones we tried it answered "safe", with a low
confidence (0.30 and 0.33, against 0.95 or more on ordinary requests), so look at the confidence
and not only at the value. Measure it on your own data before relying on it (see below).

The first release of the model (revision `e4f3964a`, four fixed tasks, a temperature measured on its
own distribution) is still available: `Jev.from_pretrained("nickprock/archai-jev-qwen-1.5b", revision="e4f3964a")`.

## Measuring and fitting calibration

```python
from archai_jev import fit_temperature
from archai_jev.calibration import calibration_report

# Logits and the right option of each row: the temperature that minimises the log-loss.
fit = fit_temperature([[1.0, -1.0]] * 3, labels=[0, 0, 1])    # fit.temperature == 2 / ln 2

# Probabilities (for instance from `ask`) and labels: ECE, Brier, log-loss, reliability bins.
report = calibration_report([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], labels=[0, 1, 1, 0])
report.ece, report.brier, report.accuracy                      # 0.4, 0.45, 0.75
```

`fit_temperature` refuses (`CalibrationError`) when the data do not determine a temperature inside
`bounds`: every row uniform, labels that always match the top option, or logits that carry no
information about the labels. It never hands back a temperature at the edge. To recalibrate the
answers of `ask`, fit the log-probabilities and multiply the result by
`jev.model_info.temperature`; then pass it as `Jev.from_pretrained(temperature=...)`. The metrics
have fixed definitions (top-label ECE with 10 bins closed on the right, multi-class Brier, natural
log-loss without clipping), written in their docstrings; other libraries use other variants.
`archai_jev.calibration` also has the public formulas of the answers and their batch versions.

## Measuring a model on your own data

```python
from archai_jev import Jev
from archai_jev.evaluation import compare, load_dataset, split_dataset

dataset = load_dataset("tickets.jsonl")        # the JSONL of Kev: the request plus a "label" per question
calibration_set, held_out = split_dataset(dataset, test_fraction=0.5, seed=0)   # by group

jev = Jev.from_pretrained()
print(jev.evaluate(held_out).table())          # accuracy and calibration per kind and per question

study = jev.calibrate(calibration_set, folds=5)  # one temperature, with an out-of-fold estimate
tuned = jev.with_temperature(study)              # a new Jev; calibrated=True; the original is unchanged
print(compare({"as is": jev, "tuned": tuned}, held_out).table())
```

Labels are exact (`"returns"`, `True`, `2`), a file with a field it does not know is an error, and
nothing is skipped: a wrong label would falsify every number. `evaluate` is all or nothing (no
partial metrics) and warns if you measure on records that fitted the temperature. `compare` fails
if a model cannot answer a record instead of comparing models on different subsets, and it ranks
nothing. The metrics have fixed definitions (see their docstrings).

## Choosing and loading models

```python
from archai_jev import list_models, Jev

list_models()                                     # the models known by name; the default is marked
Jev.from_pretrained("name-from-the-list")         # a registry model (pinned revision, checked hashes)
Jev.from_pretrained(Path("my-model-folder"))      # a local folder with an archai-jev-manifest.json
Jev.from_pretrained(offline=True)                 # never touches the network: cache or error
Jev.from_pretrained(temperature=1.8)              # your own calibration
```

A model is described by a **manifest** (family, tokenizer, head, context limit, license,
calibration, self-check vectors). Models of your own go through a manifest, never through Python
code. Files are verified by SHA-256 before use; the cache folder is `ARCHAI_JEV_CACHE` or the OS
default (`~/.cache/archai-jev`, `~/Library/Caches/archai-jev`, `%LOCALAPPDATA%\archai-jev\Cache`).
Environment variables: `ARCHAI_JEV_CACHE`, `ARCHAI_JEV_OFFLINE` / `HF_HUB_OFFLINE`,
`ARCHAI_JEV_HF_ENDPOINT` / `HF_ENDPOINT`, `ARCHAI_JEV_NUM_THREADS`, `ARCHAI_JEV_PREFIX_CACHE_MIB`.

The part of a prompt that questions share is computed once and kept in memory (see
[docs/prefix-cache.md](docs/prefix-cache.md)): `Jev.from_pretrained(prefix_cache_mib=0)` turns it off,
`jev.cache_info()` shows what it holds. The weights formats the library accepts, and why the 4-bit
variant we measured is not among them, are in [docs/quantization.md](docs/quantization.md).

`Ctrl-C` stops a running `ask`/`ask_many` at the next row of the model, and cancelling an `aask`
stops its work (see [docs/concurrency.md](docs/concurrency.md)); `ask_many(..., return_exceptions=True)`
returns the error of a failed state in its place instead of losing the rest.

## Known limits of 0.1.0

- The wheels are CPU-only. A GPU (CUDA or Vulkan, for the default model) works in a build from the
  source with a Cargo feature, and `device="cpu"` stays the default; see [docs/gpu.md](docs/gpu.md).
  Metal is built but has never been run on a Mac, so no model is allowed to use it.
- One model ships (the default above). Kev checkpoints and models of your own with a "pointer" head
  are not loadable yet; the engine and the head are implemented and tested, the file reader is not.
- One request at a time per `Jev` (calls from several threads wait their turn).
- `jev.evaluate`, `jev.calibrate` and `compare` measure a model on **your** labelled data; the
  declared calibration of the default model holds on the tasks it was trained on, not on yours.
- **No performance claims are made.** Speed and memory were measured on a single machine during
  development and will be published once they are reproduced on a second one.

## Demos

Runnable, heavily commented scripts live in [`demos/`](demos/). They need a clone of the
repository (they are not part of the wheel) and the default model.

### `000_truth_invaders.py`: a retro arcade game played by the model

![Truth Invaders](https://github.com/ArchAI-Labs/archai-jev/blob/main/docs/assets/truth-invaders.gif?raw=true)

A fact appears at the top of the screen (the *state*) and claims fall like space invaders. For
each claim the model answers one `YesNo` question ("is this statement supported by the text?"): if it says *true*, the
invader lands; if it says *false*, the cannon shoots it. Wrong calls and slow decisions cost base
blocks, and the waves speed up, so the game shows how many decisions per second the model takes. The
HUD shows the measured latency, decisions per second and accuracy live on your machine.

```bash
python demos/000_truth_invaders.py                # play (SPACE starts, ESC quits)
python demos/000_truth_invaders.py --headless     # no window: accuracy and latency on all claims
```

It uses only the standard library (`tkinter`) and `archai_jev`. The model can be wrong, and the
game says so on its game over screen.

### `001` to `007`: one concept per script

Each script is a terminal program with the library calls explained line by line. They use only
the standard library and `archai_jev`, and print what they measure on your machine.

| Script | What it shows |
|---|---|
| `001_quickstart.py` | `Choice` and `YesNo`, typed answers (`value`, `probabilities`, `confidence`), several questions in one `ask`, a request the model refuses |
| `002_grounding_check.py` | a `YesNo` question as a fact checker: is each sentence of an answer supported by its source? |
| `003_duplicate_finder.py` | a `YesNo` question to group duplicate tickets, and why comparing every pair is expensive |
| `004_never_wrong_probabilities.py` | every error the library raises on purpose, and how to catch it |
| `005_calibration_lab.py` | what `temperature=` does to probabilities (and that it never changes the decision) |
| `006_async_stream.py` | `ask_many`, `aask_many` and `asyncio`: the event loop stays free while the model decides |
| `007_test_without_the_model.py` | `MockScorer` and `Fault`: test your own code without downloading the model |

`004` (first two parts) and `007` run without the model; the others download it on the first run.

## Licenses

archai-jev is licensed under the **Apache License 2.0**. It bundles llama.cpp (MIT) through the
`llama-cpp-2` crate. The default model and its base, Qwen2.5-1.5B-Instruct, are licensed under
Apache-2.0; `model_info.license` shows the license of the model you loaded. The test data under
`tests/data/` contains outputs of Kev-0.8B (Apache-2.0): see its `NOTICE`.

---

*Built with ❤️ by **[ArchAI Labs](https://github.com/ArchAI-Labs)***
