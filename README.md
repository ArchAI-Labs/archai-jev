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

> **Status: 0.2.0.** It runs on the CPU and ships with **one small general model** that answers any
> `Choice` or `YesNo` question, reliably on the kinds of task it was trained on (see
> [The default model](#the-default-model)). Kev-0.8B loads from its original public files
> ([docs/model-files.md](docs/model-files.md)). Speed and memory are in [Performance](#performance),
> including where we are slower than Kev on PyTorch.

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

## Performance

All numbers below were measured by us on **one machine**, with the scripts in
[`scripts/bench_release/`](scripts/bench_release/); they are not reproduced on a second one yet. Treat them
as an order of magnitude, not a guarantee. Where we lose, the table says so.

**Machine**: Intel Core Ultra 7 255HX (8 performance + 12 efficiency cores, 20 threads), 32 GB RAM, Windows 11
Pro, "Best performance" power mode on AC, 2026-10-23 (Kev run) and 2026-10-25 (default-model run). **Versions**: archai_jev 0.2.0 (release wheel, CPU) and
archai_jev 0.1.0 (the wheel built for 0.1.0), Python 3.13; Kev on PyTorch 2.8.0 (CPU), transformers 5.17.0,
Python 3.12, fp32, Kev's own default loading.

**Method** (the same for every row):
- One process per contender, one model per process, **alternating blocks**: in each of 3 blocks every contender
  runs once, in the opposite order in odd blocks, so a slow minute of the machine hits all of them. A twin of the
  reference (same configuration, run twice, "A/A") must give a ratio between 0.95 and 1.05; a cell where it does
  not is marked with `†` and is not used for any ratio.
- Latency is `Jev.ask` (or Kev's library path) called from Python, so it **includes the Python-Rust boundary** and the
  tokenization. Every request has a different state (no result is ever reused). Figures are `p50 / p95` in
  milliseconds, the median over the 3 blocks of each block's value, and `[min-max]` is the spread of the block
  medians. "Apart" means a pause of 0.5 s before each call: the CPU clock drops while idle, and the same request
  then costs more on this laptop.
- Capacity is requests per second with 1, 4 and 16 caller threads sharing one model.
- Memory is the peak working set of the process over the whole run (including the 16-caller series).
- The default-model run was done twice: the first time with Firefox open (10-20% of the CPU), the second with it closed;
  the tables show the second, and the text says where the first one differs. The Kev run had a quiet machine.

### Default model (`nickprock/archai-jev-qwen-1.5b`, Q8_0), 0.2.0 against 0.1.0

0.1.0 shipped a different, older default model (four fixed tasks, a shorter prompt), so this compares the
two releases as a user gets them, not the engine alone. Requests: a 94-token request with one `Choice` (W1),
112 tokens (W2), 365 tokens (W3), and ten `YesNo` questions about one state, 1,537 tokens in all (Q10).

| Request | 0.2.0 back to back | 0.2.0 apart (0.5 s) | 0.2.0 apart (1.5 s) | 0.1.0 back to back / apart 0.5 s / apart 1.5 s (p50) |
|---|---|---|---|---|
| W1, 94 tokens | 290 / 331 [266-432] ms | 422 / 516 [411-577] ms † | 494 / 524 [472-588] ms | 224 / 380 / 434 ms |
| W2, 112 tokens | 338 / 408 [332-438] ms | 466 / 484 [451-517] ms | 548 / 642 [534-563] ms | 279 / 425 / 481 ms |
| W3, 365 tokens | 1141 / 1200 [1101-1223] ms | 1178 / 1310 [1173-1297] ms † | 1361 / 1480 [1298-1411] ms | 1041 / 1046 / 1232 ms |
| Q10, ten questions | 1684 / 1746 [1605-1694] ms | 1897 / 1920 [1728-1946] ms † | 2046 / 3076 [1813-2570] ms † | 5191 / 5390 / 5624 ms |

- A request is **slower in 0.2.0**: where the A/A check held, 0.1.0 needs 0.82 to 0.93 of the time (W1, W2 and W3). The
  new prompt is longer and the model is different. Ten questions about one state are **about 3.2 times faster** in 0.2.0
  (3.2 in this run and 3.1 and 3.3 in the first, the cells where the check held), because what the questions share is
  computed once ([docs/prefix-cache.md](docs/prefix-cache.md)).
- Capacity (requests per second on W1; one `Jev` serves one request at a time): 0.2.0 2.91 / 3.30 / 3.11 for 1 / 4 / 16
  callers, 0.1.0 3.83 / 3.90 / 3.93. In the first run the two were equal (2.95 / 2.91 / 3.23 against 2.79 / 2.99 / 2.89), and the
  spread between blocks (2.3 to 4.0) is as large as the difference: do not read a gap into it.
- Start: process start to first answer 1.43 s (import 0.23 s, load 0.38 s, first call 0.71 s) against 1.24 s for 0.1.0.
  Peak memory 1.91 GiB against 1.76 GiB. Wheel 4.5 MiB against 4.1 MiB; the model file is 1.57 GiB.
- Eight threads instead of the 20 of the default (`ARCHAI_JEV_NUM_THREADS=8`) took 0.65 of the time on W1 with a
  1.5 s pause and 1.21 times the time on W3 back to back (cells where the A/A check held). On other CPUs the best
  number differs.
- The two runs agree on the ratios; in absolute terms the second is about 10% faster than the first.

### Kev-0.8B, archai_jev against Kev on PyTorch (CPU)

The same checkpoint (`jaredpalmer/kev-0.8b`), the same inputs and the same three questions (one `Choice`, one `YesNo`,
one `Score`) on a 300-token state, 402 tokens in all: 50 requests back to back and 20 apart. Kev is measured with its
default threads and tuned (`kev 8 threads`: 8 intra-op threads for one caller, 5 for 4 callers, 1 for 16, the best
values of our earlier sweep). The comparison that counts is against the tuned one.

| | back to back | apart (0.5 s) | capacity 1 / 4 / 16 callers | start to first answer | peak memory |
|---|---|---|---|---|---|
| kev default (fp32, 20 threads) | 986 / 1035 [981-1442] ms | 1457 / 1501 ms | 1.05 / 1.75 / 1.47 req/s | 6.9 s | 6.4 GiB |
| kev 8 threads (fp32) | 1081 / 1107 ms | 1106 / 1127 ms | 0.97 / 1.95 / 1.82 req/s | 6.3 s | 6.1 GiB |
| archai_jev bf16 (Kev's default variant) | 1670 / 1734 ms | 1697 / 2105 ms | 0.58 / 0.59 / 0.59 req/s | 3.0 s | 2.8 GiB |
| archai_jev f32 | 1571 / 2057 ms | 1591 / 2071 ms | 0.58 / 0.62 / 0.62 req/s | 3.4 s | 4.2 GiB |
| archai_jev bf16, 8 threads | 2277 / 2308 ms | 2258 / 2291 ms | 0.44 / 0.44 / 0.44 req/s | 3.6 s | 2.8 GiB |

- **We are slower on CPU.** Against tuned Kev, one request costs 1.55 times as much in bf16 and 1.45 times in f32
  (blocks 1.52 to 1.55 and 1.43 to 1.46; the A/A twin agrees within 1%). Capacity is 3.3 times lower with four callers
  (0.59 against 1.95 req/s): one `Jev` serves one request at a time, while Kev's library can run several on separate
  threads. With its default threads Kev showed one slower block (981 to 1442 ms): that spread is real.
- **We start faster and use less memory**: 2.1 times faster to the first answer and 2.2 times less memory in bf16
  (1.4 times in f32; Kev holds fp32 weights plus PyTorch). No PyTorch is needed.
- Disk: the wheel is 4.5 MiB plus the converted model (1.45 GiB in bf16, kept in the cache next to the 1.7 GiB of original
  files); a clean Kev environment on Windows was 1.61 GiB plus 1.71 GiB of weights when we measured it earlier in
  development (not measured again on 2026-10-23).
- Same answers: against Kev on PyTorch the largest probability difference on five requests is 0.0059 in bf16 and
  0.0012 in f32 (Kev against itself, 9e-7).
- GPU: the wheels are CPU-only and we did not measure a 0.2.0 GPU build for this table; what a CUDA and a Vulkan
  build did for the default model is in [docs/gpu.md](docs/gpu.md).

### Reproduce

```powershell
# default model: unpack the 0.2.0 and the 0.1.0 wheel (rename to .zip), edit the paths in contenders_default.json, then
scripts\bench_release\run_blocks.ps1 -Contenders scripts\bench_release\contenders_default.json -Blocks 3 -Out out\default
python scripts\bench_release\summarize.py default out\default --reference r020 --twin r020-aa

# Kev: KEV_SRC = a checkout of Kev, KEV_PYTHON = the python.exe of an environment with torch, transformers, peft
python scripts\bench_release\kev\make_inputs.py scripts\bench_release\kev\inputs.json
scripts\bench_release\kev\run_kev_blocks.ps1 -Blocks 3 -Out out\kev
python scripts\bench_release\summarize.py kev out\kev --reference kev-t8
```

The runners refuse to start unless the machine is on AC power in the best performance mode, and wait for a quiet
CPU before every process. In the "Balanced" mode the same request can cost twice as much.

## Known limits of 0.2.0

- The wheels are CPU-only. A GPU (CUDA or Vulkan, for the default model) works in a build from the
  source with a Cargo feature, and `device="cpu"` stays the default; see [docs/gpu.md](docs/gpu.md).
  Metal is built but has never been run on a Mac, so no model is allowed to use it.
- The registry has the default model and Kev-0.8B; models of your own go through a manifest.
- One request at a time per `Jev` (calls from several threads wait their turn).
- `jev.evaluate`, `jev.calibrate` and `compare` measure a model on **your** labelled data; the
  declared calibration of the default model holds on the tasks it was trained on, not on yours.
- Kev needs several GB of free virtual memory (the Windows commit charge) on top of its weights, because each request
  creates a context sized for its 8,205-token limit. With less than about 5 GB free the process can stop with a
  `GGML_ASSERT` message instead of a Python error. The default model is not affected.

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
