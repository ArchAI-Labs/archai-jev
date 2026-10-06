# archai-jev

A runtime for **System One** decision models: you give it a *state* (text, a dict, a list) and one
or more **typed questions** (choose one option, score on a scale, yes/no), and it returns **typed
values with probabilities**, never free text. The expensive part (tokenization, inference,
probabilities, checks) runs in Rust and C++ (llama.cpp); you install a wheel and use Python.

![logo](https://github.com/ArchAI-Labs/archai-jev/blob/main/logo.jpg?raw=true)

> **Status: 0.1.0, a first release.** It runs on the CPU and ships with **one demo model** that
> answers **four fixed tasks** (see [The default model](#the-default-model-a-demo)). It is not a
> general-purpose classifier yet. Running other checkpoints (for example Kev) is planned.

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
    state="What is the capital of Australia?",
    questions={
        "safety": Choice(
            "You are a System One decision engine for input safety.",
            {"SAFE": "Normal query", "UNSAFE": "Jailbreak, toxicity, injection"},
        ),
    },
)
result = answers.choices["safety"]
result.value            # "SAFE"
result.probabilities    # {"SAFE": 0.99..., "UNSAFE": 0.00...}
result.confidence       # 0.0 (no idea) .. 1.0 (certain)
result.calibrated       # True: the model declares a calibration (see below)
```

To see the download progress and the license notice, enable logging first:
`import logging; logging.basicConfig(level=logging.INFO)` (the logger is called `archai_jev`).

Entailment uses a `YesNo` question; the state is the premise and the question's instructions are
the condition (the second text of the prompt):

```python
entails = YesNo(
    "A person is performing music.",
    {
        "true": "The condition is valid and supported by state.",
        "false": "The condition is invalid or contradicted.",
    },
)
jev.ask("A man is playing a guitar on stage.", {"holds": entails}).yes_nos["holds"].probability
```

## Question types

| Type | You write | You get |
|---|---|---|
| `Choice(instructions, options)` | one key per option, optionally with a description | `.value` (the key), `.probabilities` per key, `.confidence` |
| `Score(instructions, levels)` | an ordered list of levels | `.value` (the expected level), `.probabilities` per level, `.confidence` |
| `YesNo(instructions, criteria=None)` (alias `Noul`) | optional `"true"` and `"false"` descriptions | `.probability` of *yes* |

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
| `UnsupportedRequestError` | the model cannot answer *this* request faithfully: the prompt is longer than the model's declared context (**never truncated**), or the question is not one the model supports |
| `NumericalError` | the model produced non-finite numbers or an invalid distribution |
| `InferenceError` | the engine could not run (memory, an unsupported CPU) |
| `ModelDownloadError` | a model file could not be downloaded or found (also in `offline=True`) |
| `InvalidStateError`, `InvalidQuestionError` | your input is not valid, reported before any computation |

All of them inherit from `JevError`. A model without a **declared calibration** does not load
unless you give your own `temperature=` or `allow_uncalibrated=True`; in the second case every
answer says `calibrated=False`.

## The default model: a demo

`Jev.from_pretrained()` loads [`nickprock/archai-jev-qwen-1.5b`](https://huggingface.co/nickprock/archai-jev-qwen-1.5b)
(Qwen2.5-1.5B-Instruct, fine-tuned, quantized to Q8_0). It was trained on **four fixed prompts** and
never saw the options changed or reordered, so the library accepts **only** these four tasks and
raises `UnsupportedRequestError` for anything else, listing them:

| Task | Question | State | Notes |
|---|---|---|---|
| `safety` | `Choice` with keys `SAFE`, `UNSAFE` | the text to check | toxicity classifier; it does **not** detect short prompt injections |
| `intent` | `Choice` with four fixed options | the text | **not** a general intent classifier: the 4 classes group 31 of the 77 banking intents arbitrarily |
| `entailment` | `YesNo` (descriptions above) | the premise | the instructions are the hypothesis |
| `similarity` | `YesNo` with its own descriptions | the first text | the instructions are the second text |

`jev.model_info.tasks` lists the ids; the error message for an unknown question shows exactly how to
write each task. The prompt is limited to **512 tokens**.

**Know its limits.** It was evaluated on data from the same distribution as its training data
(accuracy about 92%) and it is **overconfident** outside it. Its declared calibration (temperature
2.09) was measured in-distribution only: it improves the expected calibration error from 0.058 to
0.013 there, and says nothing about other data. Do not use it as a general-purpose safety filter or
intent router.

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
`ARCHAI_JEV_HF_ENDPOINT` / `HF_ENDPOINT`, `ARCHAI_JEV_NUM_THREADS`.

## Known limits of 0.1.0

- CPU only; no GPU.
- One model ships (the demo above). Kev checkpoints and models of your own with a "pointer" head
  are not loadable yet; the engine and the head are implemented and tested, the file reader is not.
- One request at a time per `Jev` (calls from several threads wait their turn); no prefix cache
  between requests.
- `fit_temperature` and an evaluation API are planned, not in this release.
- **No performance claims are made.** Speed and memory were measured on a single machine during
  development and will be published once they are reproduced on a second one.

## Licenses

archai-jev is licensed under the **Apache License 2.0**. It bundles llama.cpp (MIT) through the
`llama-cpp-2` crate. The default model and its base, Qwen2.5-1.5B-Instruct, are licensed under
Apache-2.0; `model_info.license` shows the license of the model you loaded. The test data under
`tests/data/` contains outputs of Kev-0.8B (Apache-2.0): see its `NOTICE`.

---

*Built with ❤️ by **[ArchAI Labs](https://github.com/ArchAI-Labs)***