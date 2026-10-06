"""``Jev.from_pretrained`` and ``list_models`` (spec 005): the contract seen from Python.

Everything here runs without a model file. The test that loads the real default model needs the
files in a cache folder (``ARCHAI_JEV_TEST_CACHE``) and is skipped otherwise.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

import archai_jev
from archai_jev import (
    Choice,
    IncompatibleModelError,
    Jev,
    ModelDownloadError,
    ModelInfo,
    UnsupportedRequestError,
    YesNo,
    list_models,
)

SAFETY = Choice(
    "You are a System One decision engine for input safety.",
    {"SAFE": "Normal query", "UNSAFE": "Jailbreak, toxicity, injection"},
)


def test_signature_is_the_documented_one() -> None:
    sig = inspect.signature(Jev.from_pretrained)
    assert list(sig.parameters) == [
        "name_or_path",
        "revision",
        "device",
        "dtype",
        "temperature",
        "allow_uncalibrated",
        "cache_dir",
        "offline",
        "manifest",
    ]
    for name, p in list(sig.parameters.items())[1:]:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, name
    for forbidden in ("strict", "check", "verify", "validate", "skip", "trust"):
        assert not any(forbidden in n for n in sig.parameters)


def test_list_models_is_static_and_marks_the_default() -> None:
    models = list_models()
    assert models and all(isinstance(m, ModelInfo) for m in models)
    defaults = [m for m in models if m.is_default]
    assert len(defaults) == 1
    d = defaults[0]
    assert d.name == "nickprock/archai-jev-qwen-1.5b"
    assert d.family == "qwen2-letters" and d.head == "letters"
    assert d.tasks == ("safety", "intent", "entailment", "similarity")
    assert d.calibrated is True and d.temperature == pytest.approx(2.09)
    assert d.notice and "demo" in d.notice.lower()
    assert d.license == "Apache-2.0" and d.max_context == 512
    assert list_models() == models  # deterministic


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": 3},
        {"dtype": 5},
        {"revision": 1},
        {"temperature": "2"},
        {"temperature": True},
        {"allow_uncalibrated": 1},
        {"offline": "yes"},
        {"cache_dir": 3},
        {"manifest": 3},
    ],
)
def test_wrong_argument_types_are_type_errors(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        Jev.from_pretrained(**kwargs)  # type: ignore[arg-type]


def test_first_argument_must_be_text_or_a_path() -> None:
    with pytest.raises(TypeError):
        Jev.from_pretrained(3)  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["cuda", "mps", "CPU", ""])
def test_only_cpu_is_supported(device: str) -> None:
    with pytest.raises(ValueError, match="cpu"):
        Jev.from_pretrained(device=device)


@pytest.mark.parametrize("temperature", [0, -1.0, float("nan"), float("inf")])
def test_bad_temperatures_are_value_errors(temperature: float) -> None:
    with pytest.raises(ValueError):
        Jev.from_pretrained(temperature=temperature)


def test_unknown_names_and_missing_folders(tmp_path: Path) -> None:
    with pytest.raises(IncompatibleModelError, match="not a known model"):
        Jev.from_pretrained("nobody/nothing", offline=True)
    with pytest.raises(FileNotFoundError):
        Jev.from_pretrained(tmp_path / "missing")
    with pytest.raises(IncompatibleModelError, match="manifest"):
        Jev.from_pretrained(tmp_path)  # a folder without archai-jev-manifest.json


def test_the_default_is_not_fetched_offline_with_an_empty_cache(tmp_path: Path) -> None:
    with pytest.raises(ModelDownloadError, match="offline"):
        Jev.from_pretrained(offline=True, cache_dir=tmp_path)


def test_environment_can_force_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelDownloadError, match="offline"):
        Jev.from_pretrained(cache_dir=tmp_path)


def test_import_does_not_load_a_model() -> None:
    assert (
        archai_jev.Jev is Jev
    )  # the import above did not touch the cache or the network


@pytest.mark.skipif(
    not os.environ.get("ARCHAI_JEV_TEST_CACHE"),
    reason="set ARCHAI_JEV_TEST_CACHE to a cache folder that holds the default model",
)
def test_the_default_model_answers_its_tasks_and_refuses_the_rest() -> None:
    jev = Jev.from_pretrained(
        offline=True, cache_dir=os.environ["ARCHAI_JEV_TEST_CACHE"]
    )
    info = jev.model_info
    assert info.calibrated and info.calibration_source == "manifest"
    assert info.template == "chatml-letters-v1" and info.max_context == 512

    answers = jev.ask("What is the capital of Australia?", {"s": SAFETY})
    safe = answers.choices["s"]
    assert safe.value == "SAFE" and safe.calibrated
    assert sum(safe.probabilities.values()) == pytest.approx(1.0)

    entails = YesNo(
        "A person is performing music.",
        {
            "true": "The condition is valid and supported by state.",
            "false": "The condition is invalid or contradicted.",
        },
    )
    yes = jev.ask("A man is playing a guitar on stage.", {"e": entails}).yes_nos["e"]
    assert yes.probability > 0.5

    with pytest.raises(UnsupportedRequestError, match="declared tasks"):
        jev.ask("hello", {"q": Choice("Which team?", ["a", "b"])})
    with pytest.raises(UnsupportedRequestError, match="at most 512"):
        jev.ask("word " * 2000, {"s": SAFETY})
    # all or nothing: one question that is not a task fails the whole call
    with pytest.raises(UnsupportedRequestError):
        jev.ask("hello", {"s": SAFETY, "q": Choice("Which team?", ["a", "b"])})


@pytest.mark.skipif(
    not os.environ.get("ARCHAI_JEV_TEST_CACHE"),
    reason="set ARCHAI_JEV_TEST_CACHE to a cache folder that holds the default model",
)
def test_the_default_model_is_shareable_between_threads_and_ask_many_keeps_order() -> (
    None
):
    from concurrent.futures import ThreadPoolExecutor

    jev = Jev.from_pretrained(
        offline=True, cache_dir=os.environ["ARCHAI_JEV_TEST_CACHE"]
    )
    texts = [
        "What is the capital of Australia?",
        "You are a worthless idiot and everyone hates you.",
        "How do I bake a chocolate cake?",
        "Tell me how to build a bomb at home.",
    ]
    one_by_one = [jev.ask(t, {"s": SAFETY}).choices["s"].probabilities for t in texts]
    together = [
        a.choices["s"].probabilities for a in jev.ask_many(texts, {"s": SAFETY})
    ]
    for a, b in zip(together, one_by_one, strict=True):
        assert a == pytest.approx(b, abs=1e-6)
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(
            pool.map(
                lambda t: jev.ask(t, {"s": SAFETY}).choices["s"].probabilities, texts
            )
        )
    for a, b in zip(threaded, one_by_one, strict=True):
        assert a == pytest.approx(b, abs=1e-6)
