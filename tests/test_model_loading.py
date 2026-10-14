"""``Jev.from_pretrained`` and ``list_models`` (spec 005): the contract seen from Python.

Everything here runs without a model file. The test that loads the real default model needs the
files in a cache folder (``ARCHAI_JEV_TEST_CACHE``) and is skipped otherwise.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import os
import pickle
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
    VariantInfo,
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
        "prefix_cache_mib",
    ]
    for name, p in list(sig.parameters.items())[1:]:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, name
    for forbidden in ("strict", "check", "verify", "validate", "skip", "trust"):
        assert not any(forbidden in n for n in sig.parameters)


def test_no_silent_variant_choice_in_the_signature() -> None:
    """The variant is picked by ``dtype=`` and by nothing else (spec 009): no switch for a
    "lighter", "faster" or "automatic" model."""
    sig = inspect.signature(Jev.from_pretrained)
    for forbidden in (
        "quant",
        "low_mem",
        "fast",
        "precision",
        "auto",
        "light",
        "small",
    ):
        assert not any(forbidden in n for n in sig.parameters), forbidden
    assert sig.parameters["dtype"].default is None
    # the compiled function behind it has the same parameters (the stub is its description)
    stub = ast.parse(
        (Path(archai_jev.__file__).parent / "_core.pyi").read_text(encoding="utf-8")
    )
    (load,) = [
        n
        for n in ast.walk(stub)
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    ]
    names = [a.arg for a in load.args.args + load.args.kwonlyargs]
    for forbidden in ("quant", "low_mem", "fast", "precision", "auto"):
        assert not any(forbidden in n for n in names), forbidden


def test_the_environment_cannot_pick_a_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Environment variables that look like variant switches change nothing: with an empty
    cache and no network, the file asked for is the one of the default variant (q8_0).
    """
    for name in (
        "ARCHAI_JEV_DTYPE",
        "ARCHAI_JEV_QUANTIZE",
        "ARCHAI_JEV_VARIANT",
        "ARCHAI_JEV_PRECISION",
        "HF_DTYPE",
    ):
        monkeypatch.setenv(name, "q4_k_m")
    with pytest.raises(ModelDownloadError) as caught:
        Jev.from_pretrained(offline=True, cache_dir=tmp_path)
    assert "q8.gguf" in str(caught.value)


def test_list_models_is_static_and_marks_the_default() -> None:
    models = list_models()
    assert models and all(isinstance(m, ModelInfo) for m in models)
    defaults = [m for m in models if m.is_default]
    assert len(defaults) == 1
    d = defaults[0]
    assert d.name == "nickprock/archai-jev-qwen-1.5b"
    assert d.family == "qwen2-keys" and d.head == "key_logprob"
    assert d.tasks is None  # any Choice or YesNo, not a closed set
    assert d.calibrated is True and d.temperature == pytest.approx(1.7497)
    assert d.notice and "overconfident" in d.notice
    assert d.license == "Apache-2.0" and d.max_context == 768
    assert d.revision is not None and d.revision.startswith("7730248d")
    assert list_models() == models  # deterministic


def test_list_models_reports_the_variants_of_each_model() -> None:
    by_name = {m.name: m for m in list_models()}
    default = by_name["nickprock/archai-jev-qwen-1.5b"]
    assert default.variants == (
        VariantInfo(
            dtype="q8_0",
            size_bytes=1_646_572_736,
            calibrated=True,
            temperature=1.7497,
            is_default=True,
        ),
    )
    kev = by_name["jaredpalmer/kev-0.8b"]
    assert kev.variants is not None
    assert [v.dtype for v in kev.variants] == ["bf16", "f32"]
    # converted at first use: the size of the result is not known before that
    assert all(v.size_bytes is None and v.calibrated for v in kev.variants)
    assert [v.is_default for v in kev.variants] == [True, False]


def test_list_models_reports_the_limits_of_each_head() -> None:
    by_name = {m.name: m for m in list_models()}
    # the default scores the keys of up to 8 options, of up to 4 tokens each
    default = by_name["nickprock/archai-jev-qwen-1.5b"]
    assert (default.max_options, default.max_key_tokens) == (8, 4)
    # a pointer head has no limit of its own
    kev = by_name["jaredpalmer/kev-0.8b"]
    assert (kev.max_options, kev.max_key_tokens) == (None, None)


def test_the_test_scorer_has_no_limits() -> None:
    from archai_jev.testing import MockScorer

    info = Jev.from_scorer(MockScorer()).model_info
    assert (info.max_options, info.max_key_tokens) == (None, None)


def test_variant_info_is_immutable_hashable_and_picklable() -> None:
    (variant,) = list_models()[0].variants or ()
    with pytest.raises(AttributeError):
        variant.dtype = "q4_k_m"  # type: ignore[misc]
    assert hash(variant) == hash(pickle.loads(pickle.dumps(variant)))
    assert pickle.loads(pickle.dumps(variant)) == variant
    info = list_models()[0]
    assert pickle.loads(pickle.dumps(info)) == info
    assert hash(info) == hash(pickle.loads(pickle.dumps(info)))
    assert "VariantInfo" in archai_jev.__all__


def test_the_test_scorer_has_no_variants() -> None:
    from archai_jev.testing import MockScorer

    assert Jev.from_scorer(MockScorer()).model_info.variants is None


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


@pytest.mark.parametrize("dtype", ["int8", "q4", "fp16", "Q8_0", ""])
def test_unknown_dtype_names_are_value_errors_that_list_the_supported_ones(
    dtype: str,
) -> None:
    with pytest.raises(ValueError, match="not a known dtype") as caught:
        Jev.from_pretrained(dtype=dtype, offline=True)
    message = str(caught.value)
    for name in ("f32", "bf16", "q8_0"):
        assert name in message


def test_int8_is_pointed_to_q8_0() -> None:
    with pytest.raises(ValueError, match=r"'q8_0' is the 8-bit integer"):
        Jev.from_pretrained(dtype="int8", offline=True)
    with pytest.raises(ValueError) as caught:
        Jev.from_pretrained(dtype="q4", offline=True)
    assert "8-bit integer" not in str(caught.value)


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
    assert safe.probabilities["SAFE"] >= 0.99  # the README example says "0.99..."
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


GOLDEN = Path(__file__).parent / "data" / "default-golden" / "golden.jsonl"
DECLARED_TEMPERATURE = 2.09
Q8_0_TOLERANCE = 0.2  # spec 006c: the tolerance of the published Q8_0 file
ARGMAX_GAP = 0.25


def _oracle_probabilities(logits: list[float]) -> list[float]:
    """The PyTorch oracle's softmax at the declared temperature, over the asked letters only."""
    scaled = [z / DECLARED_TEMPERATURE for z in logits]
    top = max(scaled)
    exps = [math.exp(z - top) for z in scaled]
    return [e / sum(exps) for e in exps]


@pytest.mark.skipif(
    not os.environ.get("ARCHAI_JEV_TEST_CACHE"),
    reason="set ARCHAI_JEV_TEST_CACHE to a cache folder that holds the default model",
)
def test_the_default_model_agrees_with_the_oracle_on_every_task() -> None:
    jev = Jev.from_pretrained(
        offline=True, cache_dir=os.environ["ARCHAI_JEV_TEST_CACHE"]
    )
    notice = jev.model_info.notice
    assert notice and "Demo model" in notice
    assert jev.model_info.tasks == ("safety", "intent", "entailment", "similarity")

    rows = [json.loads(line) for line in GOLDEN.read_text("utf-8").splitlines()]
    assert len(rows) == 48
    worst: dict[str, float] = {}
    for row in rows:
        spec = row["request"]["questions"]["q"]
        criteria = spec["criteria"]
        question: Choice | YesNo
        if spec["type"] == "choice":
            described = any(v is not None for v in criteria.values())
            question = Choice(
                spec["instructions"], criteria if described else list(criteria)
            )
        else:
            question = YesNo(spec["instructions"], criteria)
        answers = jev.ask(row["request"]["state"], {"q": question})
        want = _oracle_probabilities(row["oracle_logits"])
        if spec["type"] == "choice":
            got = list(answers.choices["q"].probabilities.values())
        else:
            # The oracle's letters are [FALSE, TRUE]: [no, yes], as the library's YesNo.
            got = [
                1.0 - answers.yes_nos["q"].probability,
                answers.yes_nos["q"].probability,
            ]
        delta = max(abs(g - w) for g, w in zip(got, want, strict=True))
        worst[row["task"]] = max(worst.get(row["task"], 0.0), delta)
        assert delta <= Q8_0_TOLERANCE, f"{row['id']}: |dp| = {delta:.3f}"
        # The winner is the oracle's unless its top two are closer than 0.25 (spec 006c, Q8_0).
        ranked = sorted(want, reverse=True)
        if ranked[0] - ranked[1] >= ARGMAX_GAP:
            assert got.index(max(got)) == want.index(ranked[0]), f"{row['id']}: argmax"
    print("max |dp| per task:", {k: round(v, 4) for k, v in worst.items()})
