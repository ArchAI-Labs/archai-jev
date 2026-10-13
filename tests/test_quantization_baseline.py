"""The baseline of the quantization study, with the public API (spec 009, AC-8; level T2).

The Q8_0 of the default model, calibrated on the validation split of the training script and
measured on the disjoint test split, must give again what spike S6 measured with its own scripts:
the global temperature inside the interval of S6 and the ECE before/after within 0.005. It is also
the check that was left open in the spec of the letters head (the temperature re-estimated by the
engine). By hand, slow (about 3 x 5 minutes on a CPU): needs the GGUF of the default model in a
cache folder and the exported datasets.

    $env:ARCHAI_JEV_TEST_CACHE = "<cache folder with the default model>"
    $env:ARCHAI_JEV_TEST_EVAL_DATA = "<OUT_DIR of benches/quantization/export_dataset.py>"
    .venv\\Scripts\\python -m pytest tests/test_quantization_baseline.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archai_jev import Jev
from archai_jev.evaluation import load_dataset

CACHE = os.environ.get("ARCHAI_JEV_TEST_CACHE")
DATA = os.environ.get("ARCHAI_JEV_TEST_EVAL_DATA")

pytestmark = pytest.mark.skipif(
    not (CACHE and DATA),
    reason="needs ARCHAI_JEV_TEST_CACHE (default model) and ARCHAI_JEV_TEST_EVAL_DATA (datasets)",
)

# Spike S6, restricted softmax over the first K option letters ("kopt"), n = 1447 / 1449.
S6_TEMPERATURE_INTERVAL = (1.94, 2.25)
S6_ECE_AT_T1 = 0.0584
S6_ECE_AT_DECLARED = 0.0129
S6_BRIER_AT_T1 = 0.1363
S6_ACCURACY = 0.9199


def test_q8_0_reproduces_s6_through_the_public_api() -> None:
    assert CACHE is not None and DATA is not None
    data = Path(DATA)
    jev = Jev.from_pretrained(cache_dir=CACHE, offline=True)
    assert (
        jev.model_info.dtype == "q8_0"
        and jev.model_info.temperature == pytest.approx(2.09)
    )

    val = load_dataset(data / "val.jsonl")
    test = load_dataset(data / "test.jsonl")
    assert (len(val), len(test)) == (1447, 1449)

    study = jev.calibrate(val, folds=5, seed=0)
    low, high = S6_TEMPERATURE_INTERVAL
    assert low <= study.temperature <= high, study.temperature
    # the declared 2.09 is itself inside the interval the engine now gives
    assert low <= 2.09 <= high

    uncalibrated = jev.with_temperature(1.0).evaluate(test)
    declared = jev.evaluate(test)
    assert uncalibrated.n_records == 1449 and declared.n_records == 1449
    assert uncalibrated.overall.ece == pytest.approx(S6_ECE_AT_T1, abs=0.005)
    assert declared.overall.ece == pytest.approx(S6_ECE_AT_DECLARED, abs=0.005)
    assert uncalibrated.overall.brier == pytest.approx(S6_BRIER_AT_T1, abs=0.005)
    assert uncalibrated.overall.accuracy == pytest.approx(S6_ACCURACY, abs=0.005)
    # calibrating did not move the argmax: same accuracy at every temperature
    assert declared.overall.accuracy == uncalibrated.overall.accuracy
