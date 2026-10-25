"""Every public name of the calibration feature is documented, and its examples run."""

from __future__ import annotations

import doctest
import math
from typing import Any

import pytest

from archai_jev import CalibrationError, fit_temperature
from archai_jev import calibration as cal


def public_objects() -> list[tuple[str, Any]]:
    objects: list[tuple[str, Any]] = [(n, getattr(cal, n)) for n in cal.__all__]
    objects.append(("fit_temperature", fit_temperature))
    return objects


FUNCTIONS = [
    (n, o) for n, o in public_objects() if callable(o) and not isinstance(o, type)
]
CLASSES = [(n, o) for n, o in public_objects() if isinstance(o, type)]


@pytest.mark.parametrize(("name", "obj"), FUNCTIONS, ids=[n for n, _ in FUNCTIONS])
def test_every_public_function_has_raises_and_example(name: str, obj: Any) -> None:
    doc = obj.__doc__ or ""
    assert doc.strip(), name
    assert "Raises:" in doc, f"{name} has no Raises section"
    assert ">>>" in doc, f"{name} has no example"


@pytest.mark.parametrize(("name", "obj"), CLASSES, ids=[n for n, _ in CLASSES])
def test_every_public_class_has_a_docstring(name: str, obj: Any) -> None:
    assert (obj.__doc__ or "").strip(), name


@pytest.mark.parametrize(("name", "obj"), FUNCTIONS, ids=[n for n, _ in FUNCTIONS])
def test_doctests_pass(name: str, obj: Any) -> None:
    """Run the examples of each docstring (also those of functions written in Rust)."""
    parser = doctest.DocTestParser()
    examples = parser.get_examples(obj.__doc__ or "")
    assert examples, name
    globs = dict(vars(cal))
    globs["fit_temperature"] = fit_temperature
    globs["math"] = math
    test = doctest.DocTest(examples, globs, name, None, 0, obj.__doc__)
    runner = doctest.DocTestRunner(verbose=False)
    runner.run(test)
    assert runner.failures == 0, f"{name}: {runner.failures} failing example(s)"


def test_spec_example() -> None:
    # Three examples, two right and one wrong, all with logits [1, -1]: T = 2 / ln 2.
    fit = fit_temperature([[1.0, -1.0]] * 3, labels=[0, 0, 1])
    assert abs(fit.temperature - 2 / math.log(2)) < 1e-9
    assert fit.loss_after < fit.loss_before

    report = cal.calibration_report(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], labels=[0, 1, 1, 0]
    )
    assert abs(report.ece - 0.4) < 1e-12 and abs(report.brier - 0.45) < 1e-12
    assert abs(report.log_loss - 0.6455747489491624) < 1e-12

    # Data without information: an error, not an invented temperature.
    with pytest.raises(CalibrationError):
        fit_temperature([[1.0, -1.0]] * 3, labels=[1, 1, 1])

    # The recipe for the answers of `ask`: log-probabilities as logits give the relative T.
    t0 = 2.0
    logits = [[1.0, -1.0], [0.5, 0.2], [-2.0, 1.0], [0.1, 0.0]]
    labels = [0, 1, 1, 0]
    probabilities = [cal.calibrated_softmax(row, t0) for row in logits]
    relative = fit_temperature(
        [[math.log(p) for p in row] for row in probabilities], labels
    ).temperature
    assert relative * t0 == pytest.approx(
        fit_temperature(logits, labels).temperature, rel=1e-6
    )
