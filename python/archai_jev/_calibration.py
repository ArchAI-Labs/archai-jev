"""Calibration: public formulas, ``fit_temperature`` and metrics.

Everything is computed in Rust (``archai_jev._core``). The functions here convert nothing and
decide nothing: they give the public names, the docstrings and the immutable result classes.
``archai_jev.calibration`` re-exports the whole set; ``fit_temperature``, ``TemperatureFit`` and
``CalibrationError`` are also importable from ``archai_jev``.

Rows of logits or probabilities are sequences of sequences of numbers; the rows may have
different lengths. Numpy arrays are accepted, read element by element (convert with
``.tolist()`` for speed). Everything is ``float64``, nothing is rounded, and the results are
deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import _core
from ._core import calibrated_softmax

__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "TemperatureFit",
    "brier_score",
    "calibrated_softmax",
    "calibrated_softmax_batch",
    "calibration_report",
    "choice_confidence",
    "choice_confidence_batch",
    "expected_calibration_error",
    "fit_temperature",
    "log_loss",
    "log_loss_from_logits",
    "reliability_curve",
    "score_confidence",
    "score_confidence_batch",
    "score_value",
    "score_value_batch",
    "yes_no_probability",
    "yes_no_probability_batch",
]


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class TemperatureFit:
    """The result of :func:`fit_temperature`.

    Attributes:
        temperature: The ``T`` that minimises the mean log-loss.
        loss_before: Mean log-loss at ``T = 1``.
        loss_after: Mean log-loss at ``temperature``.
        n: Number of rows.
        iterations: Steps of the search for the zero of the derivative.
        bounds: The ``(lower, upper)`` bounds the search ran in.
    """

    temperature: float
    loss_before: float
    loss_after: float
    n: int
    iterations: int
    bounds: tuple[float, float]


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of the reliability diagram: the rows whose confidence is in ``(lower, upper]``.

    The first bin also holds a confidence of exactly 0. ``mean_confidence`` and ``accuracy`` are
    ``None`` when the bin is empty.
    """

    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None

    @property
    def gap(self) -> float | None:
        """``|accuracy - mean_confidence|``, or ``None`` for an empty bin."""
        if self.accuracy is None or self.mean_confidence is None:
            return None
        return abs(self.accuracy - self.mean_confidence)


@dataclass(frozen=True)
class CalibrationReport:
    """Every metric of :func:`calibration_report`, computed in one pass.

    Attributes:
        n: Number of rows.
        accuracy: Fraction of rows whose top option is the label.
        mean_confidence: Mean of the largest probability of each row.
        ece: Expected calibration error (top-label, uniform bins closed on the right).
        mce: Largest gap of the non-empty bins.
        brier: Multi-class Brier score, in ``[0, 2]``.
        log_loss: Mean natural log-loss (``inf`` if a label has probability 0).
        bins: The reliability diagram, empty bins included.
    """

    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    mce: float
    brier: float
    log_loss: float
    bins: tuple[ReliabilityBin, ...]


TemperatureFit.__module__ = "archai_jev"
ReliabilityBin.__module__ = "archai_jev.calibration"
CalibrationReport.__module__ = "archai_jev.calibration"


# --------------------------------------------------------------------------- formulas


def calibrated_softmax_batch(
    rows: Sequence[Sequence[float]], temperature: float = 1.0
) -> list[list[float]]:
    """Temperature-scaled softmax of every row; the rows are independent.

    Args:
        rows: Rows of logits (lengths may differ).
        temperature: A finite number greater than 0.

    Returns:
        One list of probabilities per row, equal to ``calibrated_softmax(row, temperature)``.

    Raises:
        TypeError: An element is not a number (``bool``, ``str`` and ``None`` are not).
        ValueError: ``rows`` is empty, a row is empty or has a non-finite logit, or
            ``temperature`` is not a finite number greater than 0.

    Examples:
        >>> calibrated_softmax_batch([[0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        [[0.5, 0.5], [0.25, 0.25, 0.25, 0.25]]
    """
    return _core._softmax_batch(rows, temperature)


def yes_no_probability(
    logit_no: float, logit_yes: float, temperature: float = 1.0
) -> float:
    """``P(yes) = sigmoid((logit_yes - logit_no) / temperature)``, without overflow.

    Args:
        logit_no: Logit of the "no" option.
        logit_yes: Logit of the "yes" option.
        temperature: A finite number greater than 0.

    Returns:
        The probability of "yes"; swapping the logits gives ``1 - p``.

    Raises:
        ValueError: A logit is not finite, or ``temperature`` is not a finite number
            greater than 0.

    Examples:
        >>> yes_no_probability(0.0, 0.0)
        0.5
        >>> yes_no_probability(0.0, 1e300)
        1.0
    """
    return _core._yes_no_probability(logit_no, logit_yes, temperature)


def yes_no_probability_batch(
    rows: Sequence[Sequence[float]], temperature: float = 1.0
) -> list[float]:
    """:func:`yes_no_probability` for every row ``[logit_no, logit_yes]``.

    Args:
        rows: Rows of exactly two logits.
        temperature: A finite number greater than 0.

    Returns:
        The probability of "yes" for each row.

    Raises:
        TypeError: An element is not a number.
        ValueError: ``rows`` is empty, a row does not have exactly two finite logits, or
            ``temperature`` is not a finite number greater than 0.

    Examples:
        >>> yes_no_probability_batch([[0.0, 0.0], [0.0, 1e300]])
        [0.5, 1.0]
    """
    return _core._yes_no_probability_batch(rows, temperature)


def choice_confidence(
    probabilities: Sequence[float], *, sum_tolerance: float = 1e-6
) -> float:
    """Confidence of a ``Choice``: ``clamp01((max p - 1/K) / (1 - 1/K))``, 1 if ``K = 1``.

    This is the confidence that ``ask`` reports, bit for bit, for the same probabilities.

    Args:
        probabilities: One row of probabilities that sums to 1 within ``sum_tolerance``.
        sum_tolerance: How far from 1 the sum may be, in ``[0, 0.05]``.

    Returns:
        A number in ``[0, 1]``.

    Raises:
        TypeError: An element is not a number.
        ValueError: The row is empty, has a value outside ``[0, 1]`` or a sum other than 1
            (a sum of 0 is an error, not "uniform"), or ``sum_tolerance`` is out of range.

    Examples:
        >>> round(choice_confidence([0.7, 0.2, 0.1]), 12)
        0.55
        >>> choice_confidence([0.25, 0.25, 0.25, 0.25])
        0.0
    """
    return _core._choice_confidence(probabilities, sum_tolerance)


def choice_confidence_batch(
    rows: Sequence[Sequence[float]], *, sum_tolerance: float = 1e-6
) -> list[float]:
    """:func:`choice_confidence` for every row.

    Args:
        rows: Rows of probabilities (lengths may differ).
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        One confidence per row.

    Raises:
        TypeError: An element is not a number.
        ValueError: ``rows`` is empty, a row is not a valid distribution (the message says
            the row and the position), or ``sum_tolerance`` is out of range.

    Examples:
        >>> choice_confidence_batch([[1.0], [0.5, 0.5]])
        [1.0, 0.0]
    """
    return _core._choice_confidence_batch(rows, sum_tolerance)


def score_value(
    probabilities: Sequence[float], *, sum_tolerance: float = 1e-6
) -> float:
    """Expected level of a ``Score``: ``sum(i * p_i)``, levels counted from 0.

    Args:
        probabilities: One row of probabilities that sums to 1 within ``sum_tolerance``.
        sum_tolerance: How far from 1 the sum may be, in ``[0, 0.05]``.

    Returns:
        The expected level, in ``[0, L - 1]``.

    Raises:
        TypeError: An element is not a number.
        ValueError: The row is not a valid distribution, or ``sum_tolerance`` is out of range.

    Examples:
        >>> round(score_value([0.1, 0.2, 0.7]), 12)
        1.6
    """
    return _core._score_value(probabilities, sum_tolerance)


def score_value_batch(
    rows: Sequence[Sequence[float]], *, sum_tolerance: float = 1e-6
) -> list[float]:
    """:func:`score_value` for every row.

    Args:
        rows: Rows of probabilities (lengths may differ).
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        One expected level per row.

    Raises:
        TypeError: An element is not a number.
        ValueError: ``rows`` is empty, a row is not a valid distribution, or
            ``sum_tolerance`` is out of range.

    Examples:
        >>> score_value_batch([[1.0], [0.0, 1.0]])
        [0.0, 1.0]
    """
    return _core._score_value_batch(rows, sum_tolerance)


def score_confidence(
    probabilities: Sequence[float], *, sum_tolerance: float = 1e-6
) -> float:
    """Confidence of a ``Score``: ``max(0, 1 - sum(p_i |i - m|) / D)`` (Kev ``main``).

    ``m`` is the first level with the largest probability and ``D`` the same spread for the
    uniform distribution; the confidence is 1 for a single level.

    Args:
        probabilities: One row of probabilities that sums to 1 within ``sum_tolerance``.
        sum_tolerance: How far from 1 the sum may be, in ``[0, 0.05]``.

    Returns:
        A number in ``[0, 1]``.

    Raises:
        TypeError: An element is not a number.
        ValueError: The row is not a valid distribution, or ``sum_tolerance`` is out of range.

    Examples:
        >>> score_confidence([0.0, 1.0, 0.0])
        1.0
        >>> score_confidence([0.5, 0.5])
        0.0
    """
    return _core._score_confidence(probabilities, sum_tolerance)


def score_confidence_batch(
    rows: Sequence[Sequence[float]], *, sum_tolerance: float = 1e-6
) -> list[float]:
    """:func:`score_confidence` for every row.

    Args:
        rows: Rows of probabilities (lengths may differ).
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        One confidence per row.

    Raises:
        TypeError: An element is not a number.
        ValueError: ``rows`` is empty, a row is not a valid distribution, or
            ``sum_tolerance`` is out of range.

    Examples:
        >>> score_confidence_batch([[1.0], [0.5, 0.5]])
        [1.0, 0.0]
    """
    return _core._score_confidence_batch(rows, sum_tolerance)


# --------------------------------------------------------------------------- fit


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    bounds: tuple[float, float] = (0.01, 100.0),
) -> TemperatureFit:
    """Find the temperature ``T`` that minimises the log-loss of labelled logits.

    The mean log-loss ``L(T) = mean[logsumexp(z/T) - z_label/T]`` is convex in ``1/T``, so it has
    one minimum; the function finds the zero of its derivative (no grid, no optimiser). Rows may
    have different lengths: one temperature for all of them, like Kev.

    To recalibrate the answers of ``ask``, whose logits are not exposed, fit the
    **log-probabilities**: ``fit_temperature([[math.log(p) for p in row] for row in rows],
    labels).temperature`` is the temperature *relative* to the model's own one, to be multiplied
    by ``jev.model_info.temperature``.

    The function never returns a temperature at the edge of ``bounds``: that would make the
    probabilities look calibrated without any basis.

    Args:
        logits: Rows of logits, at least two options each.
        labels: For each row, the 0-based index of the correct option.
        bounds: ``(lower, upper)`` limits for ``T``, ``0 < lower < upper < inf``. Widen them
            (for example ``(1e-3, 1e4)``) if you expect an extreme temperature.

    Returns:
        A :class:`TemperatureFit`. It is not applied to anything.

    Raises:
        CalibrationError: The data do not determine a calibration inside ``bounds``: every row
            is uniform, the labels always match the top option, or the logits carry no usable
            information about the labels (or point the wrong way).
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: ``logits`` is empty, ``logits`` and ``labels`` differ in length, a row is
            empty, has a single option or a non-finite logit, a label is outside its row, or
            ``bounds`` is not valid. (``CalibrationError`` is also a ``ValueError``.)

    Examples:
        Two right answers and a wrong one, all with logits ``[1, -1]``; the optimum has the
        closed form ``2 / ln 2``:

        >>> fit = fit_temperature([[1.0, -1.0]] * 3, labels=[0, 0, 1])
        >>> round(fit.temperature, 9)
        2.885390082
        >>> fit.loss_after < fit.loss_before
        True
    """
    temperature, before, after, n, iterations = _core._fit_temperature(
        logits, labels, bounds
    )
    return TemperatureFit(
        temperature=temperature,
        loss_before=before,
        loss_after=after,
        n=n,
        iterations=iterations,
        bounds=(float(bounds[0]), float(bounds[1])),
    )


# --------------------------------------------------------------------------- metrics


def _report(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int,
    sum_tolerance: float,
) -> CalibrationReport:
    n, accuracy, mean_confidence, ece, mce, brier, loss, bins = (
        _core._calibration_report(probabilities, labels, n_bins, sum_tolerance)
    )
    return CalibrationReport(
        n=n,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        ece=ece,
        mce=mce,
        brier=brier,
        log_loss=loss,
        bins=tuple(ReliabilityBin(*b) for b in bins),
    )


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
    sum_tolerance: float = 1e-6,
) -> float:
    """Top-label expected calibration error.

    The confidence of a row is its largest probability (first maximum) and the row is correct
    when that option is the label. The bins are uniform on ``[0, 1]`` and closed on the right:
    bin ``i`` is ``(i/B, (i+1)/B]`` and a confidence goes to the smallest ``i`` with
    ``c <= (i+1)/B`` (so 0 is in the first bin). ``ECE = sum_b (n_b / N) |acc_b - conf_b|`` over
    the non-empty bins. Other libraries use other variants, so numbers may differ from theirs.

    Args:
        probabilities: Rows of probabilities that sum to 1 within ``sum_tolerance``.
        labels: The correct option of each row.
        n_bins: Number of bins, an integer in ``1..=1000``.
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        The ECE, in ``[0, 1]``.

    Raises:
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: No rows, rows and labels of different lengths, a row that is not a valid
            distribution, a label outside its row, or ``n_bins``/``sum_tolerance`` out of range.

    Examples:
        >>> rows = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
        >>> round(expected_calibration_error(rows, [0, 1, 1, 0]), 12)
        0.4
    """
    return _report(probabilities, labels, n_bins, sum_tolerance).ece


def brier_score(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    sum_tolerance: float = 1e-6,
) -> float:
    """Multi-class Brier score: the mean over rows of ``sum_k (p_k - y_k)^2`` with ``y`` one-hot.

    The range is ``[0, 2]``. With two classes it is **twice** the usual binary Brier score
    ``(p - y)^2``.

    Args:
        probabilities: Rows of probabilities that sum to 1 within ``sum_tolerance``.
        labels: The correct option of each row.
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        The Brier score.

    Raises:
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: No rows, rows and labels of different lengths, a row that is not a valid
            distribution, a label outside its row, or ``sum_tolerance`` out of range.

    Examples:
        >>> round(brier_score([[0.7, 0.3]], [1]), 12)
        0.98
    """
    return _report(probabilities, labels, 1, sum_tolerance).brier


def log_loss(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    sum_tolerance: float = 1e-6,
) -> float:
    """Mean natural log-loss ``-ln p_label``. Nothing is clipped: a zero gives ``inf``.

    Args:
        probabilities: Rows of probabilities that sum to 1 within ``sum_tolerance``.
        labels: The correct option of each row.
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        The log-loss, ``>= 0``, or ``inf`` if a label has probability exactly 0.

    Raises:
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: No rows, rows and labels of different lengths, a row that is not a valid
            distribution, a label outside its row, or ``sum_tolerance`` out of range.

    Examples:
        >>> log_loss([[1.0, 0.0]], [1])
        inf
        >>> round(log_loss([[0.5, 0.5]], [0]), 12)
        0.69314718056
    """
    return _report(probabilities, labels, 1, sum_tolerance).log_loss


def log_loss_from_logits(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float = 1.0,
) -> float:
    """Mean log-loss of ``softmax(logits / temperature)``, computed with ``logsumexp``.

    This is the function that :func:`fit_temperature` minimises. It does not underflow where
    the probabilities would.

    Args:
        logits: Rows of finite logits (a row may have a single option).
        labels: The correct option of each row.
        temperature: A finite number greater than 0.

    Returns:
        The log-loss, ``>= 0``.

    Raises:
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: No rows, rows and labels of different lengths, an empty row or a non-finite
            logit, a label outside its row, or ``temperature`` not a finite number greater
            than 0.

    Examples:
        >>> round(log_loss_from_logits([[0.0, 0.0]], [1]), 12)
        0.69314718056
    """
    return _core._log_loss_from_logits(logits, labels, temperature)


def reliability_curve(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
    sum_tolerance: float = 1e-6,
) -> list[ReliabilityBin]:
    """The data of the reliability diagram: one :class:`ReliabilityBin` per bin.

    Empty bins are included (with ``None`` values), so the result always has ``n_bins``
    elements. The bins are those of :func:`expected_calibration_error`.

    Args:
        probabilities: Rows of probabilities that sum to 1 within ``sum_tolerance``.
        labels: The correct option of each row.
        n_bins: Number of bins, an integer in ``1..=1000``.
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        The bins, from the lowest confidence to the highest.

    Raises:
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: No rows, rows and labels of different lengths, a row that is not a valid
            distribution, a label outside its row, or ``n_bins``/``sum_tolerance`` out of range.

    Examples:
        >>> bins = reliability_curve([[1.0, 0.0]], [0], n_bins=4)
        >>> [b.count for b in bins]
        [0, 0, 0, 1]
    """
    return list(_report(probabilities, labels, n_bins, sum_tolerance).bins)


def calibration_report(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
    sum_tolerance: float = 1e-6,
) -> CalibrationReport:
    """Accuracy, ECE, MCE, Brier, log-loss and the reliability bins, in one pass.

    The numbers are exactly those of the single functions (:func:`expected_calibration_error`,
    :func:`brier_score`, :func:`log_loss`).

    Args:
        probabilities: Rows of probabilities that sum to 1 within ``sum_tolerance``.
        labels: The correct option of each row.
        n_bins: Number of bins, an integer in ``1..=1000``.
        sum_tolerance: How far from 1 each sum may be, in ``[0, 0.05]``.

    Returns:
        A :class:`CalibrationReport`.

    Raises:
        TypeError: An element is not a number, or a label is not an integer.
        ValueError: No rows, rows and labels of different lengths, a row that is not a valid
            distribution, a label outside its row, or ``n_bins``/``sum_tolerance`` out of range.

    Examples:
        >>> rows = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
        >>> report = calibration_report(rows, [0, 1, 1, 0])
        >>> (round(report.ece, 12), round(report.brier, 12), report.accuracy)
        (0.4, 0.45, 0.75)
    """
    return _report(probabilities, labels, n_bins, sum_tolerance)
