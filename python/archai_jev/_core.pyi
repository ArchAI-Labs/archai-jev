from collections.abc import Sequence
from typing import Any, final

from typing_extensions import Self

__all__ = [
    "Fault",
    "MockScorer",
    "_CancelToken",
    "_Dataset",
    "_Model",
    "_Question",
    "_calibration_report",
    "_check_evaluate_options",
    "_choice_confidence",
    "_choice_confidence_batch",
    "_dataset_from_items",
    "_dataset_from_jsonl",
    "_fit_temperature",
    "_log_loss_from_logits",
    "_score_confidence",
    "_score_confidence_batch",
    "_score_value",
    "_score_value_batch",
    "_softmax_batch",
    "_split",
    "_yes_no_probability",
    "_yes_no_probability_batch",
    "calibrated_softmax",
    "list_models",
    "load_model",
    "make_question",
    "render_state",
]

@final
class _CancelToken:
    """A token another thread sets to stop a call between two rows."""

    def __init__(self) -> None: ...
    def set(self) -> None:
        """Ask the call that holds this token to stop at its next row."""

    def is_set(self) -> bool:
        """Whether ``set`` was called."""

@final
class _Question:
    """A validated question. Opaque: the public classes of ``archai_jev`` wrap it."""

@final
class Fault:
    """A deliberate failure injected into a ``MockScorer``."""

    @staticmethod
    def nan(question: str, option: int) -> Fault:
        """NaN at (``question``, ``option``)."""

    @staticmethod
    def pos_inf(question: str, option: int) -> Fault:
        """+infinity at (``question``, ``option``)."""

    @staticmethod
    def neg_inf(question: str, option: int) -> Fault:
        """-infinity at (``question``, ``option``)."""

    @staticmethod
    def wrong_option_count(question: str, count: int) -> Fault:
        """``count`` logits for ``question`` instead of one per option."""

    @staticmethod
    def wrong_question_count(count: int) -> Fault:
        """Logits for ``count`` questions instead of one per question."""

@final
class MockScorer:
    """The deterministic fake scorer. It is not a model."""

    def __new__(
        cls,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        calibrated: bool = False,
        latency: float = 0.0,
    ) -> Self: ...
    @staticmethod
    def scripted(
        logits: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
        calibrated: bool = False,
        latency: float = 0.0,
    ) -> MockScorer:
        """A scorer that returns exactly these logits, whatever they are."""

    def with_fault(self, fault: Fault) -> MockScorer:
        """A new scorer with one more injected fault; the original is unchanged."""

    @property
    def calls(self) -> int:
        """How many times this scorer has been called."""

    @property
    def temperature(self) -> float:
        """The temperature this scorer declares."""

    @property
    def calibrated(self) -> bool:
        """Whether this scorer declares a calibration."""

@final
class _Model:
    """A model ready to answer: wraps a scorer."""

    @staticmethod
    def from_mock(scorer: MockScorer) -> _Model:
        """A model backed by the mock scorer."""

    def cache_info(self) -> tuple[int, int, int, int, int, int, int, int]:
        """(limit_bytes, bytes, entries, hits, misses, evictions, tokens_reused, tokens_decoded)."""

    def clear_cache(self) -> None:
        """Forget every saved prefix; never changes a result."""

    def ask(
        self,
        state: Any,
        questions: Sequence[tuple[str, _Question]],
        cancel: _CancelToken | None = None,
    ) -> list[tuple[Any, ...]]:
        """Ask every question about ``state``; one tuple per answer, in question order.

        ``cancel`` stops the call between two rows when another thread sets it.
        """

    def ask_many(
        self,
        states: Sequence[Any],
        questions: Sequence[tuple[str, _Question]],
        cancel: _CancelToken | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Same questions on several states, one after the other; all states are validated first.

        Each item is the list of answer tuples of a state or, with ``return_exceptions``, the
        exception object of a state whose calculation failed.
        """

    def evaluate(
        self,
        dataset: _Dataset,
        n_bins: object,
        bootstrap_samples: object,
        seed: object,
        cancel: _CancelToken | None = None,
    ) -> Any:
        """Evaluate on a validated dataset; plain nested tuples (see ``_evaluation.py``)."""

    def calibrate(
        self,
        dataset: _Dataset,
        folds: object,
        seed: object,
        question_types: Sequence[str] | None,
        bounds: tuple[float, float],
        cancel: _CancelToken | None = None,
    ) -> Any:
        """Study the calibration on a validated dataset; plain nested tuples."""

    def with_temperature(
        self, temperature: float, provenance: Sequence[int] | None = None
    ) -> _Model:
        """The same model with this temperature, as a calibration given by the user."""

@final
class _Dataset:
    """A validated labelled dataset. Opaque."""

    def __len__(self) -> int: ...
    @property
    def fingerprint(self) -> str:
        """16 hexadecimal digits."""

    @property
    def n_questions(self) -> int: ...
    @property
    def n_groups(self) -> int: ...

def _dataset_from_items(items: Sequence[Any]) -> _Dataset:
    """Validate records: tuples made from ``LabelledRecord`` objects, or ``dict``s."""

def _dataset_from_jsonl(data: bytes, name: str) -> _Dataset:
    """Validate the bytes of a JSONL file: all of it, or an error. ``name`` is shown in messages."""

def _check_evaluate_options(
    n_bins: object, bootstrap_samples: object, seed: object
) -> None:
    """Raise ``ValueError`` if an option of ``evaluate`` is out of range."""

def _split(
    dataset: _Dataset, test_fraction: float, seed: object
) -> tuple[list[int], list[int]]:
    """Indices of the train and test records of a split by group."""

def load_model(
    kind: str,
    value: str | None,
    revision: str | None,
    device: str,
    dtype: str | None,
    temperature: float | None,
    allow_uncalibrated: bool,
    cache_dir: str | None,
    offline: bool,
    manifest: str | None,
    prefix_cache_mib: int | None,
) -> tuple[_Model, dict[str, Any]]:
    """Load a model: ``kind`` is "default", "path" or "str"; returns the model and its facts."""

def list_models() -> list[dict[str, Any]]:
    """The models of the registry as plain dictionaries."""

def make_question(kind: str, instructions: Any, criteria: Any) -> _Question:
    """Build and validate a question (``kind``: "choice", "score" or "yes_no")."""

def render_state(state: Any) -> str:
    """The canonical text of a state (what the model will read)."""

def calibrated_softmax(
    logits: Sequence[float], temperature: float = 1.0
) -> list[float]:
    """Convert raw option scores (logits) into calibrated probabilities using a temperature-scaled softmax.

    A temperature > 1 flattens the distribution (less confident), a temperature < 1
    sharpens it (more confident).

    Raises:
        ValueError: If ``logits`` is empty or not finite, or if ``temperature`` is not
            a finite number > 0.
    """

def _softmax_batch(
    rows: Sequence[Sequence[float]], temperature: float = 1.0
) -> list[list[float]]: ...
def _yes_no_probability(
    logit_no: float, logit_yes: float, temperature: float = 1.0
) -> float: ...
def _yes_no_probability_batch(
    rows: Sequence[Sequence[float]], temperature: float = 1.0
) -> list[float]: ...
def _choice_confidence(
    probabilities: Sequence[float], sum_tolerance: float = 1e-06
) -> float: ...
def _choice_confidence_batch(
    rows: Sequence[Sequence[float]], sum_tolerance: float = 1e-06
) -> list[float]: ...
def _score_value(
    probabilities: Sequence[float], sum_tolerance: float = 1e-06
) -> float: ...
def _score_value_batch(
    rows: Sequence[Sequence[float]], sum_tolerance: float = 1e-06
) -> list[float]: ...
def _score_confidence(
    probabilities: Sequence[float], sum_tolerance: float = 1e-06
) -> float: ...
def _score_confidence_batch(
    rows: Sequence[Sequence[float]], sum_tolerance: float = 1e-06
) -> list[float]: ...
def _fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    bounds: tuple[float, float],
) -> tuple[float, float, float, int, int]: ...
def _calibration_report(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: object,
    sum_tolerance: float,
) -> tuple[
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    list[tuple[float, float, int, float | None, float | None]],
]: ...
def _log_loss_from_logits(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float = 1.0,
) -> float: ...
