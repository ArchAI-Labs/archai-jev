from collections.abc import Sequence
from typing import Any, final

from typing_extensions import Self

__all__ = [
    "Fault",
    "MockScorer",
    "_Model",
    "_Question",
    "calibrated_softmax",
    "list_models",
    "load_model",
    "make_question",
    "render_state",
]

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

    def ask(
        self, state: Any, questions: Sequence[tuple[str, _Question]]
    ) -> list[tuple[Any, ...]]:
        """Ask every question about ``state``; one tuple per answer, in question order."""

    def ask_many(
        self, states: Sequence[Any], questions: Sequence[tuple[str, _Question]]
    ) -> list[list[tuple[Any, ...]]]:
        """Same questions on several states; all states are validated first."""

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
