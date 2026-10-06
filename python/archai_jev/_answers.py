"""Typed answers: :class:`ChoiceAnswer`, :class:`ScoreAnswer`, :class:`YesNoAnswer`, :class:`Answers`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, NoReturn

__all__ = ["Answer", "Answers", "ChoiceAnswer", "ScoreAnswer", "YesNoAnswer"]


def _immutable(self: object, name: str, value: object) -> NoReturn:
    raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")


class ChoiceAnswer:
    """Answer to a :class:`Choice`.

    Attributes:
        value: The chosen key (the first one on an exact tie).
        index: Position of the chosen option.
        probabilities: Probability of each key, in option order; sums to 1.
        confidence: In ``[0, 1]``; ``(max p - 1/K) / (1 - 1/K)``.
        calibrated: ``False`` if the model has no declared calibration.
    """

    __slots__ = ("calibrated", "confidence", "index", "probabilities", "value")
    __match_args__ = ("value", "index", "probabilities", "confidence", "calibrated")

    value: str
    index: int
    probabilities: Mapping[str, float]
    confidence: float
    calibrated: bool

    def __init__(
        self,
        value: str,
        index: int,
        probabilities: Mapping[str, float],
        confidence: float,
        calibrated: bool,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "probabilities", MappingProxyType(dict(probabilities)))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "calibrated", calibrated)

    __setattr__ = _immutable

    def _key(self) -> tuple[Any, ...]:
        return (
            self.value,
            self.index,
            tuple(self.probabilities.items()),
            self.confidence,
            self.calibrated,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is ChoiceAnswer and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"ChoiceAnswer(value={self.value!r}, index={self.index!r}, "
            f"probabilities={dict(self.probabilities)!r}, "
            f"confidence={self.confidence!r}, calibrated={self.calibrated!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            ChoiceAnswer,
            (
                self.value,
                self.index,
                dict(self.probabilities),
                self.confidence,
                self.calibrated,
            ),
        )


class ScoreAnswer:
    """Answer to a :class:`Score`.

    Attributes:
        value: Expected level ``sum(i * p_i)``, 0-based.
        probabilities: Probability of level ``i`` at position ``i``; sums to 1.
        confidence: In ``[0, 1]``.
        calibrated: ``False`` if the model has no declared calibration.
    """

    __slots__ = ("calibrated", "confidence", "probabilities", "value")
    __match_args__ = ("value", "probabilities", "confidence", "calibrated")

    value: float
    probabilities: tuple[float, ...]
    confidence: float
    calibrated: bool

    def __init__(
        self,
        value: float,
        probabilities: tuple[float, ...],
        confidence: float,
        calibrated: bool,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "probabilities", tuple(probabilities))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "calibrated", calibrated)

    __setattr__ = _immutable

    def _key(self) -> tuple[Any, ...]:
        return (self.value, self.probabilities, self.confidence, self.calibrated)

    def __eq__(self, other: object) -> bool:
        return type(other) is ScoreAnswer and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"ScoreAnswer(value={self.value!r}, probabilities={self.probabilities!r}, "
            f"confidence={self.confidence!r}, calibrated={self.calibrated!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (ScoreAnswer, self._key())


class YesNoAnswer:
    """Answer to a :class:`YesNo`.

    Attributes:
        probability: Probability of "yes" (true). There is no confidence for yes/no.
        calibrated: ``False`` if the model has no declared calibration.
    """

    __slots__ = ("calibrated", "probability")
    __match_args__ = ("probability", "calibrated")

    probability: float
    calibrated: bool

    def __init__(self, probability: float, calibrated: bool) -> None:
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "calibrated", calibrated)

    __setattr__ = _immutable

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is YesNoAnswer
            and self.probability == other.probability
            and self.calibrated == other.calibrated
        )

    def __hash__(self) -> int:
        return hash((self.probability, self.calibrated))

    def __repr__(self) -> str:
        return (
            f"YesNoAnswer(probability={self.probability!r}, "
            f"calibrated={self.calibrated!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (YesNoAnswer, (self.probability, self.calibrated))


Answer = ChoiceAnswer | ScoreAnswer | YesNoAnswer


class Answers(Mapping[str, Answer]):
    """The answers of one request: an immutable ``Mapping`` from question name to answer, in
    question order, plus typed views.

    Attributes:
        choices: Only the :class:`ChoiceAnswer` entries.
        scores: Only the :class:`ScoreAnswer` entries.
        yes_nos: Only the :class:`YesNoAnswer` entries.
        nouls: Alias of ``yes_nos``.
    """

    __slots__ = ("_items", "choices", "scores", "yes_nos")

    _items: dict[str, Answer]
    choices: Mapping[str, ChoiceAnswer]
    scores: Mapping[str, ScoreAnswer]
    yes_nos: Mapping[str, YesNoAnswer]

    def __init__(self, items: Mapping[str, Answer]) -> None:
        data = dict(items)
        object.__setattr__(self, "_items", data)
        object.__setattr__(
            self,
            "choices",
            MappingProxyType(
                {k: v for k, v in data.items() if isinstance(v, ChoiceAnswer)}
            ),
        )
        object.__setattr__(
            self,
            "scores",
            MappingProxyType(
                {k: v for k, v in data.items() if isinstance(v, ScoreAnswer)}
            ),
        )
        object.__setattr__(
            self,
            "yes_nos",
            MappingProxyType(
                {k: v for k, v in data.items() if isinstance(v, YesNoAnswer)}
            ),
        )

    __setattr__ = _immutable

    @property
    def nouls(self) -> Mapping[str, YesNoAnswer]:
        """Alias of :attr:`yes_nos`."""
        return self.yes_nos

    def __getitem__(self, name: str) -> Answer:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"no answer named {name!r}; available: {list(self._items)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Answers) and list(self._items.items()) == list(
            other._items.items()
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Answers({self._items!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (Answers, (dict(self._items),))


def build_answers(raw: list[tuple[Any, ...]]) -> Answers:
    """Turn the tuples returned by the core into answer objects."""
    items: dict[str, Answer] = {}
    for row in raw:
        kind = row[0]
        name = row[1]
        if kind == "choice":
            _, _, value, index, probs, confidence, calibrated = row
            items[name] = ChoiceAnswer(
                value, index, dict(probs), confidence, calibrated
            )
        elif kind == "score":
            _, _, value, probs, confidence, calibrated = row
            items[name] = ScoreAnswer(value, tuple(probs), confidence, calibrated)
        else:
            _, _, probability, calibrated = row
            items[name] = YesNoAnswer(probability, calibrated)
    return Answers(items)
