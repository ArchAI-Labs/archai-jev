"""The three question types: :class:`Choice`, :class:`Score` and :class:`YesNo`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, NoReturn, cast

from . import _core
from ._values import canon, freeze, thaw

__all__ = ["Choice", "Noul", "Score", "YesNo"]


def _immutable(self: object, name: str, value: object) -> NoReturn:
    raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")


class Choice:
    """Pick one of several options.

    Args:
        instructions: What the model has to decide: a string (or a dict/list for structured text).
        criteria: ``{key: description or None}`` in prompt order, or just a list of keys.
            1 to 255 options, keys unique and not blank.

    Raises:
        InvalidQuestionError: If anything is invalid. A question that is invalid cannot exist.

    Examples:
        >>> q = Choice("Which team?", {"returns": "Exchanges, refunds", "billing": None})
        >>> list(q.criteria)
        ['returns', 'billing']
    """

    __slots__ = ("_criteria", "_instructions", "_q")
    __match_args__ = ("instructions", "criteria")

    _q: Any
    _instructions: Any
    _criteria: Mapping[str, Any]

    def __init__(
        self,
        instructions: Any,
        criteria: Mapping[str, Any] | Sequence[str],
    ) -> None:
        q = _core.make_question("choice", instructions, criteria)
        if isinstance(criteria, dict):
            items = {k: freeze(v) for k, v in dict.items(criteria)}
        else:
            items = {str.__str__(k): None for k in criteria}
        object.__setattr__(self, "_q", q)
        object.__setattr__(self, "_instructions", freeze(instructions))
        object.__setattr__(self, "_criteria", MappingProxyType(items))

    def __init_subclass__(cls) -> None:
        raise TypeError(f"type {Choice.__name__!r} is not an acceptable base type")

    __setattr__ = _immutable

    @property
    def instructions(self) -> Any:
        """The instructions, as given (read-only copy)."""
        return self._instructions

    @property
    def criteria(self) -> Mapping[str, Any]:
        """Read-only ``{key: description or None}`` in prompt order."""
        return self._criteria

    def _key(self) -> tuple[Any, ...]:
        return (canon(self._instructions), canon(self._criteria))

    def __eq__(self, other: object) -> bool:
        return type(other) is Choice and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(("Choice", self._key()))

    def __repr__(self) -> str:
        return (
            f"Choice(instructions={thaw(self._instructions)!r}, "
            f"criteria={thaw(self._criteria)!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (Choice, (thaw(self._instructions), thaw(self._criteria)))


class Score:
    """Pick a level on an ordered scale; the answer is the expected level.

    Args:
        instructions: What the model has to decide.
        criteria: The levels from lowest to highest, a list or tuple (1 to 255, not blank, all
            different). A plain ``str`` is rejected (it would be read one character at a time).

    Examples:
        >>> Score("How urgent is it?", ["low", "medium", "high"]).criteria
        ('low', 'medium', 'high')
    """

    __slots__ = ("_criteria", "_instructions", "_q")
    __match_args__ = ("instructions", "criteria")

    _q: Any
    _instructions: Any
    _criteria: tuple[Any, ...]

    def __init__(self, instructions: Any, criteria: Sequence[Any]) -> None:
        q = _core.make_question("score", instructions, criteria)
        object.__setattr__(self, "_q", q)
        object.__setattr__(self, "_instructions", freeze(instructions))
        object.__setattr__(self, "_criteria", freeze(list(criteria)))

    def __init_subclass__(cls) -> None:
        raise TypeError(f"type {Score.__name__!r} is not an acceptable base type")

    __setattr__ = _immutable

    @property
    def instructions(self) -> Any:
        """The instructions, as given (read-only copy)."""
        return self._instructions

    @property
    def criteria(self) -> tuple[Any, ...]:
        """The levels, lowest first."""
        return self._criteria

    def _key(self) -> tuple[Any, ...]:
        return (canon(self._instructions), canon(self._criteria))

    def __eq__(self, other: object) -> bool:
        return type(other) is Score and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(("Score", self._key()))

    def __repr__(self) -> str:
        return (
            f"Score(instructions={thaw(self._instructions)!r}, "
            f"criteria={thaw(self._criteria)!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (Score, (thaw(self._instructions), thaw(self._criteria)))


class YesNo:
    """A yes/no question; the answer is the probability of "yes". ``Noul`` is an alias.

    Args:
        instructions: What the model has to decide.
        criteria: Optional ``{"true": description, "false": description}``; no other keys.

    Examples:
        >>> YesNo("Is the customer angry?").criteria
        mappingproxy({})
    """

    __slots__ = ("_criteria", "_instructions", "_q")
    __match_args__ = ("instructions", "criteria")

    _q: Any
    _instructions: Any
    _criteria: Mapping[str, Any]

    def __init__(
        self, instructions: Any, criteria: Mapping[str, Any] | None = None
    ) -> None:
        q = _core.make_question("yes_no", instructions, criteria)
        items: dict[str, Any] = {}
        if criteria is not None:
            items = {
                str.__str__(k): freeze(v)
                for k, v in dict.items(cast("dict[str, Any]", criteria))
                if v is not None
            }
        object.__setattr__(self, "_q", q)
        object.__setattr__(self, "_instructions", freeze(instructions))
        object.__setattr__(self, "_criteria", MappingProxyType(items))

    def __init_subclass__(cls) -> None:
        raise TypeError(f"type {YesNo.__name__!r} is not an acceptable base type")

    __setattr__ = _immutable

    @property
    def instructions(self) -> Any:
        """The instructions, as given (read-only copy)."""
        return self._instructions

    @property
    def criteria(self) -> Mapping[str, Any]:
        """Read-only mapping with the descriptions that were given."""
        return self._criteria

    def _key(self) -> tuple[Any, ...]:
        return (canon(self._instructions), canon(self._criteria))

    def __eq__(self, other: object) -> bool:
        return type(other) is YesNo and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(("YesNo", self._key()))

    def __repr__(self) -> str:
        return (
            f"YesNo(instructions={thaw(self._instructions)!r}, "
            f"criteria={thaw(self._criteria)!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (YesNo, (thaw(self._instructions), thaw(self._criteria)))


Noul = YesNo
