"""Immutable copies of JSON-like values, and their canonical (order-sensitive) form.

Question entries (instructions, descriptions, levels) are copied when a question is built, so
mutating the caller's dict or list afterwards changes nothing. Subclasses are read by *value*:
no user-defined method is called.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze(value: Any) -> Any:
    """Deep immutable copy: dict -> read-only mapping, list/tuple -> tuple, scalars by value."""
    if isinstance(value, dict):
        return MappingProxyType(
            {str.__str__(k): freeze(v) for k, v in dict.items(value)}
        )
    if isinstance(value, list):
        return tuple(freeze(v) for v in list.__iter__(value))
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in tuple.__iter__(value))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int.__add__(value, 0)
    if isinstance(value, float):
        return float.__pos__(value)
    return value  # None


def thaw(value: Any) -> Any:
    """Plain dict/list/scalar copy of a frozen value (for ``repr`` and pickling)."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


def canon(value: Any) -> Any:
    """Hashable, order-sensitive, type-tagged form: ``1``, ``True`` and ``1.0`` all differ,
    because they render differently in a prompt."""
    if isinstance(value, Mapping):
        return ("d", tuple((k, canon(v)) for k, v in value.items()))
    if isinstance(value, tuple):
        return ("l", tuple(canon(v) for v in value))
    if isinstance(value, float):
        return ("float", repr(value))
    return (type(value).__name__, value)
