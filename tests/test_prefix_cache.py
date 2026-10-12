"""The memory for reusing what questions share: ``prefix_cache_mib`` and its environment variable.

These tests run without a model file: they check what is decided before anything is loaded.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from archai_jev import CacheInfo, Choice, Jev, ModelDownloadError


@pytest.mark.parametrize("value", [True, False, 1.5, "128", [128], b"1"])
def test_param_must_be_an_int_or_none(value: object) -> None:
    with pytest.raises(TypeError, match="prefix_cache_mib"):
        Jev.from_pretrained(prefix_cache_mib=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, -128])
def test_param_cannot_be_negative(value: int) -> None:
    with pytest.raises(ValueError, match="prefix_cache_mib"):
        Jev.from_pretrained(prefix_cache_mib=value)


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", "128MiB"])
def test_env_must_be_a_whole_number_of_mib(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHAI_JEV_PREFIX_CACHE_MIB", value)
    with pytest.raises(ValueError, match="ARCHAI_JEV_PREFIX_CACHE_MIB"):
        Jev.from_pretrained(cache_dir=tmp_path, offline=True)


def test_env_read_once_param_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bad variable is never looked at when the argument is given: the load goes on and stops
    # later, at the empty offline cache, with the error that has nothing to do with the variable.
    monkeypatch.setenv("ARCHAI_JEV_PREFIX_CACHE_MIB", "abc")
    for value in (0, 64):
        with pytest.raises(ModelDownloadError):
            Jev.from_pretrained(
                cache_dir=tmp_path, offline=True, prefix_cache_mib=value
            )


@pytest.mark.parametrize("value", ["0", "128", "4096", " 7 "])
def test_a_good_variable_is_accepted(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHAI_JEV_PREFIX_CACHE_MIB", value)
    with pytest.raises(ModelDownloadError):
        Jev.from_pretrained(cache_dir=tmp_path, offline=True)


def test_an_empty_variable_means_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHAI_JEV_PREFIX_CACHE_MIB", "")
    with pytest.raises(ModelDownloadError):
        Jev.from_pretrained(cache_dir=tmp_path, offline=True)


def test_a_size_that_overflows_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too large"):
        Jev.from_pretrained(cache_dir=tmp_path, offline=True, prefix_cache_mib=2**62)


def _mock_jev() -> Jev:
    from archai_jev.testing import MockScorer

    return Jev.from_scorer(MockScorer())


def test_cache_info_is_zero_without_a_cache() -> None:
    info = _mock_jev().cache_info()
    assert info == CacheInfo(
        limit_bytes=0,
        bytes=0,
        entries=0,
        hits=0,
        misses=0,
        evictions=0,
        tokens_reused=0,
        tokens_decoded=0,
    )


def test_cache_info_is_frozen_and_hashable() -> None:
    info = _mock_jev().cache_info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.hits = 1  # type: ignore[misc]
    assert hash(info) == hash(_mock_jev().cache_info())
    assert {info: 1}[_mock_jev().cache_info()] == 1


def test_clear_cache_is_harmless_and_does_not_change_answers() -> None:
    jev = _mock_jev()
    questions = {"q": Choice("Which?", ["a", "b"])}
    before = jev.ask("some state", questions)
    jev.clear_cache()
    assert jev.ask("some state", questions) == before


def test_ask_signatures_are_unchanged() -> None:
    assert list(inspect.signature(Jev.ask).parameters) == ["self", "state", "questions"]
    # `ask_many` gained one keyword-only parameter with a default (spec 008): additive.
    assert list(inspect.signature(Jev.ask_many).parameters) == [
        "self",
        "states",
        "questions",
        "return_exceptions",
    ]
    assert (
        inspect.signature(Jev.ask_many).parameters["return_exceptions"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert list(inspect.signature(Jev.cache_info).parameters) == ["self"]
    assert list(inspect.signature(Jev.clear_cache).parameters) == ["self"]


def test_threads_and_asyncio() -> None:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    jev = _mock_jev()
    questions = {"q": Choice("Which?", ["a", "b"])}
    expected = jev.ask("some state", questions)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: jev.ask("some state", questions), range(200)))
    assert all(r == expected for r in results)

    async def many() -> list[object]:
        return list(
            await asyncio.gather(
                *(jev.aask("some state", questions) for _ in range(50))
            )
        )

    assert all(r == expected for r in asyncio.run(many()))
    jev.clear_cache()
    assert jev.cache_info().entries == 0
