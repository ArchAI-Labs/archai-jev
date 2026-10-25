"""Labelled records: reading, validation, equivalence of the forms (spec 017 §1)."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import pytest

from _evaluation_data import KEYS, LOGITS, QUESTIONS, as_dict, record, scripted
from archai_jev import (
    InvalidDatasetError,
    InvalidQuestionError,
    InvalidStateError,
    Jev,
    JevError,
    Score,
    YesNo,
)
from archai_jev.evaluation import LabelledRecord, load_dataset
from archai_jev.testing import MockScorer

SPEC_LINE = (
    '{"state": "My running shoes arrived in the wrong size.", "id": "t-17", '
    '"group": "customer-4", "questions": {'
    '"team": {"type": "choice", "instructions": "Which team should handle this?", '
    '"criteria": {"returns": "Exchanges", "billing": null}, "label": "returns"}, '
    '"urgent": {"type": "noul", "instructions": "Is this urgent?", "label": true}, '
    '"priority": {"type": "score", "instructions": "How important?", '
    '"criteria": ["low", "medium", "high"], "label": 2}}}'
)


def write(tmp_path: Path, text: str | bytes, name: str = "data.jsonl") -> Path:
    path = tmp_path / name
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


def test_spec_and_kev_examples(tmp_path: Path) -> None:
    path = write(tmp_path, "\r\n" + SPEC_LINE + "\r\n   \r\n" + SPEC_LINE + "\n")
    records = load_dataset(path)
    assert len(records) == 2
    first = records[0]
    assert first.labels == {"team": "returns", "urgent": True, "priority": 2}
    assert (first.id, first.group) == ("t-17", "customer-4")
    assert list(first.questions) == ["team", "urgent", "priority"]
    assert first.questions["team"].criteria == {"returns": "Exchanges", "billing": None}
    kev = (
        '{"state": "text", "questions": {"u": {"type": "noul", '
        '"instructions": "Urgent?", "label": false}}}'
    )
    only = load_dataset(str(write(tmp_path, kev, "kev.jsonl")))
    assert only[0].labels == {"u": False} and only[0].id is None


def test_the_error_names_the_file_and_the_line(tmp_path: Path) -> None:
    path = write(tmp_path, SPEC_LINE + "\n\n" + SPEC_LINE.replace("true", "1", 1))
    with pytest.raises(InvalidDatasetError) as info:
        load_dataset(path)
    assert str(info.value) == (
        f'{path}:3: question "urgent": the label of a yes/no question must be true '
        "or false, got 1"
    )


def test_bom_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path, b"\xef\xbb\xbf" + SPEC_LINE.encode())
    with pytest.raises(InvalidDatasetError) as info:
        load_dataset(path)
    assert str(info.value) == (
        f"{path}:1: the file starts with a byte order mark; save it as UTF-8 without BOM"
    )


def test_load_dataset_str_and_path(tmp_path: Path) -> None:
    path = write(tmp_path, SPEC_LINE)
    assert load_dataset(path) == load_dataset(str(path))


def test_missing_file_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.jsonl")
    for content in ["", "\n\n  \n"]:
        with pytest.raises(InvalidDatasetError) as info:
            load_dataset(write(tmp_path, content, "empty.jsonl"))
        assert str(info.value) == "dataset is empty; it needs at least one record"


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ('{"state": NaN}', "not valid JSON: "),
        ('{"state": "x",}', "not valid JSON: "),
        ('{"state": "x", "state": "y"}', 'the key "state" appears more than once'),
        ("[1, 2]", "a record must be a JSON object, got an array"),
        ('{"state": "x', "not valid JSON: "),
    ],
)
def test_strict_json_and_nothing_partial(
    tmp_path: Path, line: str, fragment: str
) -> None:
    path = write(tmp_path, SPEC_LINE + "\n" + line)
    with pytest.raises(InvalidDatasetError) as info:
        load_dataset(path)
    assert str(info.value).startswith(f"{path}:2: ") and fragment in str(info.value)


def test_big_integers_in_the_state_are_kept(tmp_path: Path) -> None:
    line = (
        '{"state": {"n": 123456789012345678901234567890}, "questions": '
        '{"u": {"type": "noul", "instructions": "Ok?", "label": true}}}'
    )
    records = load_dataset(write(tmp_path, line))
    assert records[0].state["n"] == 123456789012345678901234567890


def test_unknown_fields_and_missing_keys(tmp_path: Path) -> None:
    for extra in ['"_meta": {}', '"src": "x"', '"model": "kev"']:
        path = write(tmp_path, SPEC_LINE.replace('"state"', extra + ', "state"', 1))
        with pytest.raises(InvalidDatasetError, match="unsupported fields"):
            load_dataset(path)
    in_question = SPEC_LINE.replace('"label": true', '"label": true, "target": 0.7')
    with pytest.raises(InvalidDatasetError, match='unsupported fields "target"'):
        load_dataset(write(tmp_path, in_question))
    with pytest.raises(InvalidDatasetError, match='needs "state" and a non-empty'):
        load_dataset(write(tmp_path, '{"state": "x", "questions": {}}'))


def test_domain_errors_keep_their_type_with_the_line(tmp_path: Path) -> None:
    bad_state = SPEC_LINE.replace(
        '"state": "My running shoes arrived in the wrong size."', '"state": 42'
    )
    with pytest.raises(InvalidStateError, match=r"data\.jsonl:1: "):
        load_dataset(write(tmp_path, bad_state))
    bad_question = SPEC_LINE.replace(
        '"instructions": "Is this urgent?"', '"instructions": ""'
    )
    with pytest.raises(InvalidQuestionError, match=r"data\.jsonl:1: "):
        load_dataset(write(tmp_path, bad_question))


def test_labelled_record_contract() -> None:
    labels = {"team": "returns", "urgent": True, "prio": 2}
    questions = dict(QUESTIONS)
    r = LabelledRecord({"k": [1, 2]}, questions, labels, id="a", group="g")
    labels["team"] = "other"
    questions.pop("urgent")
    assert r.labels["team"] == "returns" and "urgent" in r.questions
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.id = "b"  # type: ignore[misc]
    with pytest.raises(TypeError):
        r.labels["team"] = "x"  # type: ignore[index]
    same = LabelledRecord(
        {"k": [1, 2]},
        QUESTIONS,
        {"team": "returns", "urgent": True, "prio": 2},
        id="a",
        group="g",
    )
    assert r == same and hash(r) == hash(same)
    assert r != LabelledRecord({"k": [1, 3]}, QUESTIONS, same.labels, id="a", group="g")
    assert pickle.loads(pickle.dumps(r)) == r
    assert "LabelledRecord" in repr(r)


def test_dict_and_record_give_the_same_dataset() -> None:
    rows = [
        record(i, KEYS[i % 3], i % 2 == 0, i % 3, group=f"g{i // 2}") for i in range(12)
    ]
    jev = scripted()
    from_records = jev.evaluate(rows)
    from_dicts = jev.evaluate([as_dict(r) for r in rows])
    mixed = jev.evaluate([rows[0], *[as_dict(r) for r in rows[1:]]])
    assert from_records == from_dicts == mixed
    assert from_records.dataset_fingerprint == from_dicts.dataset_fingerprint


def test_dict_conversion_is_strict() -> None:
    jev = scripted()
    good = as_dict(record(0))
    with pytest.raises(InvalidStateError, match=r"^record 0: "):
        jev.evaluate([{**good, 1: "x"}])  # type: ignore[dict-item]
    with pytest.raises(InvalidDatasetError, match=r"^record 0: not valid JSON"):
        bad = as_dict(record(0))
        bad["questions"]["urgent"]["label"] = float("nan")
        jev.evaluate([bad])
    with pytest.raises(
        InvalidDatasetError, match="a record must be a JSON object, got a set"
    ):
        jev.evaluate([{1, 2}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="iterable of LabelledRecord or dict"):
        jev.evaluate("not a dataset")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="iterable of LabelledRecord or dict"):
        jev.evaluate(good)  # type: ignore[arg-type]


def test_no_user_code_runs_during_conversion() -> None:
    class Spy(dict):  # type: ignore[type-arg]
        def items(self) -> Any:
            raise AssertionError("items() was called")

        def __iter__(self) -> Any:
            raise AssertionError("__iter__() was called")

    spy = Spy(as_dict(record(0)))
    assert scripted().evaluate([spy]).n_records == 1


@pytest.mark.parametrize(
    ("label_team", "fragment"),
    [
        (
            "Returns",
            'label "Returns" is not one of the options ["returns", "billing", "other"]',
        ),
        (1, "label 1 is not one of the options"),
        (True, "label true is not one of the options"),
        (
            (None, "label null is not one of the options")
            if False
            else ("zzz", 'label "zzz" is not one of')
        ),
    ],
)
def test_choice_labels_are_exact(label_team: Any, fragment: str) -> None:
    bad = LabelledRecord(
        "s", QUESTIONS, {"team": label_team, "urgent": True, "prio": 0}
    )
    with pytest.raises(InvalidDatasetError) as info:
        scripted().evaluate([bad])
    assert str(info.value).startswith(
        'record 0: question "team": '
    ) and fragment in str(info.value)


@pytest.mark.parametrize("value", [1, 0, "true", None, 1.0])
def test_yes_no_labels_are_only_bool(value: Any) -> None:
    bad = LabelledRecord(
        "s", QUESTIONS, {"team": "returns", "urgent": value, "prio": 0}
    )
    with pytest.raises(InvalidDatasetError, match="must be true or false"):
        scripted().evaluate([bad])


@pytest.mark.parametrize("value", [2.0, True, -1, 3, "2", None])
def test_score_labels_are_only_int_in_range(value: Any) -> None:
    bad = LabelledRecord(
        "s", QUESTIONS, {"team": "returns", "urgent": True, "prio": value}
    )
    with pytest.raises(InvalidDatasetError, match="must be an integer from 0 to 2"):
        scripted().evaluate([bad])


def test_missing_and_extra_labels_and_ids() -> None:
    jev = scripted()
    short = LabelledRecord("s", QUESTIONS, {"team": "returns", "urgent": True})
    with pytest.raises(
        InvalidDatasetError, match=r'^record 0: question "prio" has no label$'
    ):
        jev.evaluate([short])
    extra = LabelledRecord("s", QUESTIONS, {**record(0).labels, "ghost": 1})
    with pytest.raises(
        InvalidDatasetError,
        match='labels has "ghost" but questions has no such question',
    ):
        jev.evaluate([record(0), extra][1:])
    with pytest.raises(
        InvalidDatasetError, match='"id" must be a string of at most 200 characters'
    ):
        jev.evaluate([LabelledRecord("s", QUESTIONS, record(0).labels, id=3)])  # type: ignore[arg-type]
    with pytest.raises(
        InvalidDatasetError, match='"group" must be a string of at most 200 characters'
    ):
        jev.evaluate(
            [LabelledRecord("s", QUESTIONS, record(0).labels, group="x" * 201)]
        )


def test_a_name_keeps_one_type_across_the_dataset() -> None:
    a = LabelledRecord("s", {"q": YesNo("Fine?")}, {"q": True})
    b = LabelledRecord("s", {"q": Score("How?", ["a", "b"])}, {"q": 1})
    with pytest.raises(InvalidDatasetError) as info:
        scripted().evaluate([a, b])
    assert str(info.value) == (
        'question name "q" is a yes_no question at record 0 and a score question at '
        "record 1; a name must keep one type across the dataset"
    )
    # same type, different instructions: accepted
    c = LabelledRecord("s", {"q": YesNo("Another?")}, {"q": False})
    assert Jev.from_scorer(MockScorer(1)).evaluate([a, c]).n_records == 2


def test_the_dataset_is_validated_before_the_model_is_asked() -> None:
    scorer = MockScorer.scripted(LOGITS)
    jev = Jev.from_scorer(scorer)
    rows = [record(i) for i in range(5)]
    rows.append(
        LabelledRecord("s", QUESTIONS, {"team": "nope", "urgent": True, "prio": 0})
    )
    with pytest.raises(InvalidDatasetError):
        jev.evaluate(rows)
    assert scorer.calls == 0
    with pytest.raises(JevError):
        jev.evaluate([])
    assert scorer.calls == 0


def test_a_bare_carriage_return_is_json_whitespace_for_both_readers(
    tmp_path: Path,
) -> None:
    # The Rust reader splits lines on "\n" only; the Python side parses the very same bytes.
    line = SPEC_LINE.replace('"state": ', '"state":\r', 1)
    records = load_dataset(write(tmp_path, line + "\n" + line))
    assert len(records) == 2 and records[0].labels["urgent"] is True
