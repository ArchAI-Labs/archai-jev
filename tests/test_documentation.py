"""The documented examples tell the truth about the default model.

Two promises are checked without the model file:

* the examples of the README that call ``ask`` are valid for the default model: only ``Choice``
  and ``YesNo`` questions, with no more options than the model scores (otherwise a reader who
  copies the example gets an ``UnsupportedRequestError``);
* no example promises the case of the model card ("Ignore previous instructions..." is *safe*
  with probability 1.00), which the model does not do (spike S5).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from archai_jev import Choice, Jev, Score, YesNo, list_models
from archai_jev.testing import MockScorer

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# The sentence of the card, and any wording of it that a reader could take as the model's skill.
CARD_EXAMPLE = re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE)
INJECTION_PROMISE = re.compile(
    r"(ignore|disregard)\s+(all\s+|any\s+)?(previous|prior)\s+instructions",
    re.IGNORECASE,
)


def _default_max_options() -> int:
    (default,) = [m for m in list_models() if m.is_default]
    assert default.max_options is not None
    return default.max_options


def _kind_of(question: Choice | YesNo | Score) -> str | None:
    """The kind of a question if the default model answers it, else ``None``."""
    if isinstance(question, YesNo):
        return "yes_no"
    if (
        isinstance(question, Choice)
        and len(question.criteria) <= _default_max_options()
    ):
        return "choice"
    return None


class _CheckedJev:
    """A ``Jev`` that answers with the mock scorer but, like the default model, refuses what the
    model does not score (and remembers which kinds of question the examples used)."""

    def __init__(self) -> None:
        self._jev = Jev.from_scorer(MockScorer())
        self.used: list[str] = []

    @property
    def model_info(self) -> Any:
        return self._jev.model_info

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        questions = kwargs.get("questions", args[1] if len(args) > 1 else {})
        for name, question in questions.items():
            kind = _kind_of(question)
            assert kind is not None, f"README example: question {name!r} is refused"
            self.used.append(kind)
        return self._jev.ask(*args, **kwargs)


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.DOTALL | re.MULTILINE)


def test_the_readme_examples_run_and_are_valid_for_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = _CheckedJev()
    monkeypatch.setattr(
        Jev, "from_pretrained", classmethod(lambda cls, *a, **k: checked)
    )
    blocks = [b for b in _python_blocks(README.read_text("utf-8")) if ".ask(" in b]
    assert len(blocks) >= 2, "the README must show the default model answering"
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    assert {"choice", "yes_no"} <= set(checked.used)


def test_the_checker_itself_refuses_what_the_default_model_refuses() -> None:
    # Guards the test above: it must not be able to pass for any question.
    limit = _default_max_options()
    assert _kind_of(Choice("Which team?", ["returns", "billing"])) == "choice"
    assert _kind_of(YesNo("Is it urgent?")) == "yes_no"
    assert _kind_of(Choice("Which?", [f"k{i}" for i in range(limit + 1)])) is None
    assert _kind_of(Score("How polite?", ["low", "high"])) is None


def _documented_files() -> list[Path]:
    package = ROOT / "python" / "archai_jev"
    return [README, *sorted(package.glob("*.py")), *sorted(package.glob("*.pyi"))]


def test_the_card_example_is_not_promised_by_the_readme_or_the_docstrings() -> None:
    files = _documented_files()
    assert README in files and len(files) > 3
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not INJECTION_PROMISE.search(
            text
        ), f"{path.name} promises the card example"


def test_the_card_example_is_not_in_the_demos_either() -> None:
    demos = sorted((ROOT / "demos").glob("*.py"))
    assert demos
    for path in demos:
        text = path.read_text(encoding="utf-8")
        assert not CARD_EXAMPLE.search(text), f"{path.name} uses the card example"
