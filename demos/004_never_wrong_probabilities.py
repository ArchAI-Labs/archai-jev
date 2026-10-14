"""004 - "Never wrong probabilities": the errors archai-jev raises ON PURPOSE.

What you will learn
-------------------
* the principle: if the library cannot guarantee that a probability really is the
  model's own, it RAISES an error instead of returning a number,
* every kind of error you can meet, with an example that provokes it,
* how to catch them: one base class (``JevError``) and specific subclasses,
* that there is no switch to turn the checks off.

Why this matters
----------------
A wrong probability returned silently is the worst failure of a decision model: your
program trusts it and acts on it. So the library prefers to stop. Prompts that are too
long are refused (never truncated), requests the model does not score are refused (never
"adapted"), and a model that does not pass its self-check does not load.

Run it
------
    python demos/004_never_wrong_probabilities.py

The first two parts need no model at all and finish instantly. The third part loads the
default model (about 1.6 GB, downloaded once).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

from _common import section, show_download_progress, title

from archai_jev import (
    Choice,
    IncompatibleModelError,
    InvalidQuestionError,
    InvalidStateError,
    Jev,
    JevError,
    ModelDownloadError,
    NumericalError,
    Score,
    UnsupportedRequestError,
    YesNo,
)
from archai_jev.testing import Fault, MockScorer

unexpected: list[str] = []  # anything that does not behave as this demo documents


def attempt(
    label: str, action: Callable[[], object], expected: type[Exception]
) -> None:
    """Run ``action``, which must raise ``expected``, and print what happened.

    This is also the pattern to use in your code: wrap the call in ``try`` and catch the
    class you can handle (or the base class ``JevError`` to catch them all).
    """
    try:
        action()
    except expected as error:
        first_line = str(error).splitlines()[0]
        print(f"  [ok] {label}")
        print(f"       {type(error).__name__}: {first_line[:96]}")
    except Exception as error:  # a different error than the documented one
        unexpected.append(label)
        print(f"  [!!] {label}: expected {expected.__name__}, got {error!r}")
    else:
        unexpected.append(label)
        print(f"  [!!] {label}: no error was raised")


def main() -> None:
    title("004 - Never wrong probabilities: the errors, one by one")

    # ----------------------------------------------------------------------------------
    # PART 1. Errors when LOADING a model. No model is needed to see them.
    # ----------------------------------------------------------------------------------
    section("1. Loading errors (nothing is downloaded)")

    with tempfile.TemporaryDirectory() as folder:
        # offline=True means "never touch the network". With an empty cache the model
        # files are missing, and the library says so instead of guessing.
        attempt(
            "offline with an empty cache",
            lambda: Jev.from_pretrained(cache_dir=folder, offline=True),
            ModelDownloadError,
        )

        # A model of your own is described by a manifest file. A folder without one
        # cannot be loaded...
        attempt(
            "folder without a manifest",
            lambda: Jev.from_pretrained(Path(folder)),
            IncompatibleModelError,
        )

        # ...and a manifest is never completed with "reasonable" defaults: whatever
        # determines the probabilities (family, template, tokenizer, calibration...) must
        # be written down.
        (Path(folder) / "archai-jev-manifest.json").write_text("{}", encoding="utf-8")
        attempt(
            "empty manifest",
            lambda: Jev.from_pretrained(Path(folder)),
            IncompatibleModelError,
        )

    attempt(
        "unknown model name",
        lambda: Jev.from_pretrained("nobody/nothing", offline=True),
        IncompatibleModelError,
    )

    # Misuse of the API is NOT a JevError: it is a plain TypeError / ValueError, like in
    # any Python library. JevError is reserved for problems with content (states,
    # questions, models, numbers).
    attempt("calling Jev() directly", lambda: Jev(), TypeError)
    attempt(
        "temperature must be positive",
        lambda: Jev.from_pretrained(temperature=-1.0, offline=True),
        ValueError,
    )

    # ----------------------------------------------------------------------------------
    # PART 2. Numerical errors. We provoke them with the test double of archai_jev.testing
    # (demo 007 explains it): a fake scorer that we can make misbehave on purpose.
    #
    # Even when a REAL model misbehaves (a bug, a corrupted file, an unsupported CPU) the
    # library checks the numbers before returning them: finite logits, one per option,
    # probabilities in [0, 1] that sum to 1.
    # ----------------------------------------------------------------------------------
    section("2. Numerical errors (a fake scorer, no model needed)")
    question = {"q": Choice("Which team?", ["billing", "support", "sales"])}
    faults = {
        "a NaN logit": Fault.nan("q", 0),
        "an infinite logit": Fault.pos_inf("q", 1),
        "the wrong number of logits": Fault.wrong_option_count("q", 5),
        "logits for the wrong number of questions": Fault.wrong_question_count(3),
    }
    for label, fault in faults.items():
        broken = Jev.from_scorer(MockScorer(seed=1).with_fault(fault))
        attempt(label, partial(broken.ask, "hello", question), NumericalError)

    # ----------------------------------------------------------------------------------
    # PART 3. Errors in a REQUEST, with the real model. Here we need to load it.
    # ----------------------------------------------------------------------------------
    section("3. Request errors (the real model)")
    show_download_progress()
    jev = Jev.from_pretrained()
    yes_no = YesNo("Is the text about cooking?")
    limit = jev.model_info.max_options or 0

    # Errors found BEFORE any computation, in the objects you build:
    attempt(
        "a question with duplicate options",
        lambda: Choice("Which one?", ["a", "a"]),
        InvalidQuestionError,
    )
    attempt("a question without instructions", lambda: YesNo(""), InvalidQuestionError)
    attempt(
        "an ask() with no questions", lambda: jev.ask("hello", {}), InvalidQuestionError
    )
    attempt(
        "a state that is not JSON-like",
        lambda: jev.ask(object(), {"q": yes_no}),
        InvalidStateError,
    )
    attempt(
        "a state with NaN inside",
        lambda: jev.ask({"price": float("nan")}, {"q": yes_no}),
        InvalidStateError,
    )

    # Errors because THIS model cannot answer THIS request faithfully. The request is
    # valid in itself; the model just does not score it. Each message names the limit
    # and the value you gave. The limits are in ``jev.model_info``.
    attempt(
        f"more than {limit} options (the model's limit)",
        lambda: jev.ask(
            "hello", {"q": Choice("Which number?", [f"n{i}" for i in range(limit + 1)])}
        ),
        UnsupportedRequestError,
    )
    attempt(
        "an option key that is too long",
        lambda: jev.ask(
            "hello", {"q": Choice("Which one?", ["a" * 40, "b"])}  # many tokens
        ),
        UnsupportedRequestError,
    )
    attempt(
        "a Score question (this model does not score levels)",
        lambda: jev.ask("hello", {"q": Score("How polite?", ["low", "high"])}),
        UnsupportedRequestError,
    )

    # The prompt is longer than the tokens the model accepts (``model_info.max_context``).
    # A "helpful" library would cut the text and answer anyway, with probabilities that
    # no longer describe the text you sent. This one refuses and tells you the numbers.
    attempt(
        "a text longer than the model's context",
        lambda: jev.ask("word " * 1200, {"q": yes_no}),
        UnsupportedRequestError,
    )

    # ----------------------------------------------------------------------------------
    # The family tree. Every content error is a JevError, and most also inherit from a
    # standard exception, so an existing ``except ValueError`` keeps working.
    # ----------------------------------------------------------------------------------
    section("4. The family tree (and how to catch everything)")
    for error_class in JevError.__subclasses__():
        others = [b.__name__ for b in error_class.__bases__ if b is not JevError]
        print(f"  {error_class.__name__:26s} is also a {', '.join(others)}")
    print()
    print("  try:")
    print("      answers = jev.ask(text, questions)")
    print("  except archai_jev.JevError as error:   # any problem with content")
    print("      ...handle it, log it, ask a human...")

    print()
    if unexpected:
        print(f"{len(unexpected)} case(s) did not behave as documented: {unexpected}")
        raise SystemExit(1)
    print("All errors were raised as documented. There is no switch to disable them.")


if __name__ == "__main__":
    main()
