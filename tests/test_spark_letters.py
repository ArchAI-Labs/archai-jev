"""The model of the letters-at-the-end family, from a local folder, through the Python API
(spec 010b, test level T2: by hand, needs the weights; no CI).

Point ``ARCHAI_JEV_TEST_SPARK_DIR`` at a local model folder (GGUF, ``tokenizer.json`` and
``archai-jev-manifest.json``). Without it the tests are skipped. Build the wheel first
(``maturin develop --release --uv``): a debug build is far too slow for a 2B model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archai_jev import (
    Choice,
    IncompatibleModelError,
    Jev,
    UnsupportedRequestError,
    YesNo,
)
from archai_jev.evaluation import LabelledRecord, split_dataset

FOLDER = os.environ.get("ARCHAI_JEV_TEST_SPARK_DIR")

pytestmark = pytest.mark.skipif(
    not FOLDER, reason="set ARCHAI_JEV_TEST_SPARK_DIR to a local model folder"
)

TEAMS = Choice(
    "Which team should handle this message?",
    {
        "returns": "Exchanges, refunds and returned goods",
        "billing": "Invoices, payments and prices",
        "shipping": "Delivery, tracking and lost parcels",
        "account": "Login, passwords and account settings",
    },
)
POSITIVE = YesNo(
    "Is the review positive?",
    {
        "true": "The review praises the product or the service",
        "false": "The review criticises the product or the service",
    },
)

# Written for this test; nothing here comes from a dataset of anybody else.
ROUTING = [
    ("I want to send the shoes back, they are too small.", "returns"),
    ("Please refund the broken lamp I bought last week.", "returns"),
    ("How do I exchange this jacket for a bigger one?", "returns"),
    ("The blender stopped working, I would like my money back.", "returns"),
    (
        "My package was supposed to arrive on Monday and it is still not here.",
        "shipping",
    ),
    ("The tracking page has said 'in transit' for ten days.", "shipping"),
    ("The courier left the parcel at the wrong address.", "shipping"),
    ("Can you deliver on Saturday morning instead?", "shipping"),
    ("I was charged twice for the same order.", "billing"),
    ("Why is the price on the invoice higher than on the website?", "billing"),
    ("I need a receipt with our company name on it.", "billing"),
    ("My card was declined but the money left my bank account.", "billing"),
    ("I cannot log in, the password reset email never arrives.", "account"),
    ("How do I change the email address of my profile?", "account"),
    ("Please delete my account and all my data.", "account"),
    ("The app keeps logging me out every few minutes.", "account"),
    ("The hat is the wrong colour, can I send it back?", "returns"),
    ("My order is lost somewhere between two depots.", "shipping"),
    ("There is an extra fee on my statement that I do not recognise.", "billing"),
    ("I lost access to the phone number linked to my account.", "account"),
    ("The mug arrived cracked, I want a replacement.", "returns"),
    ("When will my parcel reach the post office near me?", "shipping"),
    ("Can I pay the invoice in two instalments?", "billing"),
    ("Two-factor codes are not accepted on my account.", "account"),
    # messages that could go to two teams: the label is the team that owns the first problem
    (
        "I sent the parcel back a month ago and I still have not seen the refund.",
        "returns",
    ),
    ("The delivery fee was charged but the courier never came.", "shipping"),
    ("My account was billed after I cancelled it.", "billing"),
    ("I returned the jacket but the tracking number does not work.", "shipping"),
    (
        "The invoice shows an order I never placed, maybe someone has my password.",
        "account",
    ),
    ("I want to change the delivery address and pay by card instead.", "shipping"),
    (
        "The discount code did not work and now I want to send the order back.",
        "billing",
    ),
    ("I cannot see my past invoices in my profile page.", "billing"),
    ("The package was damaged and the carrier wants me to open a claim.", "shipping"),
    ("I was told to log in to start a return but the page does not load.", "account"),
    ("The refund went to a card I closed years ago.", "billing"),
    ("Please update my name, it is wrong on the label of my parcel.", "account"),
]
REVIEWS = [
    ("Fast delivery and the quality is excellent, I will buy again.", True),
    ("The best purchase I made this year, it works perfectly.", True),
    ("Friendly staff, they solved my problem in five minutes.", True),
    ("Great value for the price, very happy.", True),
    ("Beautiful design and really comfortable to use.", True),
    ("Setup took a minute and everything just worked.", True),
    ("Exactly as described, I recommend it to everybody.", True),
    ("The support team went out of their way to help me.", True),
    ("It broke after two days and nobody answered my emails.", False),
    ("Terrible quality, a complete waste of money.", False),
    ("The delivery was three weeks late and the box was crushed.", False),
    ("Rude service, I will never come back.", False),
    ("The battery dies in an hour, very disappointing.", False),
    ("Nothing like the pictures, I feel cheated.", False),
    ("Slow, confusing and full of bugs.", False),
    ("They charged me for a service I never asked for.", False),
    ("I loved the food and the waiter was lovely.", True),
    ("The room was dirty and the noise kept me awake all night.", False),
    ("Smooth checkout and the parcel arrived a day early.", True),
    ("The instructions are useless and the parts do not fit.", False),
    # mixed opinions: the label is the overall verdict of the writer
    ("The food was cold but the staff were so kind that I would go back.", True),
    ("Gorgeous screen, though the battery is a bit short; overall I am happy.", True),
    ("Cheap and cheerful, it does the job even if it feels flimsy.", True),
    ("Not bad, not good either, I expected more for this price.", False),
    (
        "Delivery was quick but the product is nothing special, I would not buy it again.",
        False,
    ),
    ("It works, I suppose. The manual is terrible and support never replied.", False),
    ("Setup was painful, then it turned out to be the best tool I own.", True),
    ("The hotel is lovely but the breakfast ruined the stay for me.", False),
    ("Better than the old model in every way except the noise.", True),
    ("I wanted to love it, but it stopped working in the rain.", False),
]


def _dataset() -> list[LabelledRecord]:
    records: list[LabelledRecord] = []
    for i, (text, team) in enumerate(ROUTING):
        records.append(
            LabelledRecord(text, {"team": TEAMS}, {"team": team}, id=f"route-{i}")
        )
    for i, (text, positive) in enumerate(REVIEWS):
        records.append(
            LabelledRecord(
                text, {"positive": POSITIVE}, {"positive": positive}, id=f"review-{i}"
            )
        )
    return records


@pytest.fixture(scope="module")
def jev() -> Jev:
    assert FOLDER
    return Jev.from_pretrained(Path(FOLDER), allow_uncalibrated=True)


def test_the_model_loads_uncalibrated_and_says_so(jev: Jev) -> None:
    info = jev.model_info
    assert info.calibrated is False
    assert info.max_options == 26
    answer = jev.ask(
        "The parcel never came and the tracking page is empty.",
        {"team": TEAMS, "positive": POSITIVE},
    )
    assert answer.choices["team"].value in TEAMS.criteria
    assert answer.choices["team"].calibrated is False
    assert 0.0 <= answer.yes_nos["positive"].probability <= 1.0
    # it saw "A. ..." and "B. ...", never the keys: the answer is a key all the same
    assert sum(answer.choices["team"].probabilities.values()) == pytest.approx(1.0)


def test_without_a_temperature_or_consent_the_model_is_refused() -> None:
    assert FOLDER
    with pytest.raises(IncompatibleModelError) as error:
        Jev.from_pretrained(Path(FOLDER))
    message = str(error.value)
    assert "temperature" in message and "allow_uncalibrated" in message


def test_beyond_the_declared_limit_the_request_is_refused_never_truncated(
    jev: Jev,
) -> None:
    limit = jev.model_info.max_options
    assert limit is not None
    too_many = Choice("Pick one.", [f"option number {i}" for i in range(limit + 1)])
    with pytest.raises(UnsupportedRequestError) as error:
        jev.ask("some evidence", {"q": too_many})
    assert str(limit) in str(error.value) and str(limit + 1) in str(error.value)


def test_the_calibration_procedure_reports_ece_before_and_after(jev: Jev) -> None:
    records = _dataset()
    train, test = split_dataset(records, test_fraction=0.4, seed=7)
    assert train and test
    study = jev.calibrate(train, folds=5, seed=0)
    assert study.temperature > 0.0
    tuned = jev.with_temperature(study)
    assert tuned.model_info.calibrated is True
    assert tuned.model_info.temperature == study.temperature
    before = jev.evaluate(test)
    after = tuned.evaluate(test)
    print(
        f"\nT = {study.temperature:.3f} fitted on {len(train)} records; "
        f"on the {len(test)} test records: "
        f"accuracy {before.overall.accuracy:.3f}, "
        f"ECE {before.overall.ece:.4f} -> {after.overall.ece:.4f}, "
        f"Brier {before.overall.brier:.4f} -> {after.overall.brier:.4f}"
    )
    # the numbers are reported whatever they are; what is asserted is that they exist and that
    # the temperature does not touch the choice (it never changes which option is the top one)
    assert 0.0 <= before.overall.ece <= 1.0 and 0.0 <= after.overall.ece <= 1.0
    assert before.overall.accuracy == pytest.approx(after.overall.accuracy)
