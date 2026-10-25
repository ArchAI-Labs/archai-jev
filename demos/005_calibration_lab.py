"""005 - Calibration lab: what does the "temperature" do to a probability?

What you will learn
-------------------
* what calibration means: when a model says "90% sure", it should be right about 90% of
  the time,
* how ``temperature`` reshapes the probabilities WITHOUT changing the decisions,
* how to load a model with the temperature declared by its author or with your own
  (``Jev.from_pretrained(temperature=...)``),
* how to measure confidence against accuracy on a small labelled set.

The idea
--------
A language model produces raw scores called *logits*. Turning them into probabilities with
a plain softmax often gives numbers that are too extreme: the model says 99% and is right
80% of the time (it is *overconfident*). Dividing the logits by a number T > 1 before the
softmax flattens the distribution. T is the *temperature*. It never changes which answer
is the most probable, only how sure the model sounds.

Run it
------
    python demos/005_calibration_lab.py

It loads the default model three times (about 1.6 GB each time, from the local cache),
so expect a short wait at the start of each step.

A word of caution
-----------------
The numbers below come from only 36 hand-written claims. They illustrate the method, they
do not certify anything. The author of the default model measured its temperature on data
from the model's own training distribution; on other data the model can still be too sure.
"""

from __future__ import annotations

import gc

from _common import section, show_download_progress, title

from archai_jev import Jev, YesNo, calibrated_softmax

# (premise, [(claim, really supported?)]). We wrote them, so we know the right answers.
# They include a few harder ones (small arithmetic, common sense) on purpose.
BRIEFINGS = [
    (
        "The train from Milan to Rome leaves at 9:15 and takes three hours.",
        [
            ("The train arrives around noon.", True),
            ("The train goes from Milan to Rome.", True),
            ("The journey lasts three hours.", True),
            ("The train arrives in the evening.", False),
            ("The train leaves at 7:00.", False),
            ("The journey takes less than one hour.", False),
        ],
    ),
    (
        "Maria bought two apples and three oranges at the market.",
        [
            ("Maria bought five fruits.", True),
            ("Maria bought more oranges than apples.", True),
            ("Maria went to the market.", True),
            ("Maria bought four apples.", False),
            ("Maria bought only apples.", False),
            ("Maria bought fewer than four fruits.", False),
        ],
    ),
    (
        "The meeting was moved from Tuesday to Thursday because the manager was ill.",
        [
            ("The meeting will not take place on Tuesday.", True),
            ("The manager was unwell.", True),
            ("The meeting is now on Thursday.", True),
            ("The meeting was moved to Monday.", False),
            ("The manager was on holiday.", False),
            ("The meeting was cancelled.", False),
        ],
    ),
    (
        "A small bakery in Turin sells bread, focaccia and pastries and opens at six.",
        [
            ("The bakery sells bread.", True),
            ("The bakery opens early in the morning.", True),
            ("The bakery is in Italy.", True),
            ("The bakery opens at noon.", False),
            ("The bakery only sells pizza.", False),
            ("The bakery is in Spain.", False),
        ],
    ),
    (
        "The glacier lost about ten percent of its ice in the last decade.",
        [
            ("The glacier is shrinking.", True),
            ("The glacier has less ice than ten years ago.", True),
            ("Most of the ice is still there.", True),
            ("The glacier grew in the last decade.", False),
            ("The glacier lost all its ice.", False),
            ("The glacier did not change.", False),
        ],
    ),
    (
        "Tom's flight to Berlin was delayed by two hours, so he missed his train.",
        [
            ("Tom's flight did not arrive on time.", True),
            ("Tom missed a train.", True),
            ("Tom was travelling to Berlin.", True),
            ("Tom's flight arrived early.", False),
            ("Tom caught his train.", False),
            ("Tom was travelling to Paris.", False),
        ],
    ),
]
CLAIMS = [
    (premise, claim, label) for premise, items in BRIEFINGS for claim, label in items
]


def evaluate(jev: Jev) -> tuple[float, float, float, list[bool]]:
    """Ask every claim; return accuracy, mean confidence, Brier score and the decisions.

    * accuracy         share of claims where the decision (P >= 0.5) matches our label,
    * mean confidence  the average probability the model gave to the answer it chose,
    * Brier score      the mean squared distance between P(yes) and the truth (0 or 1);
                       lower is better and it rewards honest probabilities.
    """
    decisions: list[bool] = []
    confidences: list[float] = []
    squared_errors: list[float] = []
    for premise, claim, label in CLAIMS:
        question = {"claim": YesNo(f"Is this statement supported by the text? {claim}")}
        p_yes = jev.ask(premise, question).yes_nos["claim"].probability
        decisions.append(p_yes >= 0.5)
        confidences.append(max(p_yes, 1.0 - p_yes))
        squared_errors.append((p_yes - float(label)) ** 2)
    n = len(CLAIMS)
    right = sum(d == c[2] for d, c in zip(decisions, CLAIMS, strict=True))
    accuracy = right / n
    return accuracy, sum(confidences) / n, sum(squared_errors) / n, decisions


def main() -> None:
    title("005 - Calibration lab")

    # ----------------------------------------------------------------------------------
    # PART 1. The temperature on plain numbers. No model needed.
    #
    # ``calibrated_softmax(logits, temperature)`` is the very function the library uses.
    # Look at how the same two scores turn into different probabilities, while the first
    # option always stays the most probable one.
    # ----------------------------------------------------------------------------------
    section("1. The same two scores [3.0, 1.0] at different temperatures")
    for temperature in (0.5, 1.0, 1.75, 4.0, 10.0):
        probabilities = calibrated_softmax([3.0, 1.0], temperature)
        print(f"  T = {temperature:5.2f}   P(first) = {probabilities[0]:6.1%}")
    print("  (T < 1 sharpens, T > 1 flattens; the order never changes)")

    # ----------------------------------------------------------------------------------
    # PART 2. The same model with three different temperatures.
    #
    # * ``from_pretrained()``: the temperature DECLARED by the model's author (stored in
    #   its manifest). The answers say ``calibrated=True``.
    # * ``from_pretrained(temperature=T)``: YOUR calibration, for example one you measured
    #   on your own data. It replaces the declared one.
    # ----------------------------------------------------------------------------------
    section("2. The default model at three temperatures")
    show_download_progress()
    print(f"  {len(CLAIMS)} claims, asked at each temperature")
    print()
    print("  temperature          accuracy  mean confidence   gap   Brier")

    # ``temperature=None`` (the default) means "use the one the model declares".
    setups: list[tuple[str, float | None]] = [
        ("declared by author", None),
        ("your own: T = 1.0", 1.0),
        ("your own: T = 4.0", 4.0),
    ]
    all_decisions: list[list[bool]] = []
    for label, own_temperature in setups:
        jev = Jev.from_pretrained(temperature=own_temperature)
        t = jev.model_info.temperature
        accuracy, confidence, brier, decisions = evaluate(jev)
        all_decisions.append(decisions)
        gap = confidence - accuracy  # > 0: more sure than right (overconfident)
        print(
            f"  {label:19s} (T={t:4.2f})  {accuracy:5.1%}  {confidence:11.1%}"
            f"   {gap:+6.1%}  {brier:.3f}"
        )
        del jev  # free the model before loading the next one
        gc.collect()

    # ----------------------------------------------------------------------------------
    # What to look at
    #
    # 1. The decisions are the same at every temperature, so accuracy does not move: the
    #    temperature changes how sure the model SOUNDS, not what it decides.
    # 2. "gap" = mean confidence - accuracy. Positive means overconfident, negative means
    #    underconfident, near zero means well calibrated ON THIS SET.
    # 3. The Brier score measures the quality of the probabilities themselves.
    #
    # The right temperature depends on YOUR data. The library will let you fit one on a
    # labelled set (a planned feature); until then you can do it by hand as above.
    # ----------------------------------------------------------------------------------
    same = all(d == all_decisions[0] for d in all_decisions)
    print()
    print(f"  Same decisions at every temperature: {same}")
    print(
        "  Gap > 0 = overconfident, gap < 0 = underconfident. Only 36 claims: a sketch."
    )


if __name__ == "__main__":
    main()
