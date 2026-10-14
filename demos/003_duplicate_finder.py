"""003 - Duplicate finder: group support tickets that say the same thing.

What you will learn
-------------------
* how to use a ``YesNo`` question to compare two texts,
* how to turn pairwise probabilities into groups (a tiny clustering),
* why comparing every pair is expensive (the cost grows with the SQUARE of the list),
* that similarity answers are not perfectly symmetric.

The use case
------------
A bug tracker receives tickets written by different people about the same problem.
Finding duplicates by hand is boring; finding them by keywords misses "dark mode" vs "night
theme". A decision model can read two titles and say whether they mean the same.

Run it
------
    python demos/003_duplicate_finder.py

It asks about every pair of the 12 tickets below: 66 questions. It prints the time it took
on YOUR machine; nothing in this file claims a speed.
"""

from __future__ import annotations

import time

from _common import section, show_download_progress, title

from archai_jev import Jev, YesNo

# (group we have in mind, ticket title). The group is OUR label, used only at the end to
# check the result: "A", "B" and "C" are three real duplicate groups, "-" are tickets that
# only LOOK similar to a group (same topic, different problem): the hard cases.
TICKETS = [
    ("A", "Cannot log in with my Google account"),
    ("A", "Google sign-in fails with an error"),
    ("A", "Login via Google does not work"),
    ("B", "Export to CSV produces an empty file"),
    ("B", "The CSV export is empty"),
    ("B", "Exported CSV file has no rows"),
    ("C", "Please add a dark mode"),
    ("C", "Dark theme support requested"),
    ("C", "Need a night mode option"),
    ("-", "Login with email and password fails"),  # looks like A, it is not
    ("-", "The CSV import rejects my file"),  # looks like B, it is not
    ("-", "Invoice shows the wrong VAT number"),
]

THRESHOLD = 0.5  # P(same meaning) at or above this counts as "duplicate"


def same_meaning(other: str) -> YesNo:
    """The question "does the state mean the same as ``other``?"."""
    return YesNo(f"Does this ticket mean the same as: {other}")


def find(parent: list[int], i: int) -> int:
    """Union-find: the representative of the group that ticket ``i`` belongs to."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # shorten the path as we go
        i = parent[i]
    return i


def main() -> None:
    title("003 - Duplicate finder")
    show_download_progress()
    jev = Jev.from_pretrained()
    n = len(TICKETS)

    # ----------------------------------------------------------------------------------
    # Ask about every pair (i, j) with i < j.
    #
    # The state is the FIRST text and the question carries the SECOND text. So for
    # ticket i we send ONE ask() with a question per ticket j > i, and the model reads
    # ticket i only once for all of them. That is n - 1 calls and n * (n - 1) / 2
    # questions in total.
    # ----------------------------------------------------------------------------------
    probability: dict[tuple[int, int], float] = {}
    started = time.perf_counter()
    for i in range(n - 1):
        questions = {f"t{j}": same_meaning(TICKETS[j][1]) for j in range(i + 1, n)}
        answers = jev.ask(TICKETS[i][1], questions)
        for j in range(i + 1, n):
            probability[(i, j)] = answers.yes_nos[f"t{j}"].probability
    seconds = time.perf_counter() - started

    # ----------------------------------------------------------------------------------
    # From pairs to groups: "single linkage".
    #
    # Every pair at or above the threshold joins its two tickets in the same group. If A
    # matches B and B matches C, all three end up together even if A and C did not match
    # directly. That forgives one wrong "no" inside a group, but one wrong "yes" between
    # two groups merges them: choose a stricter threshold if you want to avoid that.
    # ----------------------------------------------------------------------------------
    parent = list(range(n))
    for (i, j), p in probability.items():
        if p >= THRESHOLD:
            parent[find(parent, i)] = find(parent, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(parent, i), []).append(i)

    section(f"Groups found (threshold {THRESHOLD})")
    for members in groups.values():
        if len(members) > 1:
            print(f"  duplicates ({len(members)} tickets):")
            for i in members:
                print(f"     - {TICKETS[i][1]}")
    alone = [TICKETS[m[0]][1] for m in groups.values() if len(m) == 1]
    print(f"  alone ({len(alone)} tickets): " + "; ".join(alone))

    # ----------------------------------------------------------------------------------
    # How good was it? We know the truth (we wrote the tickets), the model does not.
    # A pair is "really the same" when both tickets are in the same real group A/B/C.
    # ----------------------------------------------------------------------------------
    section("Pairs where the model disagrees with our labels")
    disagree = 0
    for (i, j), p in probability.items():
        really_same = TICKETS[i][0] == TICKETS[j][0] and TICKETS[i][0] != "-"
        if (p >= THRESHOLD) != really_same:
            disagree += 1
            verdict = "said same" if p >= THRESHOLD else "said different"
            print(f"  {verdict} (P={p:.0%}): {TICKETS[i][1]!r} / {TICKETS[j][1]!r}")
    if not disagree:
        print("  none")
    print(f"  -> {len(probability) - disagree} of {len(probability)} pairs agree.")

    # ----------------------------------------------------------------------------------
    # The cost of comparing everything with everything.
    #
    # n tickets mean n * (n - 1) / 2 pairs. The numbers below are plain arithmetic on the
    # time YOU just measured, a projection and not a benchmark. The way out in real life
    # is to avoid most pairs: compare only tickets that share a keyword or an embedding
    # neighbourhood, and use the decision model only on those candidates.
    # ----------------------------------------------------------------------------------
    section("What it cost")
    per_question = seconds / len(probability)
    print(
        f"  {len(probability)} questions in {seconds:.1f} s ({per_question * 1000:.0f} ms each)"
    )
    for tickets in (100, 1000):
        pairs = tickets * (tickets - 1) // 2
        hours = pairs * per_question / 3600
        print(
            f"  {tickets:5d} tickets -> {pairs:7d} pairs -> about {hours:5.1f} hours at this rate"
        )

    # ----------------------------------------------------------------------------------
    # A detail: similarity is not perfectly symmetric. The model reads the two texts in a
    # fixed order, so asking (A, B) and (B, A) can give slightly different numbers. We
    # asked each pair once. If you need symmetry, ask both directions and average.
    # ----------------------------------------------------------------------------------
    section("Is it symmetric? (the first pair, both directions)")
    first, second = TICKETS[0][1], TICKETS[1][1]
    forward = probability[(0, 1)]
    backward = jev.ask(second, {"q": same_meaning(first)})
    print(f"  A -> B: {forward:.1%}    B -> A: {backward.yes_nos['q'].probability:.1%}")


if __name__ == "__main__":
    main()
