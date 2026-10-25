"""002 - Grounding check: is each sentence of an answer supported by the source?

What you will learn
-------------------
* how to use a ``YesNo`` question as a *fact checker* for generated text,
* how to ask MANY questions about ONE state in a single ``ask`` call,
* how to turn a probability into a decision, and why the threshold is yours to choose,
* what a YesNo answer can and cannot tell you.

The use case
------------
Imagine an assistant that answers questions about a document (a "RAG" system). Language
models sometimes add details that are not in the document. A cheap, fast check: split the
answer into sentences and ask a decision model whether each sentence is *supported by the
source*. Sentences that are not supported are the ones to review or to remove.

Run it
------
    python demos/002_grounding_check.py

The default model reads at most 512 tokens per question (roughly 300 words), including the
source. A longer document must be split in chunks first: the library refuses to truncate
text silently (demo 004 shows the error).
"""

from __future__ import annotations

import textwrap

from _common import bar, section, show_download_progress, title

from archai_jev import Jev, YesNo

# The source document: the only thing the assistant is allowed to rely on.
SOURCE = (
    "Orion Labs released version 2.4 of its mapping app on 12 March. The update adds "
    "offline maps for 40 countries and reduces battery usage by improving location "
    "sampling. The app is free on Android and iOS, but offline maps require a Pro "
    "subscription costing 4 euros per month. The company is based in Lisbon and has "
    "about 60 employees."
)

# The "assistant answer", already split in sentences. The second element is OUR OWN
# label (we wrote the sentences, so we know which ones are made up). The model never
# sees it: we only use it at the end to check how the model did.
ANSWER = [
    ("Version 2.4 came out on 12 March.", True),
    ("The new offline maps cover 40 countries.", True),
    ("Battery usage went down thanks to better location sampling.", True),
    ("The Pro plan costs 4 euros a month.", True),
    ("Orion Labs is headquartered in Lisbon.", True),
    # The next ones are NOT supported by the source:
    ("The offline maps are free for all users.", False),  # contradicted
    ("Orion Labs has a team of around 600 people.", False),  # contradicted
    ("The app is available only on iOS.", False),  # contradicted
    ("The company was founded in Lisbon in 2012.", False),  # partly invented
    ("The update also adds a built-in route planner for cyclists.", False),  # invented
]


def main() -> None:
    title("002 - Grounding check")
    show_download_progress()
    jev = Jev.from_pretrained()

    # ----------------------------------------------------------------------------------
    # One question per sentence, all about the same state (the source).
    #
    # The state is the source, and each question asks "is this sentence supported by the
    # source?". We give every question a name (s0, s1, ...) to find its answer later.
    # All ten questions share the same state, so the model reads the source only once.
    # ----------------------------------------------------------------------------------
    questions = {
        f"s{i}": YesNo(f"Is this statement supported by the text? {sentence}")
        for i, (sentence, _label) in enumerate(ANSWER)
    }

    # ONE call, ten questions. It is all or nothing: you either get all ten answers or an
    # exception, never a half-filled result.
    answers = jev.ask(SOURCE, questions)

    # ----------------------------------------------------------------------------------
    # From probabilities to decisions.
    #
    # ``yes_nos[name].probability`` is P(yes) = P("the source supports this sentence").
    # The natural threshold is 0.5, but it is YOUR choice: a stricter one (say 0.9) flags
    # also the sentences the model is only moderately sure about. We use two levels.
    # ----------------------------------------------------------------------------------
    sure, unsure = 0.9, 0.5
    section("Source")
    print(textwrap.indent(textwrap.fill(SOURCE, 68), "  "))
    section("Verdict for each sentence of the assistant answer")
    print("   P(supported)                          verdict       sentence")

    flagged = 0
    agree = 0
    for i, (sentence, label) in enumerate(ANSWER):
        p = answers.yes_nos[f"s{i}"].probability
        if p >= sure:
            verdict = "supported"
        elif p >= unsure:
            verdict = "CHECK (weak)"
        else:
            verdict = "NOT SUPPORTED"
        flagged += p < unsure
        agree += (p >= unsure) == label
        print(f"  {bar(p, 14)} {p:5.1%}   {verdict:13s} {sentence}")

    print()
    print(f"  {flagged} of {len(ANSWER)} sentences are not supported by the source.")
    print(
        f"  The model agrees with our own labels on {agree} of {len(ANSWER)} sentences."
    )

    # ----------------------------------------------------------------------------------
    # What a YesNo answer cannot tell you.
    #
    # There are only two outcomes. A sentence that CONTRADICTS the source and a sentence
    # that simply is not MENTIONED in it both get a low P(supported): the model does not
    # separate them. For a fact checker that is usually what you want (both are
    # "unsupported"), but do not read "low" as "false".
    #
    # Remember also that this is a small model: it is overconfident outside the data it
    # was trained on, and these ten sentences are an easy, hand-made test. Treat the
    # output as a first filter that points a human at the doubtful sentences.
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
