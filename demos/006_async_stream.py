"""006 - Async: keep your program responsive while the model decides.

What you will learn
-------------------
* ``ask_many``: the same question about MANY texts in one call,
* ``aask`` / ``aask_many``: the ``async`` versions, for ``asyncio`` programs,
* how to prove that the event loop is not blocked while the model works,
* that one ``Jev`` handles ONE request at a time: concurrent calls wait their turn,
* that cancelling a task really stops its work (and so does ``Ctrl-C`` on a plain ``ask``).

The use case
------------
Customer messages arrive in a stream. For each one we want to know "does this customer want
to cancel their subscription?" (so we can send it to the retention team). The question is
always the same, only the text changes: exactly what ``ask_many`` is for.

(Worth noticing: a ``YesNo`` question is a flexible classifier. The message is the state and
the question is the statement we want to test: ANY statement can be the question.)

Run it
------
    python demos/006_async_stream.py
"""

from __future__ import annotations

import asyncio
import itertools
import time

from _common import bar, section, show_download_progress, title

from archai_jev import Jev, YesNo

# The statement we test on every message. Built once, reused for every message.
WANTS_TO_CANCEL = YesNo("The customer wants to cancel their subscription.")

# (message, does the customer really want to cancel?). The label is OUR OWN, used only to
# count the model's mistakes. It includes tricky cases: "cancel" about a meeting, and a
# customer who wants to pause (not to cancel).
MESSAGES = [
    ("I don't want to pay for this anymore, please stop my plan.", True),
    ("Please close my account, I'm leaving the service.", True),
    ("How do I terminate my membership?", True),
    ("I'd like to unsubscribe from the premium plan.", True),
    ("Cancel my subscription effective immediately.", True),
    ("I am switching to a competitor, so please end my contract.", True),
    ("Stop billing me, I no longer use the app.", True),
    ("I want out of this subscription.", True),
    ("How can I upgrade to the premium plan?", False),
    ("My invoice for March is missing.", False),
    ("The app crashes when I open the camera.", False),
    ("Can I change my billing address?", False),
    ("Thanks, the support team was very helpful!", False),
    ("Do you offer a student discount?", False),
    ("I forgot my password.", False),
    ("What is included in the Pro plan?", False),
    ("Can I pause my subscription for a month?", False),  # tricky: pause, not cancel
    (
        "Cancel the meeting scheduled for Friday.",
        False,
    ),  # tricky: "cancel", other meaning
]


async def heartbeat(ticks: list[float], period: float = 0.05) -> None:
    """A task that does nothing but wake up every 50 ms and note the time.

    If the event loop were blocked by the model, these wake-ups would be missing. We run
    it next to the model call and count them afterwards.
    """
    while True:
        await asyncio.sleep(period)
        ticks.append(time.perf_counter())


async def main_async(jev: Jev) -> None:
    # ----------------------------------------------------------------------------------
    # PART 2. ``aask_many``: the same call, but awaitable.
    #
    # Under the hood the work runs in a worker thread (the Rust engine releases Python's
    # global lock while it computes), so the event loop stays free to run other tasks.
    # ----------------------------------------------------------------------------------
    section("2. The async version, with a heartbeat running next to it")
    ticks: list[float] = []
    beat = asyncio.create_task(heartbeat(ticks))
    started = time.perf_counter()
    await jev.aask_many([m for m, _ in MESSAGES], {"cancel": WANTS_TO_CANCEL})
    seconds = time.perf_counter() - started
    beat.cancel()
    longest_gap = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    print(f"  the model worked for {seconds:.1f} s")
    print(f"  meanwhile the heartbeat ticked {len(ticks)} times (it asks for 50 ms)")
    print(f"  longest silence between two ticks: {longest_gap * 1000:.0f} ms")
    print("  -> the loop kept running during the whole call, so it could serve")
    print("     other requests meanwhile (timers on Windows are coarse: a tick can")
    print("     be a few tens of ms late)")

    # ----------------------------------------------------------------------------------
    # PART 3. Several concurrent ``aask`` calls on the same ``Jev``.
    #
    # A ``Jev`` can be shared safely between tasks and threads, but it computes ONE request
    # at a time: the others wait their turn. The printout shows each call finishing about
    # one "model time" after the previous one, even though we started all together.
    # ----------------------------------------------------------------------------------
    section("3. Three concurrent aask() calls share one model")
    started = time.perf_counter()

    async def one(index: int, text: str) -> tuple[int, float]:
        await jev.aask(text, {"cancel": WANTS_TO_CANCEL})
        return index, time.perf_counter() - started

    results = await asyncio.gather(*(one(i, MESSAGES[i][0]) for i in range(3)))
    for index, finished in sorted(results, key=lambda r: r[1]):
        print(f"  call {index} finished after {finished:5.2f} s")

    # ----------------------------------------------------------------------------------
    # PART 4. Cancelling really stops the work.
    #
    # Eight requests are waiting for the model and we change our mind after a third of a
    # second. A request that has not started never starts; the one that is running stops at
    # its next row. So the model is free again almost at once: a new request does not have
    # to wait for the eight we gave up. (This is also what ``asyncio.wait_for(...,
    # timeout=...)`` does when the timeout expires.)
    # ----------------------------------------------------------------------------------
    section("4. Cancelling stops the work")
    tasks = [
        asyncio.create_task(jev.aask(MESSAGES[i][0], {"cancel": WANTS_TO_CANCEL}))
        for i in range(8)
    ]
    await asyncio.sleep(0.3)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    started = time.perf_counter()
    await jev.aask(MESSAGES[0][0], {"cancel": WANTS_TO_CANCEL})
    print("  eight requests cancelled after 0.3 s; a new one then took")
    per_request = max(finished for _, finished in results) / len(results)
    print(
        f"  {time.perf_counter() - started:.2f} s (eight finished requests would have taken"
        f" about {8 * per_request:.1f} s, at {per_request:.2f} s each as in part 3)"
    )


def main() -> None:
    title("006 - Async decisions")
    show_download_progress()
    jev = Jev.from_pretrained()

    # ----------------------------------------------------------------------------------
    # PART 1. ``ask_many``: one question, many states.
    #
    # It returns a list of ``Answers`` in the same order as the states. All states are
    # validated BEFORE any computation, so a bad message at position 17 fails fast.
    # ----------------------------------------------------------------------------------
    section("1. ask_many: the same question about 18 messages")
    started = time.perf_counter()
    batch = jev.ask_many([m for m, _ in MESSAGES], {"cancel": WANTS_TO_CANCEL})
    seconds = time.perf_counter() - started
    mistakes = 0
    for (message, wants), answers in zip(MESSAGES, batch, strict=True):
        p = answers.yes_nos["cancel"].probability
        mistakes += (p >= 0.5) != wants
        mark = "CANCEL" if p >= 0.5 else "      "
        flag = "" if (p >= 0.5) == wants else "   <-- not what we expected"
        print(f"  {bar(p, 10)} {p:5.1%} {mark}  {message}{flag}")
    print(
        f"  {len(MESSAGES) - mistakes} of {len(MESSAGES)} as we expected, in {seconds:.1f} s"
    )

    asyncio.run(main_async(jev))


if __name__ == "__main__":
    main()
