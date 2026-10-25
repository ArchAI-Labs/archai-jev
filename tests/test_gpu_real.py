"""The default model on a real GPU, through the public API (spec 011, level T2-GPU).

By hand, on a machine with a GPU and a build that has its backend (``maturin develop --features
cuda``, or a wheel). Needs the model in a cache folder and names the device:

    ARCHAI_JEV_TEST_CACHE=<folder>  ARCHAI_JEV_TEST_DEVICE=cuda:0  python -m pytest tests/test_gpu_real.py

* ``ARCHAI_JEV_TEST_DLL_DIR``: on Windows, the folder with the DLLs of the backend (the ``bin`` of
  the CUDA toolkit, the ``Bin`` of the Vulkan SDK); Python does not look in ``PATH`` for them.
* ``ARCHAI_JEV_TEST_CPU_ANSWERS``: a file written by ``python tests/test_gpu_real.py --write
  FILE`` with the **CPU wheel** (a build without a GPU backend). It lets the test that
  ``device="cpu"`` in a GPU build is bit-identical compare against the other build.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("ARCHAI_JEV_TEST_DLL_DIR") and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.environ["ARCHAI_JEV_TEST_DLL_DIR"])

from archai_jev import Answers, ChoiceAnswer, Jev, YesNoAnswer
from archai_jev.evaluation import load_dataset

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "data" / "v2-golden" / "golden.jsonl"
CACHE = os.environ.get("ARCHAI_JEV_TEST_CACHE")
DEVICE = os.environ.get("ARCHAI_JEV_TEST_DEVICE")
CPU_ANSWERS = os.environ.get("ARCHAI_JEV_TEST_CPU_ANSWERS")

#: The rows of the tolerance table for the default model (``src/models/tolerance.rs``, in
#: probability at the declared temperature); the Rust tests pin the numbers.
ROWS = {"cuda": 0.097, "vulkan": 0.057}

pytestmark = pytest.mark.skipif(
    not CACHE,
    reason="set ARCHAI_JEV_TEST_CACHE to a cache folder that holds the default model",
)

Request = tuple[str, Any, Any]


def golden_requests() -> list[Request]:
    """The 77 requests of the v2 golden as (id, state, questions), through the public loader."""
    out: list[Request] = []
    with tempfile.TemporaryDirectory() as folder:
        for k, line in enumerate(GOLDEN.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            req = json.loads(json.dumps(row["request"]))
            for q in req["questions"].values():
                # the loader wants a label; it is not used here
                q["label"] = next(iter(q["criteria"])) if "criteria" in q else True
            req["id"] = row["id"]
            path = Path(folder) / f"r{k}.jsonl"
            path.write_text(json.dumps(req) + "\n", encoding="utf-8", newline="\n")
            (rec,) = load_dataset(path)
            out.append((row["id"], row["request"]["state"], rec.questions))
    return out


def numbers(answers: Answers) -> list[float]:
    """Every probability of a request, in question order (a yes/no gives its one number)."""
    out: list[float] = []
    for answer in answers.values():
        if isinstance(answer, ChoiceAnswer):
            out.extend(answer.probabilities.values())
        elif isinstance(answer, YesNoAnswer):
            out.append(answer.probability)
        else:
            raise AssertionError(f"unexpected answer {answer!r}")
    return out


def answer_all(jev: Jev, requests: list[Request]) -> dict[str, list[float]]:
    return {rid: numbers(jev.ask(state, qs)) for rid, state, qs in requests}


def load(device: str) -> Jev:
    assert CACHE is not None
    return Jev.from_pretrained(offline=True, cache_dir=CACHE, device=device)


@pytest.fixture(scope="module")
def requests() -> list[Request]:
    return golden_requests()


@pytest.fixture(scope="module")
def gpu() -> str:
    if not DEVICE:
        pytest.skip("set ARCHAI_JEV_TEST_DEVICE (for example cuda:0)")
    return DEVICE


@pytest.fixture(scope="module")
def on_gpu(gpu: str) -> Jev:
    return load(gpu)


@pytest.fixture(scope="module")
def on_cpu() -> Jev:
    return load("cpu")


def test_selfcheck_on_the_device(gpu: str, on_gpu: Jev) -> None:
    # imported here because the CPU wheel of the other build does not have it
    from archai_jev import available_devices

    # reaching here means the self-check passed on that device with the tolerance row of its
    # backend; the information says where the model runs
    assert on_gpu.model_info.device == gpu
    assert on_gpu.model_info.calibrated
    assert gpu in {d.name for d in available_devices()}


def test_calibration_on_the_device(
    gpu: str, on_gpu: Jev, on_cpu: Jev, requests: list[Request]
) -> None:
    # the calibrated answers on the device stay within the row of its backend of the ones on the
    # CPU (both are within a row of the reference), and are proper probabilities
    row = ROWS[gpu.split(":")[0]]
    worst = 0.0
    for rid, state, qs in requests:
        a, b = on_gpu.ask(state, qs), on_cpu.ask(state, qs)
        assert all(x.calibrated for x in a.values()), rid
        got, want = numbers(a), numbers(b)
        assert len(got) == len(want), rid
        assert all(0.0 <= p <= 1.0 for p in got), rid
        worst = max(worst, *(abs(x - y) for x, y in zip(got, want, strict=True)))
    # the two gaps to the reference add up in the worst case: twice the row is a hard ceiling
    assert worst <= 2 * row, worst


def test_repeatability(on_gpu: Jev, requests: list[Request]) -> None:
    for rid, state, qs in requests[:6]:
        first = on_gpu.ask(state, qs)
        for _ in range(19):
            assert on_gpu.ask(state, qs) == first, rid


def test_sixteen_threads_get_the_answers_of_one(
    on_gpu: Jev, requests: list[Request]
) -> None:
    one = answer_all(on_gpu, requests)
    with ThreadPoolExecutor(16) as pool:
        many = list(
            pool.map(lambda r: (r[0], numbers(on_gpu.ask(r[1], r[2]))), requests)
        )
    assert dict(many) == one


def test_cpu_device_is_bit_identical_in_a_gpu_build(
    on_cpu: Jev, requests: list[Request]
) -> None:
    if not CPU_ANSWERS:
        pytest.skip("set ARCHAI_JEV_TEST_CPU_ANSWERS to the file of the CPU wheel")
    expected = json.loads(Path(CPU_ANSWERS).read_text(encoding="utf-8"))
    got = answer_all(on_cpu, requests)
    assert set(got) == set(expected)
    different = [k for k in got if got[k] != expected[k]]
    assert different == []


def _gpu_used_mib() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(out.splitlines()[0])


def _child() -> None:
    """Load on the CPU, answer twenty requests, say what the GPU holds (run in a fresh process)."""
    requests = golden_requests()
    before = _gpu_used_mib()
    jev = load("cpu")
    jev.ask(requests[0][1], requests[0][2])
    first = _gpu_used_mib()
    for _, state, qs in requests[1:21]:
        jev.ask(state, qs)
    print(
        "CHILD "
        + json.dumps({"before": before, "first": first, "last": _gpu_used_mib()})
    )


def test_cpu_device_leaves_the_gpu_idle() -> None:
    if not shutil.which("nvidia-smi"):
        pytest.skip("needs nvidia-smi")
    run = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(x for x in run.stdout.splitlines() if x.startswith("CHILD "))
    mib = json.loads(line[len("CHILD ") :])
    # a build with a GPU backend starts the driver and creates a context even on the CPU: at
    # most 150 MiB (the model and its cache are never put there), and it does not grow
    assert mib["first"] - mib["before"] <= 150, mib
    assert mib["last"] - mib["first"] <= 32, mib


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _child()
    elif len(sys.argv) == 3 and sys.argv[1] == "--write":
        Path(sys.argv[2]).write_text(
            json.dumps(answer_all(load("cpu"), golden_requests())), encoding="utf-8"
        )
    else:
        raise SystemExit(__doc__)
