"""The vectors recorded by the engine on a quantised file, and the defects they must notice
(spec 009, AC-9, AC-11 and AC-12; level T2).

A quantised variant is checked against numbers the engine itself recorded on the very file
(``selfcheck.reference = "engine"``). That is only worth something if the check *fails* when
something is wrong: a prompt built differently, a different order of the options, a score moved by
one, a file of another dtype. By hand, needs the model folder that
``benches/quantization/make_v2_manifest.py --engine-vectors`` writes:

    $env:ARCHAI_JEV_TEST_Q4_FOLDER = "<folder with archai-jev-manifest.json, model.gguf, tokenizer.json>"
    $env:ARCHAI_JEV_TEST_OTHER_GGUF = "<a Q6_K or Q8_0 GGUF of the same model>"    # optional
    .venv\\Scripts\\python -m pytest tests/test_quantized_vectors.py -s
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from archai_jev import IncompatibleModelError, Jev, ModelVerificationError

FOLDER = os.environ.get("ARCHAI_JEV_TEST_Q4_FOLDER")
OTHER_GGUF = os.environ.get("ARCHAI_JEV_TEST_OTHER_GGUF")

pytestmark = pytest.mark.skipif(
    not FOLDER,
    reason="needs ARCHAI_JEV_TEST_Q4_FOLDER (a model folder with engine-recorded vectors)",
)

Manifest = dict[str, Any]


def _manifest() -> Manifest:
    assert FOLDER is not None
    text = (Path(FOLDER) / "archai-jev-manifest.json").read_text(encoding="utf-8")
    data: Manifest = json.loads(text)
    return data


def _vectors(m: Manifest) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = m["variants"][0]["selfcheck"]["vectors"]
    return vectors


def _softmax(scores: list[float], temperature: float) -> list[float]:
    z = [s / temperature for s in scores]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    return [v / sum(e) for v in e]


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _folder_with(tmp_path: Path, manifest: Manifest, gguf: Path | None = None) -> Path:
    """A model folder like FOLDER (files linked, not copied) whose manifest is `manifest`."""
    assert FOLDER is not None
    source = Path(FOLDER)
    folder = tmp_path / "model"
    folder.mkdir()
    _link_or_copy(gguf or source / "model.gguf", folder / "model.gguf")
    _link_or_copy(source / "tokenizer.json", folder / "tokenizer.json")
    (folder / "archai-jev-manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return folder


def _load(folder: Path, cache: Path) -> Jev:
    return Jev.from_pretrained(folder, cache_dir=cache, offline=True)


def test_the_recorded_vectors_pass(tmp_path: Path) -> None:
    manifest = _manifest()
    assert manifest["variants"][0]["selfcheck"]["reference"] == "engine"
    assert 8 <= len(_vectors(manifest)) <= 12
    jev = _load(_folder_with(tmp_path, manifest), tmp_path / "cache")
    assert jev.model_info.dtype == manifest["variants"][0]["dtype"]


def test_the_check_does_not_depend_on_the_cache_setting(tmp_path: Path) -> None:
    # the numbers of a K-quant move by a few hundredths of probability between the row in one
    # decode and the row in the segments of its cuts: the check always uses the second, so a
    # model loaded with the cache off is verified against the same vectors
    assert FOLDER is not None
    jev = Jev.from_pretrained(
        _folder_with(tmp_path, _manifest()),
        cache_dir=tmp_path / "cache",
        offline=True,
        prefix_cache_mib=0,
    )
    assert jev.model_info.dtype == "q4_k_m"


def _shift_a_token(m: Manifest) -> None:
    # a template that builds another prompt: the ids differ at one place
    for v in _vectors(m):
        v["expected"]["input_ids"][5] += 1


def _drop_the_last_id(m: Manifest) -> None:
    for v in _vectors(m):
        v["expected"]["input_ids"].pop()


def _reverse_the_options(m: Manifest) -> None:
    # the order of the targets: the scores of the options in the other order
    for v in _vectors(m):
        for q in v["expected"]["questions"].values():
            q["logits"].reverse()
            q["probabilities"].reverse()


def _add_one_to_the_first_target(m: Manifest) -> None:
    t = m["variants"][0]["calibration"]["temperature"]
    for v in _vectors(m):
        for q in v["expected"]["questions"].values():
            q["logits"][0] += 1.0
            q["probabilities"] = _softmax(q["logits"], t)


def _a_slightly_different_score(m: Manifest) -> None:
    # 0.3 on one score of every vector: more than the tolerance of an engine row allows
    t = m["variants"][0]["calibration"]["temperature"]
    for v in _vectors(m):
        for q in v["expected"]["questions"].values():
            q["logits"][-1] -= 0.3 * t
            q["probabilities"] = _softmax(q["logits"], t)


MUTANTS: dict[str, Callable[[Manifest], None]] = {
    "a token of the prompt differs": _shift_a_token,
    "the prompt is one token shorter": _drop_the_last_id,
    "the options are in the other order": _reverse_the_options,
    "+1 on the score of the first option": _add_one_to_the_first_target,
    "a score off by 0.3 temperatures": _a_slightly_different_score,
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_a_mutant_of_the_vectors_is_refused(name: str, tmp_path: Path) -> None:
    manifest = copy.deepcopy(_manifest())
    MUTANTS[name](manifest)
    with pytest.raises(ModelVerificationError):
        _load(_folder_with(tmp_path, manifest), tmp_path / "cache")


@pytest.mark.skipif(not OTHER_GGUF, reason="needs ARCHAI_JEV_TEST_OTHER_GGUF")
def test_the_file_of_another_dtype_is_refused(tmp_path: Path) -> None:
    # the manifest of the Q4_K_M variant with the bytes of another quantisation: the size and the
    # SHA-256 are the ones of the manifest's file, so the first thing to notice is the hash; with
    # the hash and size rewritten the types of the tensors do not match the recipe either
    assert OTHER_GGUF is not None
    other = Path(OTHER_GGUF)
    manifest = copy.deepcopy(_manifest())
    with pytest.raises((IncompatibleModelError, ModelVerificationError, OSError)):
        _load(_folder_with(tmp_path, manifest, gguf=other), tmp_path / "cache_hash")

    import hashlib

    digest = hashlib.sha256()
    with other.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    entry = manifest["variants"][0]["source"]["file"]
    entry["size"] = other.stat().st_size
    entry["sha256"] = digest.hexdigest()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    with pytest.raises(IncompatibleModelError, match="tensor"):
        _load(_folder_with(sealed, manifest, gguf=other), sealed / "cache")
