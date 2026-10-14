"""Checkpoints made of original files (spec 018): the registry entry of Kev, and the safety of
``head.pt`` seen from Python.

The conversion itself is tested in Rust (against the output of the official llama.cpp script);
here is what only Python can show: a ``head.pt`` that would run code if anything executed it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import shutil
import zipfile
from pathlib import Path

import pytest

from archai_jev import IncompatibleModelError, Jev, list_models

FIXTURE = Path(__file__).parent / "data" / "qwen35-mini"


def test_kev_is_a_known_model_with_two_variants() -> None:
    models = {m.name: m for m in list_models()}
    kev = models["jaredpalmer/kev-0.8b"]
    assert kev.revision == "9a45d25eb2ab761841196625383fa1dff0e56c1e"
    assert kev.family == "qwen35-pointer" and kev.head == "pointer"
    assert kev.template == "kev-v1" and kev.dtype == "bf16"
    assert kev.calibrated is True and kev.temperature == pytest.approx(
        2.3510958125672174
    )
    assert kev.license == "Apache-2.0" and (kev.max_context or 0) >= 1
    assert kev.notice and "bf16" in kev.notice
    assert not kev.is_default, "the Decision Index model is the default"
    assert models["nickprock/archai-jev-qwen-1.5b"].is_default


def role(text: str, token_id: int) -> dict[str, object]:
    return {"text": text, "id": token_id}


def _entry(root: Path, path: str) -> dict[str, object]:
    data = (root / path).read_bytes()
    return {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _mini_model(tmp_path: Path, head_pt: bytes) -> Path:
    """A local folder: the originals of the tiny checkpoint and a manifest around ``head_pt``."""
    root = tmp_path / "model"
    shutil.copytree(FIXTURE / "base", root / "base")
    shutil.copytree(FIXTURE / "adapter", root / "adapter")
    shutil.copy(FIXTURE / "prompt-tokenizer.json", root / "prompt-tokenizer.json")
    (root / "head.pt").write_bytes(head_pt)
    ids = [1, 2, 3, 4, 5, 6]
    vector = {
        "id": "v",
        "request": {
            "state": "x",
            "questions": {
                "c": {
                    "type": "choice",
                    "instructions": "i",
                    "criteria": {"a": "x", "b": "y"},
                },
                "n": {"type": "noul", "instructions": "i"},
                "s": {"type": "score", "instructions": "i", "criteria": ["lo", "hi"]},
            },
        },
        "expected": {
            "input_ids": ids,
            "questions": {
                "c": {"logits": [0.0, 0.0], "probabilities": [0.5, 0.5]},
                "n": {"logits": [0.0, 0.0], "probabilities": [0.5, 0.5]},
                "s": {"logits": [0.0, 0.0], "probabilities": [0.5, 0.5]},
            },
        },
    }
    source = {
        "kind": "hf-lora",
        "base": {
            "repo": "Org/Mini-Base",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "config": _entry(root, "base/config.json"),
            "weights": [_entry(root, "base/model.safetensors")],
            "tokenizer": _entry(root, "base/tokenizer.json"),
            "tokenizer_config": _entry(root, "base/tokenizer_config.json"),
        },
        "adapter": {
            "config": _entry(root, "adapter/adapter_config.json"),
            "weights": _entry(root, "adapter/adapter_model.safetensors"),
        },
    }
    manifest = {
        "schema_version": 1,
        "name": "local/mini-kev",
        "revision": "v1",
        "family": "qwen35-pointer",
        "architecture": {
            "name": "qwen35",
            "n_layers": 4,
            "n_mtp_layers": 1,
            "n_embd": 16,
            "n_ff": 32,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 8,
            "n_vocab": 40,
            "tie_embeddings": True,
            "full_attention_interval": 4,
            "ssm_conv_kernel": 4,
            "ssm_state_size": 4,
            "ssm_group_count": 2,
            "ssm_time_step_rank": 2,
            "ssm_inner_size": 8,
            "rope_theta": 10000000,
            "rope_dim": 2,
        },
        "head": {
            "kind": "pointer",
            "d_model": 16,
            "proj_dim": 4,
            "weights": _entry(root, "head.pt"),
        },
        "template": {"id": "kev", "version": 1},
        "tokenizer": {
            "file": _entry(root, "prompt-tokenizer.json"),
            "special_tokens": {
                "fim_prefix": role("<|fim_prefix|>", 32),
                "fim_middle": role("<|fim_middle|>", 33),
                "fim_suffix": role("<|fim_suffix|>", 34),
                "box_start": role("<|box_start|>", 35),
                "box_end": role("<|box_end|>", 36),
            },
        },
        "max_context": 128,
        "license": {"spdx": "Apache-2.0", "url": "https://example.test/license"},
        "tasks": None,
        "default_dtype": "bf16",
        "variants": [
            {
                "dtype": "bf16",
                "source": source,
                "calibration": {
                    "declared": True,
                    "temperature": 2.0,
                    "evidence": "fixture",
                },
                "selfcheck": {"vectors": [vector]},
            }
        ],
    }
    (root / "archai-jev-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


class _Evil:
    """A pickle that runs a command when it is loaded by anything that executes pickles."""

    def __init__(self, command: str) -> None:
        self.command = command

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, (self.command,))


def _torch_like_zip(pickled: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as z:
        z.writestr("head/data.pkl", pickled)
    return buffer.getvalue()


def test_a_head_file_that_would_run_code_is_refused_and_nothing_runs(
    tmp_path: Path,
) -> None:
    # the control: the very same pickle really does run its command when something executes it
    control = tmp_path / "control"
    pickle.loads(pickle.dumps(_Evil(f'echo x > "{control}"'), protocol=2))
    assert (
        control.exists()
    ), "the command of the test must work, or the test proves nothing"

    marker = tmp_path / "marker"
    evil = _torch_like_zip(pickle.dumps(_Evil(f'echo x > "{marker}"'), protocol=2))
    root = _mini_model(tmp_path, evil)
    with pytest.raises(IncompatibleModelError) as info:
        Jev.from_pretrained(root, cache_dir=tmp_path / "cache")
    assert "allow-list" in str(info.value), str(info.value)
    assert not marker.exists(), "the command in the file must never run"
    # a plain pickle of a dictionary is refused because it is not a head, not executed either
    plain = _torch_like_zip(pickle.dumps({"head": {"q.weight": 1}}, protocol=2))
    root2 = _mini_model(tmp_path / "second", plain)
    with pytest.raises(IncompatibleModelError):
        Jev.from_pretrained(root2, cache_dir=tmp_path / "cache2")
