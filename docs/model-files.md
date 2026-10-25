# Where the model files go, and how much room they take

`archai-jev` never ships model weights and hosts none. The first time you ask for a model it
downloads the **original public files** from Hugging Face, checks each one against a SHA-256 fixed
in the library, and keeps them in a cache folder. Some models are published as a ready-made GGUF
(the default model is). Others are published as a base model plus a LoRA adapter (Kev is): for
those the library **converts** the originals into a GGUF once, in Rust, and keeps the result too.
No Python, no PyTorch and no extra tool is involved.

```python
from archai_jev import Jev

jev = Jev.from_pretrained("jaredpalmer/kev-0.8b")   # first call: download, convert, self-check
jev = Jev.from_pretrained("jaredpalmer/kev-0.8b")   # later calls: straight from the cache
```

## What the first call does

1. Downloads the files (about 1.7 GiB for `jaredpalmer/kev-0.8b`), each checked by SHA-256.
2. Converts them into a GGUF (a few seconds on a recent CPU; the numbers are in the changelog).
   The log (`logging`, logger `archai_jev`) says when it starts, how far it is, and how large the
   result is. `Ctrl-C` stops it cleanly and leaves nothing half-written.
3. Runs the model's self-check vectors on the converted file, once.

Later calls find the converted file and the record of the checks and do none of this again. If the
converted file in the cache was damaged, it is rebuilt from the verified originals; it is never
used as it is.

## Where it lives and how big it is

The cache folder is, in this order: the `cache_dir` argument, the `ARCHAI_JEV_CACHE` variable, or
the folder your operating system uses for caches (`~/.cache/archai-jev` on Linux,
`~/Library/Caches/archai-jev` on macOS, `%LOCALAPPDATA%\archai-jev\Cache` on Windows).

| What | Where | Size for Kev-0.8B |
|---|---|---|
| Original files | `v1/blobs/<sha256>` | about 1.7 GiB |
| Converted GGUF, bf16 (default) | `v1/materialized/…/bf16/…/model.gguf` | about 1.45 GiB |
| Converted GGUF, f32 (`dtype="f32"`) | `v1/materialized/…/f32/…/model.gguf` | about 2.9 GiB |
| Records of the checks | `v1/verified/` | a few KB |

Nothing is deleted for you: the originals are kept so that a new version of the converter can
convert again without downloading. To free the space, delete the folder (or only
`v1/blobs/` once the GGUF you use is in `v1/materialized/` and you do not plan to change dtype).
Offline use (`offline=True`) needs the originals in the cache only the first time.

## What is refused

The library does not guess. A checkpoint is refused (`IncompatibleModelError`, with the file and
the value) when a file does not match its hash, a tensor is missing, unexpected, of another shape
or type, a number is not finite, the LoRA uses options other than plain LoRA, or a `head.pt` holds
anything but plain tensors. `head.pt` files are read with a reader that **never runs** what a
pickle file asks for.
