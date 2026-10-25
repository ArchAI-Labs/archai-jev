# Weights formats (`dtype`) and quantization

A `dtype` names a **variant of a model's weights**. Each variant has its own calibration and its own
self-check vectors: a different quantization is a different model, and the temperature fitted for
one does not transfer to another.

```python
Jev.from_pretrained()                    # the default model, its default variant (q8_0)
Jev.from_pretrained(dtype="q8_0")        # asking for a variant by name
```

The names this release accepts are `f32`, `bf16` and `q8_0` (uniform types: every weight matrix has
that type, every norm and bias is `F32`). `Jev.list_models()` shows, for each model, the variants it
offers with their size and calibration (`ModelInfo.variants`). The default variant is chosen by the
model's manifest, never by an environment variable, a flag or the hardware: nothing makes the
library switch to a lower precision on its own.

## Why only these

A quantized variant enters the registry only if it passes an admission gate that was written before
any number was measured (accuracy, calibration, Brier score and a measured advantage, plus a
self-check that holds on every platform). The gate is code (`tests/quantization_gate.py`) and the
report it reads is archived (`tests/data/quantization/report.json`, aggregates only).

The one candidate measured for the default model, the **Q4_K_M** of the v2 (K-quant, 4 bits for most
matrices), passed the quality and advantage criteria: the file is 0.60 times the size of the Q8_0,
accuracy is within 0.4 points and Brier within 0.01. It was **refused by the self-check**. The numbers
the engine produces for a 4-bit file differ from one CPU to another (Windows, Linux and macOS, Intel and
Apple Silicon): the same twelve requests moved by up to 0.14 in probability (at temperature 1) between machines. A check loose
enough to accept that (a tolerance of at least 0.27) would hardly notice a wrong weight or template,
so the variant stays out. This is a result, not a bug; the report keeps the figures. It also did not
save memory in our measurement (the engine rewrites the 4-bit weights when it loads them).

K-quants (`q4_k_m`, `q5_k_m`, `q6_k`) are therefore not names of this release.

## The self-check and the cache setting

At the first use of every model revision the library runs the vectors of the manifest and compares
probabilities with the tolerance of the library's table (never one from a manifest). The vectors always
run the way the library computes a prompt by default, in the segments of the template's cuts, **even
when the memory of prefixes is off** (`prefix_cache_mib=0`). The two ways of computing a row give
numbers that differ by a few hundredths of probability on a quantized model, so the check follows the
default computation and does not depend on the setting. With `prefix_cache_mib=0` the answers are
computed in one piece and can differ from those of the default setting by that amount.

## Your own quantized file

A manifest of your own may declare any of the names above with a GGUF whose tensors have the types of
that name (a `q8_0` file with a Q4_K tensor is refused, naming the tensor). Measure it on your data
(`Jev.evaluate`) and calibrate it (`Jev.calibrate`) before you rely on its probabilities: the library
does not carry a temperature over from another variant.
