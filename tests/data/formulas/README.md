# Formula vectors

38 synthetic vectors with the outputs of Kev's confidence, score value and yes/no formulas, used by
the tests of the public calibration functions (`archai_jev.calibration`).

| File | Rows | Content |
|---|---|---|
| `formulas.jsonl` | 38 | 13 `choice`, 17 `score`, 8 `noul` (yes/no) |

SHA-256 of `formulas.jsonl`: `1617a9aeae6a852337fd3ae370358c9ad9275ba451602804d7de721a5f365665`.

## Rows

- `choice`: `p` (probabilities, may not sum to 1), `confidence`, `argmax`. Also `confidence_45923b7`, the
  historical formula; the tests do not use it.
- `score`: `p`, `value`, `D`, `mode`, `confidence`. Also the `*_rounded_4dp` and `confidence_45923b7` fields,
  which the tests do not use.
- `noul`: `logits_raw` (`[no, yes]`), `temperature`, `p_true`. Also `logits_tempered`, `p_true_sigmoid_form`
  and `p_true_rounded_4dp`.

Four rows have a sum other than 1 (`[0, 0, 0]` and `[2, 1, 1]`, as a choice and as a score). Kev
normalises them; the public functions of archai-jev reject them on purpose.

## Where the data comes from

- **Inputs**: written by the authors of archai-jev.
- **Outputs**: computed with Kev's own code (`jaredpalmer/kev`, commit `67117d85`, Apache-2.0), in float64.
- The script that produced them is not part of this repository.
