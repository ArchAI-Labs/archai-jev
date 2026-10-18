//! Which `dtype` names exist and which ggml types the tensors of each may have (spec 009).
//!
//! A `dtype` names a *variant of the weights* (D22): its own calibration, its own vectors, its own
//! tolerance. `f32`, `bf16` and `q8_0` are uniform: every weight matrix has that type and every
//! vector (norms, biases) is `F32`. A K-quant is not: a recipe of `llama-quantize` picks a type
//! per tensor and changes between llama.cpp releases, so a name alone does not say what the file
//! holds. A K-quant that passed the gate (spec 009, section 5) has an **exact** recipe here, read
//! from the file that was measured (`q4_k_m`); one that did not exists only as a study row behind
//! the `quant-study` feature, with a permissive set of types.

use super::families::Role;
use super::gguf::GgmlType;

/// The canonical `dtype` names, in the order messages list them.
pub const DTYPES: &[&str] = &[
    "f32",
    "bf16",
    "q8_0",
    "q4_k_m",
    #[cfg(feature = "quant-study")]
    "q6_k",
    #[cfg(feature = "quant-study")]
    "q5_k_m",
];

/// Whether a `dtype` stores its weights quantised (everything but full `f32` and `bf16`).
///
/// A quantised variant is a model of its own: its calibration has to be measured for it (D22).
pub fn is_quantized(dtype: &str) -> bool {
    !matches!(dtype, "f32" | "bf16")
}

/// The canonical name for `given` (`float32` and `bfloat16` are synonyms), if it is one.
///
/// The match is exact: `"Q8_0"` is not `"q8_0"` (never guess).
pub fn canonical(given: &str) -> Option<&'static str> {
    match given {
        "float32" => Some("f32"),
        "bfloat16" => Some("bf16"),
        other => DTYPES.iter().copied().find(|d| *d == other),
    }
}

/// Types a K-quant study file may use for a matrix (any K-quant, or `Q8_0` where a recipe keeps
/// 8 bits). Permissive on purpose: the measurement reads the real recipe from the file.
#[cfg(feature = "quant-study")]
const STUDY_MATRIX: &[GgmlType] = &[
    GgmlType::Q4_K,
    GgmlType::Q3_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
];

/// The ggml types a tensor with `role` may have in a file of `dtype`; `None` for a matrix of a
/// `dtype` without a recipe.
pub fn allowed_types(dtype: &str, role: Role) -> Option<&'static [GgmlType]> {
    if role == Role::Vector {
        return Some(&[GgmlType::F32]);
    }
    match dtype {
        "f32" => Some(&[GgmlType::F32]),
        "bf16" => Some(&[GgmlType::BF16]),
        "q8_0" => Some(&[GgmlType::Q8_0]),
        // the matrices of a Q4_K_M are Q4_K or Q6_K: which one is decided per tensor by
        // `expected_types`; this is the set a matrix of this dtype can have at all
        "q4_k_m" => Some(&[GgmlType::Q4_K, GgmlType::Q6_K]),
        #[cfg(feature = "quant-study")]
        "q6_k" | "q5_k_m" => Some(STUDY_MATRIX),
        _ => None,
    }
}

/// llama.cpp's rule for giving a tensor more bits in a `_M` recipe: the first eighth of the
/// layers, the last eighth, and every third one in between.
fn uses_more_bits(layer: u64, layers: u64) -> bool {
    layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2
}

/// The layer index in a tensor name like `blk.12.attn_v.weight`, and what follows it.
fn layer_of(name: &str) -> Option<(u64, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (index, tail) = rest.split_once('.')?;
    Some((index.parse().ok()?, tail))
}

/// The ggml types the tensor `name` (a `role`) may have in a file of `dtype` with `layers`
/// transformer layers. For a uniform `dtype` this is [`allowed_types`]. For `q4_k_m` it is **one**
/// type, as in the file that passed the gate (`llama-quantize` build 11425, no importance matrix):
/// `Q6_K` for `token_embd.weight` and for `attn_v` and `ffn_down` of the layers that get more
/// bits, `Q4_K` for every other matrix, `F32` for the vectors. A file made with another release,
/// or by hand, is refused by name (spec 009 section 4).
pub fn expected_types(
    dtype: &str,
    name: &str,
    role: Role,
    layers: u64,
) -> Option<&'static [GgmlType]> {
    if dtype != "q4_k_m" || role == Role::Vector {
        return allowed_types(dtype, role);
    }
    let more_bits = match layer_of(name) {
        Some((layer, "attn_v.weight" | "ffn_down.weight")) => uses_more_bits(layer, layers),
        Some(_) => false,
        None => name == "token_embd.weight",
    };
    Some(if more_bits {
        &[GgmlType::Q6_K]
    } else {
        &[GgmlType::Q4_K]
    })
}

/// The type a test checkpoint writes for the matrices of `dtype`.
#[cfg(feature = "testing")]
pub fn default_matrix_type(dtype: &str) -> Option<GgmlType> {
    allowed_types(dtype, Role::Matrix).and_then(|a| a.first().copied())
}

/// `Q8_0`, or `Q4_K or Q6_K` for several: how an expectation reads in a message.
pub fn describe(allowed: Option<&[GgmlType]>) -> String {
    match allowed {
        Some(types) if !types.is_empty() => types
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>()
            .join(" or "),
        _ => "a supported type".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_of_a_release_are_the_admitted_ones() {
        // spec 009 AC-1: the set a release accepts is f32, bf16, q8_0 plus the K-quants that
        // passed the gate (q4_k_m, D58). Study names exist only behind `quant-study`.
        #[cfg(not(feature = "quant-study"))]
        assert_eq!(DTYPES, &["f32", "bf16", "q8_0", "q4_k_m"]);
        #[cfg(feature = "quant-study")]
        assert_eq!(DTYPES, &["f32", "bf16", "q8_0", "q4_k_m", "q6_k", "q5_k_m"]);
    }

    #[test]
    fn canonical_names_synonyms_and_refusals() {
        assert_eq!(canonical("f32"), Some("f32"));
        assert_eq!(canonical("float32"), Some("f32"));
        assert_eq!(canonical("bfloat16"), Some("bf16"));
        assert_eq!(canonical("q8_0"), Some("q8_0"));
        for bad in [
            "int8", "q4", "fp16", "f16", "Q8_0", "BF16", " q8_0", "", "q8",
        ] {
            assert_eq!(canonical(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn uniform_recipes_and_vectors() {
        assert_eq!(
            allowed_types("q8_0", Role::Matrix),
            Some(&[GgmlType::Q8_0][..])
        );
        assert_eq!(
            allowed_types("bf16", Role::Matrix),
            Some(&[GgmlType::BF16][..])
        );
        // a vector is F32 whatever the dtype, and a matrix of an unknown dtype has no recipe
        assert_eq!(
            allowed_types("q8_0", Role::Vector),
            Some(&[GgmlType::F32][..])
        );
        assert_eq!(allowed_types("q2_k", Role::Matrix), None);
    }

    #[test]
    fn only_full_precision_is_not_quantized() {
        assert!(!is_quantized("f32") && !is_quantized("bf16"));
        for d in ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "anything"] {
            assert!(is_quantized(d), "{d}");
        }
    }

    #[test]
    fn every_name_has_a_recipe() {
        for d in DTYPES {
            assert!(allowed_types(d, Role::Matrix).is_some(), "{d}");
        }
    }

    #[test]
    fn describing_expectations() {
        assert_eq!(describe(Some(&[GgmlType::Q8_0])), "Q8_0");
        assert_eq!(
            describe(Some(&[GgmlType::Q4_K, GgmlType::Q6_K])),
            "Q4_K or Q6_K"
        );
        assert_eq!(describe(None), "a supported type");
    }

    #[test]
    fn the_q4_k_m_recipe_is_the_one_of_the_measured_file() {
        // the 28 layers of the v2 (Qwen2.5-1.5B); the inventory is that of the measured GGUF:
        // Q6_K for the embeddings and for attn_v and ffn_down of layers 0-2, 5, 8, 11, 14, 17,
        // 20 and 23-27; Q4_K for every other matrix; F32 for norms and biases
        let more = [0u64, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27];
        let ty = |name: &str, role| expected_types("q4_k_m", name, role, 28);
        assert_eq!(
            ty("token_embd.weight", Role::Matrix),
            Some(&[GgmlType::Q6_K][..])
        );
        for layer in 0..28u64 {
            let v = format!("blk.{layer}.attn_v.weight");
            let down = format!("blk.{layer}.ffn_down.weight");
            let want = if more.contains(&layer) {
                GgmlType::Q6_K
            } else {
                GgmlType::Q4_K
            };
            assert_eq!(ty(&v, Role::Matrix), Some(&[want][..]), "{v}");
            assert_eq!(ty(&down, Role::Matrix), Some(&[want][..]), "{down}");
            for other in ["attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"] {
                let name = format!("blk.{layer}.{other}.weight");
                assert_eq!(
                    ty(&name, Role::Matrix),
                    Some(&[GgmlType::Q4_K][..]),
                    "{name}"
                );
            }
            let bias = format!("blk.{layer}.attn_q.bias");
            assert_eq!(ty(&bias, Role::Vector), Some(&[GgmlType::F32][..]));
        }
        // a uniform dtype ignores the name
        assert_eq!(
            expected_types("q8_0", "blk.0.attn_v.weight", Role::Matrix, 28),
            Some(&[GgmlType::Q8_0][..])
        );
    }

    #[test]
    fn the_more_bits_rule_follows_llama_cpp() {
        // n = 28: the first 3, the last 4 (24..27) and every third one from layer 5
        let layers: Vec<u64> = (0..28).filter(|i| uses_more_bits(*i, 28)).collect();
        assert_eq!(
            layers,
            vec![0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27]
        );
        // another depth gives another set: the recipe is tied to the layer count
        let other: Vec<u64> = (0..24).filter(|i| uses_more_bits(*i, 24)).collect();
        assert_ne!(other, layers);
    }

    #[cfg(feature = "quant-study")]
    #[test]
    fn study_recipes_accept_k_quants_and_nothing_else() {
        for d in ["q6_k", "q5_k_m"] {
            let a = allowed_types(d, Role::Matrix).unwrap();
            assert!(a.contains(&GgmlType::Q4_K) && a.contains(&GgmlType::Q6_K));
            assert!(!a.contains(&GgmlType::F32) && !a.contains(&GgmlType::BF16));
        }
    }
}
