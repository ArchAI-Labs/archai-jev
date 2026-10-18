//! Which `dtype` names exist and which ggml types the tensors of each may have (spec 009).
//!
//! A `dtype` names a *variant of the weights* (D22): its own calibration, its own vectors, its own
//! tolerance. `f32`, `bf16` and `q8_0` are uniform: every weight matrix has that type and every
//! vector (norms, biases) is `F32`. A K-quant is not: a recipe of `llama-quantize` picks a type
//! per tensor and changes between llama.cpp releases, so a name alone does not say what the file
//! holds. Until a K-quant is admitted to the registry (spec 009, section 5) it exists only as a
//! study row behind the `quant-study` feature, with a permissive set of types.

use super::families::Role;
use super::gguf::GgmlType;

/// The canonical `dtype` names, in the order messages list them.
pub const DTYPES: &[&str] = &[
    "f32",
    "bf16",
    "q8_0",
    #[cfg(feature = "quant-study")]
    "q6_k",
    #[cfg(feature = "quant-study")]
    "q5_k_m",
    #[cfg(feature = "quant-study")]
    "q4_k_m",
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
        #[cfg(feature = "quant-study")]
        "q6_k" | "q5_k_m" | "q4_k_m" => Some(STUDY_MATRIX),
        _ => None,
    }
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
        // passed the gate (none yet). Study names exist only behind `quant-study`.
        #[cfg(not(feature = "quant-study"))]
        assert_eq!(DTYPES, &["f32", "bf16", "q8_0"]);
        #[cfg(feature = "quant-study")]
        assert_eq!(DTYPES, &["f32", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"]);
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

    #[cfg(feature = "quant-study")]
    #[test]
    fn study_recipes_accept_k_quants_and_nothing_else() {
        for d in ["q6_k", "q5_k_m", "q4_k_m"] {
            let a = allowed_types(d, Role::Matrix).unwrap();
            assert!(a.contains(&GgmlType::Q4_K) && a.contains(&GgmlType::Q6_K));
            assert!(!a.contains(&GgmlType::F32) && !a.contains(&GgmlType::BF16));
        }
    }
}
