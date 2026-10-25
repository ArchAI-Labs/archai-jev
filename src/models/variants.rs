//! What a manifest says about its variants, for `ModelInfo.variants` (spec 009, section 2).
//!
//! Static facts read from the manifest: no network, no disk. They let a user see which weights
//! a model offers and which of them carry a calibration of their own, instead of guessing names.

use super::manifest::{Manifest, Source};

/// One variant of the weights of a model, as the manifest declares it.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantData {
    /// Weights type name, e.g. `q8_0`.
    pub dtype: String,
    /// Size of the GGUF file in bytes; `None` when the variant is produced by converting the
    /// original files at first use (the size of the result is not known before that).
    pub size_bytes: Option<u64>,
    /// Whether the manifest declares a calibration for **this** variant.
    pub calibrated: bool,
    /// The declared temperature of this variant, if there is one.
    pub temperature: Option<f64>,
    /// Whether this is the variant used when no `dtype` is given.
    pub is_default: bool,
}

/// The variants of `manifest`, in manifest order.
pub fn variants_of(manifest: &Manifest) -> Vec<VariantData> {
    manifest
        .variants
        .iter()
        .map(|v| VariantData {
            dtype: v.dtype.clone(),
            size_bytes: match &v.source {
                Source::Gguf { file } => Some(file.size),
                Source::HfLora(_) | Source::Reserved { .. } => None,
            },
            calibrated: v.calibration.declared,
            temperature: if v.calibration.declared {
                v.calibration.temperature
            } else {
                None
            },
            is_default: v.dtype == manifest.default_dtype,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry::Registry;

    fn entry(name: &str) -> Manifest {
        Registry::builtin()
            .unwrap()
            .entries()
            .iter()
            .find(|m| m.name == name)
            .cloned()
            .unwrap()
    }

    #[test]
    fn the_default_model_offers_its_declared_q8_0() {
        let v = variants_of(&entry("nickprock/archai-jev-qwen-1.5b"));
        assert_eq!(
            v,
            vec![VariantData {
                dtype: "q8_0".to_string(),
                size_bytes: Some(1_646_572_736),
                calibrated: true,
                temperature: Some(2.09),
                is_default: true,
            }]
        );
    }

    #[test]
    fn a_converted_variant_has_no_size_and_exactly_one_is_the_default() {
        let v = variants_of(&entry("jaredpalmer/kev-0.8b"));
        let names: Vec<&str> = v.iter().map(|x| x.dtype.as_str()).collect();
        assert_eq!(names, ["bf16", "f32"]);
        assert!(v.iter().all(|x| x.size_bytes.is_none() && x.calibrated));
        assert_eq!(v.iter().filter(|x| x.is_default).count(), 1);
        assert!(v[0].is_default);
    }
}
