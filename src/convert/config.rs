//! `config.json` of the base model: the numbers a Qwen3.5 GGUF is built from (spec 018, 3.1).
//!
//! The file belongs to a third party, so we read only what we need, but we **refuse** what would
//! change the meaning of the weights if we ignored it (experts, quantisation, untied embeddings,
//! different numbers of key and value heads), and nothing has a default.

use std::path::Path;

use super::problem_text;
use super::safetensors::io_error;
use crate::error::{Error, Result};
use crate::json_strict::{self, Json, Obj};
use crate::models::families::ArchParams;
use crate::models::incompat::Incompat;

const FILE: &str = "config.json of the base model";

/// What the converter needs from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Hparams {
    /// Text layers (without the MTP layer).
    pub n_layers: u64,
    /// MTP layers (`mtp_num_hidden_layers`).
    pub n_mtp: u64,
    /// Hidden size.
    pub hidden: u64,
    /// Feed-forward size.
    pub ff: u64,
    /// Attention heads.
    pub heads: u64,
    /// Key/value heads of the attention.
    pub kv_heads: u64,
    /// Size of an attention head.
    pub head_dim: u64,
    /// Rows of the embedding.
    pub vocab: u64,
    /// `max_position_embeddings`.
    pub max_position: u64,
    /// RMS norm epsilon.
    pub rms_eps: f64,
    /// `full_attention_interval`.
    pub full_attention_interval: u64,
    /// For each text layer: `true` for linear attention (Gated DeltaNet).
    pub linear_layers: Vec<bool>,
    /// Convolution kernel of the linear attention.
    pub conv_kernel: u64,
    /// Head size of the keys of the linear attention.
    pub key_head_dim: u64,
    /// Number of key heads of the linear attention.
    pub key_heads: u64,
    /// Head size of the values of the linear attention.
    pub value_head_dim: u64,
    /// Number of value heads of the linear attention.
    pub value_heads: u64,
    /// `rope_theta`.
    pub rope_theta: f64,
    /// `partial_rotary_factor`.
    pub partial_rotary: f64,
    /// `mrope_section`.
    pub mrope_section: Vec<u64>,
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::SourceFile {
        file: FILE.to_string(),
        detail: detail.into(),
    })
}

fn unsupported(what: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::SourceUnsupported {
        what: what.into(),
        detail: detail.into(),
    })
}

fn num(j: &Json, key: &str) -> Result<f64> {
    match j.get(key) {
        Some(Json::Int(n)) => Ok(*n as f64),
        Some(Json::Float(f)) => Ok(*f),
        Some(other) => Err(invalid(format!(
            "'{key}' must be a number, got {}",
            other.kind()
        ))),
        None => Err(invalid(format!("'{key}' is missing (nothing is assumed)"))),
    }
}

fn int(j: &Json, key: &str) -> Result<u64> {
    match j.get(key) {
        Some(Json::Int(n)) if *n >= 0 => {
            u64::try_from(*n).map_err(|_| invalid(format!("'{key}' is too large")))
        }
        Some(other) => Err(invalid(format!(
            "'{key}' must be a non-negative integer, got {}",
            other.to_canonical_string()
        ))),
        None => Err(invalid(format!("'{key}' is missing (nothing is assumed)"))),
    }
}

fn positive(j: &Json, key: &str) -> Result<u64> {
    match int(j, key)? {
        0 => Err(invalid(format!("'{key}' must be greater than 0"))),
        n => Ok(n),
    }
}

fn text<'a>(j: &'a Json, key: &str) -> Result<&'a str> {
    match j.get(key) {
        Some(Json::Str(s)) => Ok(s),
        Some(other) => Err(invalid(format!(
            "'{key}' must be a string, got {}",
            other.kind()
        ))),
        None => Err(invalid(format!("'{key}' is missing (nothing is assumed)"))),
    }
}

fn boolean(j: &Json, key: &str) -> Result<bool> {
    match j.get(key) {
        Some(Json::Bool(b)) => Ok(*b),
        Some(other) => Err(invalid(format!(
            "'{key}' must be a boolean, got {}",
            other.kind()
        ))),
        None => Err(invalid(format!("'{key}' is missing (nothing is assumed)"))),
    }
}

impl Hparams {
    /// Read and check `config.json`.
    ///
    /// # Errors
    /// `IncompatibleModelError` (`SourceFile` or `SourceUnsupported`) for anything missing,
    /// inconsistent or not supported; an I/O error if the file cannot be read.
    pub fn read(path: &Path) -> Result<Hparams> {
        let bytes = std::fs::read(path).map_err(|e| io_error(path, &e))?;
        let json = json_strict::parse(&bytes).map_err(|e| invalid(e.to_string()))?;
        Hparams::from_json(&json)
    }

    /// Check an already parsed `config.json`.
    ///
    /// # Errors
    /// See [`Hparams::read`].
    pub fn from_json(json: &Json) -> Result<Hparams> {
        if !matches!(json, Json::Object(_)) {
            return Err(invalid("it is not a JSON object"));
        }
        let archs = match json.get("architectures") {
            Some(Json::Array(a)) => a.iter().filter_map(|x| match x {
                Json::Str(s) => Some(s.as_str()),
                _ => None,
            }),
            _ => return Err(invalid("'architectures' is missing or not a list")),
        }
        .collect::<Vec<_>>();
        if archs != ["Qwen3_5ForConditionalGeneration"] && archs != ["Qwen3_5ForCausalLM"] {
            return Err(invalid(format!(
                "architectures is {archs:?}; this converter handles Qwen3_5ForConditionalGeneration and Qwen3_5ForCausalLM"
            )));
        }
        let model_type = text(json, "model_type")?;
        if model_type != "qwen3_5" {
            return Err(invalid(format!(
                "model_type is {model_type:?}, expected \"qwen3_5\""
            )));
        }
        let Some(tc) = json
            .get("text_config")
            .filter(|t| matches!(t, Json::Object(_)))
        else {
            return Err(invalid(
                "'text_config' is missing: only the layout with a text_config is supported",
            ));
        };
        if text(tc, "model_type")? != "qwen3_5_text" {
            return Err(invalid("text_config.model_type is not \"qwen3_5_text\""));
        }
        for key in [
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_intermediate_size",
            "quantization_config",
        ] {
            if json.get(key).is_some() || tc.get(key).is_some() {
                return Err(unsupported(
                    format!("'{key}' (experts or quantised weights)"),
                    "only dense, unquantised checkpoints are converted",
                ));
            }
        }

        let n_layers = positive(tc, "num_hidden_layers")?;
        let interval = positive(tc, "full_attention_interval")?;
        let layer_types = match tc.get("layer_types") {
            Some(Json::Array(a)) => a,
            _ => return Err(invalid("'layer_types' is missing or not a list")),
        };
        if layer_types.len() as u64 != n_layers {
            return Err(invalid(format!(
                "layer_types has {} entries but num_hidden_layers is {n_layers}",
                layer_types.len()
            )));
        }
        let mut linear_layers = Vec::with_capacity(layer_types.len());
        for (i, t) in layer_types.iter().enumerate() {
            let linear = match t {
                Json::Str(s) if s == "linear_attention" => true,
                Json::Str(s) if s == "full_attention" => false,
                other => {
                    return Err(invalid(format!(
                        "layer_types[{i}] is {}, expected linear_attention or full_attention",
                        other.to_canonical_string()
                    )));
                }
            };
            if linear == (i as u64 + 1).is_multiple_of(interval) {
                return Err(invalid(format!(
                    "layer_types[{i}] disagrees with full_attention_interval {interval}"
                )));
            }
            linear_layers.push(linear);
        }

        let key_heads = positive(tc, "linear_num_key_heads")?;
        let value_heads = positive(tc, "linear_num_value_heads")?;
        if key_heads != value_heads {
            return Err(unsupported(
                "different numbers of key and value heads in the linear attention",
                format!(
                    "linear_num_key_heads is {key_heads} and linear_num_value_heads is {value_heads}: the heads of V would have to be reordered, which cannot be checked against a reference yet"
                ),
            ));
        }
        if !boolean(tc, "attn_output_gate")? {
            return Err(unsupported(
                "an attention without output gate",
                "attn_output_gate is false",
            ));
        }
        if !boolean(tc, "tie_word_embeddings")? {
            return Err(unsupported(
                "untied embeddings",
                "tie_word_embeddings is false; only tied embeddings are converted",
            ));
        }
        if let Some(Json::Array(a)) = tc.get("mlp_only_layers")
            && !a.is_empty()
        {
            return Err(unsupported("mlp_only_layers", "the list is not empty"));
        }
        let n_mtp = int(tc, "mtp_num_hidden_layers")?;
        if n_mtp != 1 {
            return Err(unsupported(
                "a number of MTP layers other than one",
                format!(
                    "mtp_num_hidden_layers is {n_mtp}: the shared MTP tensors (mtp.fc, mtp.norm…) are laid out for exactly one layer"
                ),
            ));
        }
        if tc.get("mtp_use_dedicated_embeddings") == Some(&Json::Bool(true)) {
            return Err(unsupported(
                "dedicated embeddings for the MTP layer",
                "mtp_use_dedicated_embeddings is true",
            ));
        }
        let Some(rope) = tc.get("rope_parameters") else {
            return Err(invalid("'rope_parameters' is missing"));
        };
        let mut r = Obj::new(rope, "rope_parameters").map_err(|e| invalid(problem_text(&e)))?;
        let rope_type = r.str("rope_type").map_err(|e| invalid(problem_text(&e)))?;
        if rope_type != "default" {
            return Err(unsupported(
                format!("rope_type {rope_type:?}"),
                "only the default rope is converted",
            ));
        }
        if !boolean(rope, "mrope_interleaved")? {
            return Err(unsupported(
                "a non-interleaved mrope",
                "mrope_interleaved is false",
            ));
        }
        let mrope_section = match rope.get("mrope_section") {
            Some(Json::Array(a)) => a
                .iter()
                .map(|v| match v {
                    Json::Int(n) if *n >= 0 => u64::try_from(*n).ok(),
                    _ => None,
                })
                .collect::<Option<Vec<u64>>>()
                .filter(|v| (1..=4).contains(&v.len()))
                .ok_or_else(|| invalid("mrope_section must be 1 to 4 non-negative integers"))?,
            _ => return Err(invalid("'mrope_section' is missing or not a list")),
        };
        let partial_rotary = num(rope, "partial_rotary_factor")?;
        let rope_theta = num(rope, "rope_theta")?;
        if partial_rotary <= 0.0 || partial_rotary > 1.0 || rope_theta <= 0.0 {
            return Err(invalid(
                "rope_theta and partial_rotary_factor are out of range",
            ));
        }
        let rms_eps = num(tc, "rms_norm_eps")?;
        if !(rms_eps > 0.0 && rms_eps.is_finite()) {
            return Err(invalid("rms_norm_eps must be a number above 0"));
        }
        let head_dim = positive(tc, "head_dim")?;
        if ((head_dim as f64) * partial_rotary).fract() != 0.0 {
            return Err(invalid(
                "head_dim * partial_rotary_factor is not a whole number",
            ));
        }
        Ok(Hparams {
            n_layers,
            n_mtp,
            hidden: positive(tc, "hidden_size")?,
            ff: positive(tc, "intermediate_size")?,
            heads: positive(tc, "num_attention_heads")?,
            kv_heads: positive(tc, "num_key_value_heads")?,
            head_dim,
            vocab: positive(tc, "vocab_size")?,
            max_position: positive(tc, "max_position_embeddings")?,
            rms_eps,
            full_attention_interval: interval,
            linear_layers,
            conv_kernel: positive(tc, "linear_conv_kernel_dim")?,
            key_head_dim: positive(tc, "linear_key_head_dim")?,
            key_heads,
            value_head_dim: positive(tc, "linear_value_head_dim")?,
            value_heads,
            rope_theta,
            partial_rotary,
            mrope_section,
        })
    }

    /// The size of the rotary part of a head, `head_dim * partial_rotary_factor`.
    pub fn rope_dim(&self) -> u64 {
        (self.head_dim as f64 * self.partial_rotary) as u64
    }

    /// The `architecture` parameters of the manifest, as `(name, value)` pairs of the family.
    fn expected(&self) -> Vec<(&'static str, Option<u64>)> {
        let theta = (self.rope_theta.fract() == 0.0).then_some(self.rope_theta as u64);
        vec![
            ("n_layers", Some(self.n_layers)),
            ("n_mtp_layers", Some(self.n_mtp)),
            ("n_embd", Some(self.hidden)),
            ("n_ff", Some(self.ff)),
            ("n_heads", Some(self.heads)),
            ("n_kv_heads", Some(self.kv_heads)),
            ("head_dim", Some(self.head_dim)),
            ("n_vocab", Some(self.vocab)),
            (
                "full_attention_interval",
                Some(self.full_attention_interval),
            ),
            ("ssm_conv_kernel", Some(self.conv_kernel)),
            ("ssm_state_size", Some(self.key_head_dim)),
            ("ssm_group_count", Some(self.key_heads)),
            ("ssm_time_step_rank", Some(self.value_heads)),
            (
                "ssm_inner_size",
                Some(self.value_head_dim * self.value_heads),
            ),
            ("rope_theta", theta),
            ("rope_dim", Some(self.rope_dim())),
        ]
    }

    /// Compare with the `architecture` of the manifest: every parameter must be equal.
    ///
    /// # Errors
    /// `ArchParamMismatch` naming the first parameter that differs.
    pub fn check_against(&self, params: &ArchParams) -> std::result::Result<(), Incompat> {
        for (name, value) in self.expected() {
            let manifest = params.int(name);
            if value != Some(manifest) {
                return Err(Incompat::ArchParamMismatch {
                    param: name.to_string(),
                    manifest: manifest.to_string(),
                    file: "config.json".to_string(),
                    got: value.map_or_else(|| "a non-integer value".to_string(), |v| v.to_string()),
                });
            }
        }
        if !params.flag("tie_embeddings") {
            return Err(Incompat::ArchParamMismatch {
                param: "tie_embeddings".to_string(),
                manifest: "false".to_string(),
                file: "config.json".to_string(),
                got: "true".to_string(),
            });
        }
        Ok(())
    }
}
