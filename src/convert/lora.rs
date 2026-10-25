//! The LoRA adapter (spec 018, 3.3): its configuration and the names of its tensors.
//!
//! The merge formula `W' = W + (alpha / r) * B * A` is only right for **plain** LoRA. Every
//! option of `adapter_config.json` that would change it (rank-stabilised scaling, DoRA, other
//! ranks or alphas per module, trained token embeddings, extra modules, ...) is therefore
//! refused, and so is any key we do not know: a newer `peft` may have added a feature we would
//! silently ignore.

use std::path::Path;

use super::safetensors::io_error;
use crate::error::{Error, Result};
use crate::json_strict::{self, Json};
use crate::models::incompat::Incompat;

const FILE: &str = "adapter_config.json";

/// What the merge needs from `adapter_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraConfig {
    /// The rank `r`.
    pub r: u64,
    /// `lora_alpha`.
    pub alpha: f64,
    /// The module names LoRA was applied to (`q_proj`, ...).
    pub target_modules: Vec<String>,
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::SourceFile {
        file: FILE.to_string(),
        detail: detail.into(),
    })
}

fn unsupported(key: &str, shown: &Json, why: &str) -> Error {
    Error::IncompatibleModel(Incompat::SourceUnsupported {
        what: format!("the LoRA option '{key}'"),
        detail: format!("its value is {}, {why}", shown.to_canonical_string()),
    })
}

/// How an option of `adapter_config.json` is treated.
enum Rule {
    /// Must be `null` (absent counts as `null`).
    Null,
    /// Must be `false` or `null`.
    False,
    /// Must be an empty object, `null` or absent.
    Empty,
    /// Does not change the merged weights.
    Inert,
}

const OPTIONS: &[(&str, Rule)] = &[
    ("alora_invocation_tokens", Rule::Null),
    ("alpha_pattern", Rule::Empty),
    ("arrow_config", Rule::Null),
    ("auto_mapping", Rule::Null),
    ("corda_config", Rule::Null),
    ("ensure_weight_tying", Rule::False),
    ("eva_config", Rule::Null),
    ("exclude_modules", Rule::Null),
    ("fan_in_fan_out", Rule::False),
    ("inference_mode", Rule::Inert),
    ("init_lora_weights", Rule::Inert),
    ("kasa_config", Rule::Null),
    ("layer_replication", Rule::Null),
    ("layers_pattern", Rule::Null),
    ("layers_to_transform", Rule::Null),
    ("loftq_config", Rule::Empty),
    ("lora_bias", Rule::False),
    ("lora_dropout", Rule::Inert),
    ("lora_ga_config", Rule::Null),
    ("megatron_config", Rule::Null),
    ("megatron_core", Rule::Inert),
    ("modules_to_save", Rule::Null),
    ("monteclora_config", Rule::Null),
    ("peft_version", Rule::Inert),
    ("qalora_group_size", Rule::Inert),
    ("rank_pattern", Rule::Empty),
    ("revision", Rule::Inert),
    ("target_parameters", Rule::Null),
    ("task_type", Rule::Inert),
    ("trainable_token_indices", Rule::Null),
    ("use_bdlora", Rule::False),
    ("use_dora", Rule::False),
    ("use_qalora", Rule::False),
    ("use_rslora", Rule::False),
    ("velora_config", Rule::Null),
];

impl LoraConfig {
    /// Read and check `adapter_config.json`; `base_repo` is the repository of the base model it
    /// must have been trained on.
    ///
    /// # Errors
    /// `IncompatibleModelError` (`SourceFile` or `SourceUnsupported`) naming the option; an I/O
    /// error if the file cannot be read.
    pub fn read(path: &Path, base_repo: &str) -> Result<LoraConfig> {
        let bytes = std::fs::read(path).map_err(|e| io_error(path, &e))?;
        let json = json_strict::parse(&bytes).map_err(|e| invalid(e.to_string()))?;
        LoraConfig::from_json(&json, base_repo)
    }

    /// Check an already parsed `adapter_config.json`.
    ///
    /// # Errors
    /// See [`LoraConfig::read`].
    pub fn from_json(json: &Json, base_repo: &str) -> Result<LoraConfig> {
        let Json::Object(pairs) = json else {
            return Err(invalid("it is not a JSON object"));
        };
        for (key, value) in pairs {
            let known = ["peft_type", "r", "lora_alpha", "target_modules", "bias"]
                .contains(&key.as_str())
                || key == "base_model_name_or_path";
            let Some((_, rule)) = OPTIONS.iter().find(|(k, _)| k == key) else {
                if known {
                    continue;
                }
                return Err(Error::IncompatibleModel(Incompat::SourceUnsupported {
                    what: format!("the option '{key}' of adapter_config.json"),
                    detail: "the converter does not know it and cannot tell whether it changes the weights".to_string(),
                }));
            };
            let ok = match (rule, value) {
                (Rule::Inert, _) | (Rule::Null | Rule::False | Rule::Empty, Json::Null) => true,
                (Rule::False, Json::Bool(false)) => true,
                (Rule::Empty, Json::Object(o)) => o.is_empty(),
                _ => false,
            };
            if !ok {
                return Err(unsupported(
                    key,
                    value,
                    match rule {
                        Rule::Null => "only plain LoRA (null) is supported",
                        Rule::False => "only plain LoRA (false) is supported",
                        Rule::Empty => "only plain LoRA (nothing set per module) is supported",
                        Rule::Inert => "",
                    },
                ));
            }
        }
        match json.get("peft_type") {
            Some(Json::Str(s)) if s == "LORA" => {}
            other => {
                return Err(invalid(format!(
                    "peft_type is {}, expected \"LORA\"",
                    other.map_or_else(|| "missing".to_string(), Json::to_canonical_string)
                )));
            }
        }
        match json.get("bias") {
            Some(Json::Str(s)) if s == "none" => {}
            other => {
                return Err(match other {
                    Some(v) => unsupported("bias", v, "only \"none\" is supported"),
                    None => invalid("'bias' is missing (nothing is assumed)"),
                });
            }
        }
        let r = match json.get("r") {
            Some(Json::Int(n)) if *n > 0 && *n <= 4096 => u64::try_from(*n).unwrap_or(0),
            other => {
                return Err(invalid(format!(
                    "r must be an integer from 1 to 4096, got {}",
                    other.map_or_else(|| "nothing".to_string(), Json::to_canonical_string)
                )));
            }
        };
        let alpha = match json.get("lora_alpha") {
            Some(Json::Int(n)) => *n as f64,
            Some(Json::Float(f)) => *f,
            other => {
                return Err(invalid(format!(
                    "lora_alpha must be a number, got {}",
                    other.map_or_else(|| "nothing".to_string(), Json::to_canonical_string)
                )));
            }
        };
        if !(alpha.is_finite() && alpha > 0.0 && (alpha / r as f64).is_finite()) {
            return Err(invalid(format!(
                "lora_alpha {alpha} and r {r} do not give a finite scale above 0"
            )));
        }
        let target_modules = match json.get("target_modules") {
            Some(Json::Array(a)) if !a.is_empty() => a
                .iter()
                .map(|v| match v {
                    Json::Str(s) if !s.is_empty() => Some(s.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<String>>>()
                .ok_or_else(|| invalid("target_modules must be a list of module names"))?,
            _ => {
                return Err(invalid(
                    "target_modules must be a non-empty list (a pattern string is not supported)",
                ));
            }
        };
        match json.get("base_model_name_or_path") {
            Some(Json::Str(s)) if s == base_repo => {}
            other => {
                return Err(invalid(format!(
                    "base_model_name_or_path is {}, but the manifest's base is {base_repo}",
                    other.map_or_else(|| "missing".to_string(), Json::to_canonical_string)
                )));
            }
        }
        Ok(LoraConfig {
            r,
            alpha,
            target_modules,
        })
    }

    /// The scale `alpha / r` as the `f32` the merge multiplies by.
    pub fn scale(&self) -> f32 {
        (self.alpha / self.r as f64) as f32
    }

    /// Whether the module with this last path component carries a LoRA.
    pub fn targets(&self, module_name: &str) -> bool {
        self.target_modules.iter().any(|t| t == module_name)
    }
}

/// Which half of a LoRA pair an adapter tensor is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    /// `lora_A`, shape `[r, in]`.
    A,
    /// `lora_B`, shape `[out, r]`.
    B,
}

/// For an adapter tensor name such as `base_model.model.layers.3.mlp.down_proj.lora_A.weight`
/// the name of the base weight it adapts (`model.language_model.layers.3.mlp.down_proj.weight`)
/// and which half it is. `None` for any other name.
pub fn base_of(adapter_tensor: &str) -> Option<(String, Half)> {
    let rest = adapter_tensor.strip_prefix("base_model.model.")?;
    let (module, half) = if let Some(m) = rest.strip_suffix(".lora_A.weight") {
        (m, Half::A)
    } else {
        (rest.strip_suffix(".lora_B.weight")?, Half::B)
    };
    Some((format!("model.language_model.{module}.weight"), half))
}
