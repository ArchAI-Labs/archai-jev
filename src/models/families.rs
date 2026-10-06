//! The table of supported families (D19): architecture + head + prompt template.
//!
//! This is the only place that knows family names. Everything else asks the table: how many
//! tensors with which names, shapes and types are expected, which special-token roles the
//! template needs, which question kinds the head can answer.

use super::gguf::GgmlType;
use super::incompat::Incompat;
use crate::json_strict::Json;

/// A structural parameter of an architecture, as read from the manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchParams(Vec<(String, ParamValue)>);

/// Value of an architecture parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamValue {
    /// A non-negative integer.
    Int(u64),
    /// A boolean.
    Bool(bool),
}

/// Kind of an architecture parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Non-negative integer.
    Int,
    /// Boolean.
    Bool,
}

impl ArchParams {
    /// An integer parameter (0 if absent: absence is rejected when the table is applied).
    pub fn int(&self, name: &str) -> u64 {
        match self.0.iter().find(|(k, _)| k == name) {
            Some((_, ParamValue::Int(n))) => *n,
            _ => 0,
        }
    }

    /// A boolean parameter (false if absent).
    pub fn flag(&self, name: &str) -> bool {
        matches!(
            self.0.iter().find(|(k, _)| k == name),
            Some((_, ParamValue::Bool(true)))
        )
    }

    /// Build parameters from `(name, value)` pairs.
    pub fn from_pairs(pairs: Vec<(String, ParamValue)>) -> Self {
        ArchParams(pairs)
    }
}

/// Whether a tensor is a weight matrix (stored in the variant's dtype) or a vector (norms and
/// biases, always F32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Stored with the dtype of the variant.
    Matrix,
    /// Always F32.
    Vector,
}

/// A tensor a model file of this family must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// GGUF tensor name.
    pub name: String,
    /// Dimensions in GGUF order (`ne[0]` first).
    pub dims: Vec<u64>,
    /// Matrix or vector.
    pub role: Role,
}

/// Expectation on one GGUF metadata key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaExpect {
    /// A string equal to this.
    Str(String),
    /// An unsigned integer equal to this.
    UInt(u64),
}

/// One supported family.
pub struct Family {
    /// The key used in manifests.
    pub key: &'static str,
    /// Architecture name (also the GGUF `general.architecture`).
    pub arch: &'static str,
    /// Head kind (`letters` or `pointer`).
    pub head_kind: &'static str,
    /// Template id.
    pub template_id: &'static str,
    /// Template version.
    pub template_version: u64,
    /// Special-token roles the template needs.
    pub special_roles: &'static [&'static str],
    /// Question types the head can answer (`choice`, `noul`, `score`).
    pub question_types: &'static [&'static str],
    /// Architecture parameters the manifest must give.
    pub params: &'static [(&'static str, ParamKind)],
    /// Expected tensors.
    pub tensors: fn(&ArchParams) -> Vec<TensorSpec>,
    /// Expected GGUF metadata.
    pub metadata: fn(&ArchParams) -> Vec<(String, MetaExpect)>,
    /// GGUF key holding the context length of the model.
    pub context_key: &'static str,
    /// Vocabulary size (rows of the embedding).
    pub n_vocab: fn(&ArchParams) -> u64,
}

fn t(name: impl Into<String>, dims: &[u64], role: Role) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dims: dims.to_vec(),
        role,
    }
}

fn qwen2_tensors(p: &ArchParams) -> Vec<TensorSpec> {
    let (layers, d, ff, heads, kv) = (
        p.int("n_layers"),
        p.int("n_embd"),
        p.int("n_ff"),
        p.int("n_heads").max(1),
        p.int("n_kv_heads"),
    );
    let vocab = p.int("n_vocab");
    let kv_dim = d / heads * kv;
    let mut out = vec![
        t("token_embd.weight", &[d, vocab], Role::Matrix),
        t("output_norm.weight", &[d], Role::Vector),
    ];
    if !p.flag("tie_embeddings") {
        out.push(t("output.weight", &[d, vocab], Role::Matrix));
    }
    for i in 0..layers {
        let b = |s: &str| format!("blk.{i}.{s}");
        out.push(t(b("attn_norm.weight"), &[d], Role::Vector));
        out.push(t(b("attn_q.weight"), &[d, d], Role::Matrix));
        out.push(t(b("attn_q.bias"), &[d], Role::Vector));
        out.push(t(b("attn_k.weight"), &[d, kv_dim], Role::Matrix));
        out.push(t(b("attn_k.bias"), &[kv_dim], Role::Vector));
        out.push(t(b("attn_v.weight"), &[d, kv_dim], Role::Matrix));
        out.push(t(b("attn_v.bias"), &[kv_dim], Role::Vector));
        out.push(t(b("attn_output.weight"), &[d, d], Role::Matrix));
        out.push(t(b("ffn_norm.weight"), &[d], Role::Vector));
        out.push(t(b("ffn_gate.weight"), &[d, ff], Role::Matrix));
        out.push(t(b("ffn_up.weight"), &[d, ff], Role::Matrix));
        out.push(t(b("ffn_down.weight"), &[ff, d], Role::Matrix));
    }
    out
}

fn qwen2_metadata(p: &ArchParams) -> Vec<(String, MetaExpect)> {
    vec![
        (
            "general.architecture".to_string(),
            MetaExpect::Str("qwen2".to_string()),
        ),
        (
            "qwen2.block_count".to_string(),
            MetaExpect::UInt(p.int("n_layers")),
        ),
        (
            "qwen2.embedding_length".to_string(),
            MetaExpect::UInt(p.int("n_embd")),
        ),
        (
            "qwen2.feed_forward_length".to_string(),
            MetaExpect::UInt(p.int("n_ff")),
        ),
        (
            "qwen2.attention.head_count".to_string(),
            MetaExpect::UInt(p.int("n_heads")),
        ),
        (
            "qwen2.attention.head_count_kv".to_string(),
            MetaExpect::UInt(p.int("n_kv_heads")),
        ),
    ]
}

fn qwen2_vocab(p: &ArchParams) -> u64 {
    p.int("n_vocab")
}

const QWEN2_PARAMS: &[(&str, ParamKind)] = &[
    ("n_layers", ParamKind::Int),
    ("n_embd", ParamKind::Int),
    ("n_ff", ParamKind::Int),
    ("n_heads", ParamKind::Int),
    ("n_kv_heads", ParamKind::Int),
    ("n_vocab", ParamKind::Int),
    ("tie_embeddings", ParamKind::Bool),
];

#[cfg(feature = "testing")]
fn fake_tensors(p: &ArchParams) -> Vec<TensorSpec> {
    vec![
        t(
            "w.weight",
            &[p.int("n_embd"), p.int("n_vocab")],
            Role::Matrix,
        ),
        t("norm.weight", &[p.int("n_embd")], Role::Vector),
    ]
}

#[cfg(feature = "testing")]
fn fake_metadata(p: &ArchParams) -> Vec<(String, MetaExpect)> {
    vec![
        (
            "general.architecture".to_string(),
            MetaExpect::Str("fakearch".to_string()),
        ),
        (
            "fakearch.embedding_length".to_string(),
            MetaExpect::UInt(p.int("n_embd")),
        ),
    ]
}

#[cfg(feature = "testing")]
const FAKE_PARAMS: &[(&str, ParamKind)] =
    &[("n_embd", ParamKind::Int), ("n_vocab", ParamKind::Int)];

/// Families of the production table.
static FAMILIES: &[Family] = &[Family {
    key: "qwen2-letters",
    arch: "qwen2",
    head_kind: "letters",
    template_id: "chatml-letters",
    template_version: 1,
    special_roles: &["im_start", "im_end"],
    question_types: &["choice", "noul"],
    params: QWEN2_PARAMS,
    tensors: qwen2_tensors,
    metadata: qwen2_metadata,
    context_key: "qwen2.context_length",
    n_vocab: qwen2_vocab,
}];

/// Extra families that exist only in test builds (feature `testing`): they prove the loader
/// is generic and let Python tests run the whole pipeline with the fake backend.
#[cfg(feature = "testing")]
static TEST_FAMILIES: &[Family] = &[
    Family {
        key: "test-kev",
        arch: "fakearch",
        head_kind: "pointer",
        template_id: "kev",
        template_version: 1,
        special_roles: &[
            "fim_prefix",
            "fim_middle",
            "fim_suffix",
            "box_start",
            "box_end",
        ],
        question_types: &["choice", "noul", "score"],
        params: FAKE_PARAMS,
        tensors: fake_tensors,
        metadata: fake_metadata,
        context_key: "fakearch.context_length",
        n_vocab: qwen2_vocab,
    },
    Family {
        key: "test-fake",
        arch: "fakearch",
        head_kind: "letters",
        template_id: "fake-v1",
        template_version: 1,
        special_roles: &["open", "close"],
        question_types: &["choice", "noul", "score"],
        params: FAKE_PARAMS,
        tensors: fake_tensors,
        metadata: fake_metadata,
        context_key: "fakearch.context_length",
        n_vocab: qwen2_vocab,
    },
];

fn all() -> impl Iterator<Item = &'static Family> {
    #[cfg(feature = "testing")]
    let extra = TEST_FAMILIES.iter();
    #[cfg(not(feature = "testing"))]
    let extra = [].iter();
    FAMILIES.iter().chain(extra)
}

/// The family called `key`.
pub fn lookup(key: &str) -> Option<&'static Family> {
    all().find(|f| f.key == key)
}

/// The keys of all supported families.
pub fn keys() -> Vec<String> {
    all().map(|f| f.key.to_string()).collect()
}

/// The ggml type a weight matrix has for `dtype`.
pub fn matrix_type(dtype: &str) -> Option<GgmlType> {
    match dtype {
        "f32" => Some(GgmlType::F32),
        "bf16" => Some(GgmlType::BF16),
        "q8_0" => Some(GgmlType::Q8_0),
        _ => None,
    }
}

impl Family {
    /// Read and check the architecture parameters of a manifest against this family.
    ///
    /// # Errors
    /// Missing, unknown or mistyped parameters.
    pub fn read_params(
        &self,
        arch: &super::manifest::ArchitectureSpec,
    ) -> Result<ArchParams, Incompat> {
        let mut out = Vec::new();
        for (name, kind) in self.params {
            let path = format!("architecture.{name}");
            let Some((_, value)) = arch.params.iter().find(|(k, _)| k == name) else {
                return Err(Incompat::ManifestMissingField { path });
            };
            let parsed = match (kind, value) {
                (ParamKind::Int, Json::Int(n)) if *n >= 0 => {
                    u64::try_from(*n).ok().map(ParamValue::Int)
                }
                (ParamKind::Bool, Json::Bool(b)) => Some(ParamValue::Bool(*b)),
                _ => None,
            };
            let Some(v) = parsed else {
                return Err(Incompat::ManifestBadValue {
                    path,
                    detail: format!(
                        "must be {}, got {}",
                        match kind {
                            ParamKind::Int => "a non-negative integer",
                            ParamKind::Bool => "a boolean",
                        },
                        value.kind()
                    ),
                });
            };
            out.push(((*name).to_string(), v));
        }
        if let Some((extra, _)) = arch
            .params
            .iter()
            .find(|(k, _)| !self.params.iter().any(|(n, _)| n == k))
        {
            return Err(Incompat::ManifestUnknownField {
                path: "architecture".to_string(),
                field: extra.clone(),
            });
        }
        Ok(ArchParams(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(layers: u64, tie: bool) -> ArchParams {
        ArchParams::from_pairs(vec![
            ("n_layers".into(), ParamValue::Int(layers)),
            ("n_embd".into(), ParamValue::Int(1536)),
            ("n_ff".into(), ParamValue::Int(8960)),
            ("n_heads".into(), ParamValue::Int(12)),
            ("n_kv_heads".into(), ParamValue::Int(2)),
            ("n_vocab".into(), ParamValue::Int(151_936)),
            ("tie_embeddings".into(), ParamValue::Bool(tie)),
        ])
    }

    #[test]
    fn qwen2_28_layers_has_338_tensors_with_tied_embeddings() {
        // S5: the default model's GGUF has 338 tensors and no output.weight.
        let tensors = qwen2_tensors(&params(28, true));
        assert_eq!(tensors.len(), 338);
        assert!(!tensors.iter().any(|t| t.name == "output.weight"));
        let matrices = tensors.iter().filter(|t| t.role == Role::Matrix).count();
        assert_eq!((matrices, tensors.len() - matrices), (197, 141));
        assert_eq!(qwen2_tensors(&params(28, false)).len(), 339);
    }

    #[test]
    fn lookup_and_keys() {
        assert!(lookup("qwen2-letters").is_some());
        assert!(lookup("nope").is_none());
        assert!(keys().contains(&"qwen2-letters".to_string()));
    }

    #[test]
    fn matrix_types() {
        assert_eq!(matrix_type("q8_0"), Some(GgmlType::Q8_0));
        assert_eq!(matrix_type("q4_k_m"), None);
    }
}
