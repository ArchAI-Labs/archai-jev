//! The table of supported families (D19): architecture + head + prompt template.
//!
//! This is the only place that knows family names. Everything else asks the table: how many
//! tensors with which names, shapes and types are expected, which special-token roles the
//! template needs, which question kinds the head can answer.

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
    /// An array of booleans equal to this.
    BoolArray(Vec<bool>),
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
    /// Longest prefix, in tokens, whose saved state is bit-identical to a cold decode, or `None`
    /// when any length is (families with a sliding attention window: the window).
    pub snapshot_max_tokens: fn(&ArchParams) -> Option<u64>,
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

fn qwen35_tensors(p: &ArchParams) -> Vec<TensorSpec> {
    crate::convert::layout::Dims::from_params(p)
        .entries()
        .into_iter()
        .map(|e| TensorSpec {
            name: e.gguf,
            dims: e.dims,
            role: e.role,
        })
        .collect()
}

fn qwen35_metadata(p: &ArchParams) -> Vec<(String, MetaExpect)> {
    let uint = |key: &str, v: u64| (key.to_string(), MetaExpect::UInt(v));
    vec![
        (
            "general.architecture".to_string(),
            MetaExpect::Str("qwen35".to_string()),
        ),
        uint(
            "qwen35.block_count",
            p.int("n_layers") + p.int("n_mtp_layers"),
        ),
        uint("qwen35.embedding_length", p.int("n_embd")),
        uint("qwen35.feed_forward_length", p.int("n_ff")),
        uint("qwen35.attention.head_count", p.int("n_heads")),
        uint("qwen35.attention.head_count_kv", p.int("n_kv_heads")),
        uint("qwen35.attention.key_length", p.int("head_dim")),
        uint("qwen35.attention.value_length", p.int("head_dim")),
        uint("qwen35.nextn_predict_layers", p.int("n_mtp_layers")),
        uint("qwen35.ssm.conv_kernel", p.int("ssm_conv_kernel")),
        uint("qwen35.ssm.state_size", p.int("ssm_state_size")),
        uint("qwen35.ssm.group_count", p.int("ssm_group_count")),
        uint("qwen35.ssm.time_step_rank", p.int("ssm_time_step_rank")),
        uint("qwen35.ssm.inner_size", p.int("ssm_inner_size")),
        uint(
            "qwen35.full_attention_interval",
            p.int("full_attention_interval"),
        ),
        uint("qwen35.rope.dimension_count", p.int("rope_dim")),
    ]
}

const QWEN35_PARAMS: &[(&str, ParamKind)] = &[
    ("n_layers", ParamKind::Int),
    ("n_mtp_layers", ParamKind::Int),
    ("n_embd", ParamKind::Int),
    ("n_ff", ParamKind::Int),
    ("n_heads", ParamKind::Int),
    ("n_kv_heads", ParamKind::Int),
    ("head_dim", ParamKind::Int),
    ("n_vocab", ParamKind::Int),
    ("tie_embeddings", ParamKind::Bool),
    ("full_attention_interval", ParamKind::Int),
    ("ssm_conv_kernel", ParamKind::Int),
    ("ssm_state_size", ParamKind::Int),
    ("ssm_group_count", ParamKind::Int),
    ("ssm_time_step_rank", ParamKind::Int),
    ("ssm_inner_size", ParamKind::Int),
    ("rope_theta", ParamKind::Int),
    ("rope_dim", ParamKind::Int),
];

fn no_snapshot_limit(_: &ArchParams) -> Option<u64> {
    None
}

/// Tensors of the `spark2_5` architecture, as written in the published GGUF files (1.7B: 226
/// tensors, 4B: 290): a fused QKV matrix, one attention gate per head, no biases, and no
/// `output.weight` when the embeddings are tied.
fn spark2_5_tensors(p: &ArchParams) -> Vec<TensorSpec> {
    let (layers, d, ff, heads, kv, hd) = (
        p.int("n_layers"),
        p.int("n_embd"),
        p.int("n_ff"),
        p.int("n_heads"),
        p.int("n_kv_heads"),
        p.int("head_dim"),
    );
    let vocab = p.int("n_vocab");
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
        out.push(t(
            b("attn_qkv.weight"),
            &[d, hd * (heads + 2 * kv)],
            Role::Matrix,
        ));
        out.push(t(b("attn_gate.weight"), &[d, heads], Role::Matrix));
        out.push(t(b("attn_output.weight"), &[hd * heads, d], Role::Matrix));
        out.push(t(b("ffn_norm.weight"), &[d], Role::Vector));
        out.push(t(b("ffn_gate.weight"), &[d, ff], Role::Matrix));
        out.push(t(b("ffn_up.weight"), &[d, ff], Role::Matrix));
        out.push(t(b("ffn_down.weight"), &[ff, d], Role::Matrix));
    }
    out
}

/// Which layers use the sliding window: all but every `full_attention_every`-th one (the
/// published models have a full-attention layer after every three sliding ones).
fn spark2_5_sliding_pattern(p: &ArchParams) -> Vec<bool> {
    let every = p.int("full_attention_every");
    (0..p.int("n_layers"))
        .map(|i| every == 0 || (i + 1) % every != 0)
        .collect()
}

fn spark2_5_metadata(p: &ArchParams) -> Vec<(String, MetaExpect)> {
    let uint = |key: &str, v: u64| (key.to_string(), MetaExpect::UInt(v));
    vec![
        (
            "general.architecture".to_string(),
            MetaExpect::Str("spark2_5".to_string()),
        ),
        uint("spark2_5.block_count", p.int("n_layers")),
        uint("spark2_5.embedding_length", p.int("n_embd")),
        uint("spark2_5.feed_forward_length", p.int("n_ff")),
        uint("spark2_5.attention.head_count", p.int("n_heads")),
        uint("spark2_5.attention.head_count_kv", p.int("n_kv_heads")),
        uint("spark2_5.attention.key_length", p.int("head_dim")),
        uint("spark2_5.attention.value_length", p.int("head_dim")),
        uint("spark2_5.vocab_size", p.int("n_vocab")),
        uint("spark2_5.attention.sliding_window", p.int("sliding_window")),
        (
            "spark2_5.attention.sliding_window_pattern".to_string(),
            MetaExpect::BoolArray(spark2_5_sliding_pattern(p)),
        ),
    ]
}

fn spark2_5_snapshot_max(p: &ArchParams) -> Option<u64> {
    Some(p.int("sliding_window"))
}

const SPARK2_5_PARAMS: &[(&str, ParamKind)] = &[
    ("n_layers", ParamKind::Int),
    ("n_embd", ParamKind::Int),
    ("n_ff", ParamKind::Int),
    ("n_heads", ParamKind::Int),
    ("n_kv_heads", ParamKind::Int),
    ("head_dim", ParamKind::Int),
    ("n_vocab", ParamKind::Int),
    ("tie_embeddings", ParamKind::Bool),
    ("sliding_window", ParamKind::Int),
    ("full_attention_every", ParamKind::Int),
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
static FAMILIES: &[Family] = &[
    Family {
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
        snapshot_max_tokens: no_snapshot_limit,
    },
    Family {
        key: "qwen2-keys",
        arch: "qwen2",
        head_kind: "key_logprob",
        template_id: "decision-index-chatml",
        template_version: 1,
        special_roles: &["im_start", "im_end"],
        question_types: &["choice", "noul"],
        params: QWEN2_PARAMS,
        tensors: qwen2_tensors,
        metadata: qwen2_metadata,
        context_key: "qwen2.context_length",
        n_vocab: qwen2_vocab,
        snapshot_max_tokens: no_snapshot_limit,
    },
    Family {
        key: "qwen35-pointer",
        arch: "qwen35",
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
        params: QWEN35_PARAMS,
        tensors: qwen35_tensors,
        metadata: qwen35_metadata,
        context_key: "qwen35.context_length",
        n_vocab: qwen2_vocab,
        snapshot_max_tokens: no_snapshot_limit,
    },
    Family {
        key: "spark2_5-letters",
        arch: "spark2_5",
        head_kind: "answer_letters",
        template_id: "spark-decisions",
        template_version: 3,
        special_roles: &["bos", "system", "user", "bot", "eos", "think_end"],
        question_types: &["choice", "noul"],
        params: SPARK2_5_PARAMS,
        tensors: spark2_5_tensors,
        metadata: spark2_5_metadata,
        context_key: "spark2_5.context_length",
        n_vocab: qwen2_vocab,
        snapshot_max_tokens: spark2_5_snapshot_max,
    },
];

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
        snapshot_max_tokens: no_snapshot_limit,
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
        snapshot_max_tokens: no_snapshot_limit,
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

    /// The two published sizes, read from the headers of the real files (S13, 2026-10-18).
    fn spark_params(layers: u64, d: u64, ff: u64, heads: u64, kv: u64, tie: bool) -> ArchParams {
        ArchParams::from_pairs(vec![
            ("n_layers".into(), ParamValue::Int(layers)),
            ("n_embd".into(), ParamValue::Int(d)),
            ("n_ff".into(), ParamValue::Int(ff)),
            ("n_heads".into(), ParamValue::Int(heads)),
            ("n_kv_heads".into(), ParamValue::Int(kv)),
            ("head_dim".into(), ParamValue::Int(256)),
            ("n_vocab".into(), ParamValue::Int(131_072)),
            ("tie_embeddings".into(), ParamValue::Bool(tie)),
            ("sliding_window".into(), ParamValue::Int(512)),
            ("full_attention_every".into(), ParamValue::Int(4)),
        ])
    }

    fn small(tie: bool) -> ArchParams {
        spark_params(28, 2048, 6656, 8, 2, tie)
    }

    fn large(tie: bool) -> ArchParams {
        spark_params(36, 2560, 10_240, 16, 4, tie)
    }

    #[test]
    fn spark_tables_have_the_tensor_counts_of_the_published_files() {
        // 2 + 8 tensors per layer, embeddings tied: 226 (28 layers) and 290 (36 layers).
        assert_eq!(spark2_5_tensors(&small(true)).len(), 226);
        assert_eq!(spark2_5_tensors(&large(true)).len(), 290);
        assert_eq!(spark2_5_tensors(&small(false)).len(), 227);
        assert!(
            !spark2_5_tensors(&small(true))
                .iter()
                .any(|t| t.name == "output.weight")
        );
        // no biases and no separate q/k/v: that is what the converter writes
        let names: Vec<String> = spark2_5_tensors(&small(true))
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(
            !names
                .iter()
                .any(|n| n.contains("bias") || n.contains("attn_q."))
        );
    }

    #[test]
    fn spark_shapes_are_the_ones_of_the_real_files() {
        let find = |params: &ArchParams, name: &str| {
            spark2_5_tensors(params)
                .into_iter()
                .find(|t| t.name == name)
                .map(|t| (t.dims, t.role))
        };
        let m = Role::Matrix;
        assert_eq!(
            find(&small(true), "blk.0.attn_qkv.weight"),
            Some((vec![2048, 3072], m))
        );
        assert_eq!(
            find(&small(true), "blk.27.attn_gate.weight"),
            Some((vec![2048, 8], m))
        );
        assert_eq!(
            find(&small(true), "blk.3.attn_output.weight"),
            Some((vec![2048, 2048], m))
        );
        assert_eq!(
            find(&small(true), "blk.3.ffn_down.weight"),
            Some((vec![6656, 2048], m))
        );
        assert_eq!(
            find(&large(true), "blk.3.attn_qkv.weight"),
            Some((vec![2560, 6144], m))
        );
        assert_eq!(
            find(&large(true), "blk.35.attn_gate.weight"),
            Some((vec![2560, 16], m))
        );
        assert_eq!(
            find(&large(true), "blk.3.attn_output.weight"),
            Some((vec![4096, 2560], m))
        );
        assert_eq!(
            find(&large(true), "blk.0.ffn_norm.weight"),
            Some((vec![2560], Role::Vector))
        );
        assert_eq!(
            find(&large(true), "token_embd.weight"),
            Some((vec![2560, 131_072], m))
        );
        assert_eq!(find(&large(true), "blk.36.attn_qkv.weight"), None);
    }

    #[test]
    fn spark_pattern_is_three_sliding_layers_then_a_full_one() {
        let pattern = spark2_5_sliding_pattern(&large(true));
        assert_eq!(pattern.len(), 36);
        assert_eq!(pattern.iter().filter(|s| **s).count(), 27);
        assert_eq!(pattern.iter().filter(|s| !**s).count(), 9);
        assert_eq!(
            &pattern[..8],
            &[true, true, true, false, true, true, true, false]
        );
        assert!(!pattern[35]);
        let metadata = spark2_5_metadata(&small(true));
        let (_, expect) = metadata
            .iter()
            .find(|(k, _)| k == "spark2_5.attention.sliding_window_pattern")
            .expect("the pattern is checked");
        assert_eq!(
            expect,
            &MetaExpect::BoolArray(spark2_5_sliding_pattern(&small(true)))
        );
    }

    #[test]
    fn spark_snapshots_stop_at_the_window_and_the_others_have_no_limit() {
        let spark = lookup("spark2_5-letters").expect("family");
        assert_eq!((spark.snapshot_max_tokens)(&small(true)), Some(512));
        let qwen = lookup("qwen2-letters").expect("family");
        assert_eq!((qwen.snapshot_max_tokens)(&params(28, true)), None);
        assert_eq!(
            (spark.head_kind, spark.template_id, spark.template_version),
            ("answer_letters", "spark-decisions", 3)
        );
    }

    #[test]
    fn spark_metadata_names_every_size_of_the_architecture() {
        let keys: Vec<String> = spark2_5_metadata(&large(true))
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        for want in [
            "general.architecture",
            "spark2_5.block_count",
            "spark2_5.embedding_length",
            "spark2_5.feed_forward_length",
            "spark2_5.attention.head_count",
            "spark2_5.attention.head_count_kv",
            "spark2_5.attention.key_length",
            "spark2_5.attention.value_length",
            "spark2_5.vocab_size",
            "spark2_5.attention.sliding_window",
            "spark2_5.attention.sliding_window_pattern",
        ] {
            assert!(keys.iter().any(|k| k == want), "{want} is not checked");
        }
    }

    #[test]
    fn lookup_and_keys() {
        assert!(lookup("qwen2-letters").is_some());
        assert!(lookup("spark2_5-letters").is_some());
        assert!(lookup("nope").is_none());
        assert!(keys().contains(&"qwen2-letters".to_string()));
    }
}
