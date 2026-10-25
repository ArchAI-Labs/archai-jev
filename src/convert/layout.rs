//! The tensors of a Qwen3.5 GGUF and where each comes from (spec 018, 3.2).
//!
//! One function lists, from the dimensions of the model, **every** tensor the GGUF must have: its
//! name in the Hugging Face checkpoint, its name in the GGUF, its shapes, how the data is
//! transformed on the way and whether it is stored in the variant's type or always as `f32`.
//! The converter builds its plan from it and the table of families (`qwen35-pointer`) builds the
//! expected set for the check of the GGUF header from the same function, so the two cannot drift.
//! The names and rules come from `Qwen3_5TextModel` of llama.cpp and were checked tensor by
//! tensor against the GGUF its script wrote for Kev-0.8B (335 tensors).

use super::config::Hparams;
use super::safetensors::Dtype;
use crate::models::families::{ArchParams, Role};

/// What happens to the numbers of a tensor between the checkpoint and the GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Nothing (the type may change).
    Copy,
    /// `-exp(x)` (`A_log`).
    NegExp,
    /// `x + 1` (the weights of the norms, which the checkpoint stores minus one).
    PlusOne,
    /// The dimension of size 1 is dropped (`conv1d`); the numbers are the same.
    Squeeze,
}

/// One tensor of the GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Name in the checkpoint.
    pub hf: String,
    /// Name in the GGUF.
    pub gguf: String,
    /// Shape in the checkpoint (row-major, outermost first).
    pub hf_shape: Vec<u64>,
    /// Dimensions in the GGUF (`ne[0]` first, the reverse of the row-major shape).
    pub dims: Vec<u64>,
    /// The type the checkpoint stores it in.
    pub hf_dtype: Dtype,
    /// What happens to the numbers.
    pub transform: Transform,
    /// `Matrix` is stored in the variant's type, `Vector` always as `f32`.
    pub role: Role,
    /// The last component of the module name (`q_proj`), for weights a LoRA may adapt.
    pub module: Option<&'static str>,
}

/// The dimensions of the model that decide its tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dims {
    /// Text layers.
    pub n_layers: u64,
    /// MTP layers.
    pub n_mtp: u64,
    /// Hidden size.
    pub hidden: u64,
    /// Feed-forward size.
    pub ff: u64,
    /// Attention heads.
    pub heads: u64,
    /// Key/value heads.
    pub kv_heads: u64,
    /// Head size.
    pub head_dim: u64,
    /// Rows of the embedding.
    pub vocab: u64,
    /// Every how many layers there is a full attention layer.
    pub interval: u64,
    /// Kernel of the convolution of the linear attention.
    pub conv_kernel: u64,
    /// Key head size of the linear attention (`ssm.state_size`).
    pub key_head_dim: u64,
    /// Key heads of the linear attention (`ssm.group_count`).
    pub key_heads: u64,
    /// Value heads of the linear attention (`ssm.time_step_rank`).
    pub value_heads: u64,
    /// `ssm.inner_size`: value head size times value heads.
    pub inner: u64,
}

impl Dims {
    /// From the `architecture` of a manifest.
    pub fn from_params(p: &ArchParams) -> Dims {
        Dims {
            n_layers: p.int("n_layers"),
            n_mtp: p.int("n_mtp_layers"),
            hidden: p.int("n_embd"),
            ff: p.int("n_ff"),
            heads: p.int("n_heads"),
            kv_heads: p.int("n_kv_heads"),
            head_dim: p.int("head_dim"),
            vocab: p.int("n_vocab"),
            interval: p.int("full_attention_interval").max(1),
            conv_kernel: p.int("ssm_conv_kernel"),
            key_head_dim: p.int("ssm_state_size"),
            key_heads: p.int("ssm_group_count"),
            value_heads: p.int("ssm_time_step_rank"),
            inner: p.int("ssm_inner_size"),
        }
    }

    /// From `config.json`.
    pub fn from_hparams(h: &Hparams) -> Dims {
        Dims {
            n_layers: h.n_layers,
            n_mtp: h.n_mtp,
            hidden: h.hidden,
            ff: h.ff,
            heads: h.heads,
            kv_heads: h.kv_heads,
            head_dim: h.head_dim,
            vocab: h.vocab,
            interval: h.full_attention_interval,
            conv_kernel: h.conv_kernel,
            key_head_dim: h.key_head_dim,
            key_heads: h.key_heads,
            value_heads: h.value_heads,
            inner: h.value_head_dim * h.value_heads,
        }
    }

    /// Channels of the convolution: queries, keys and values of the linear attention.
    pub fn conv_dim(&self) -> u64 {
        2 * self.key_heads * self.key_head_dim + self.inner
    }

    /// Size of one value head.
    pub fn value_head_dim(&self) -> u64 {
        self.inner.checked_div(self.value_heads).unwrap_or(0)
    }

    /// Whether text layer `i` is a full attention layer (the MTP layer always is).
    pub fn is_full(&self, i: u64) -> bool {
        i >= self.n_layers || (i + 1).is_multiple_of(self.interval)
    }

    /// Every tensor of the GGUF, in the order the converter writes them.
    pub fn entries(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        let mut add = |hf: String,
                       gguf: String,
                       hf_shape: Vec<u64>,
                       dtype: Dtype,
                       transform: Transform,
                       module: Option<&'static str>| {
            let dims: Vec<u64> = if transform == Transform::Squeeze {
                // [channels, 1, kernel] -> ggml [kernel, channels]
                hf_shape
                    .first()
                    .zip(hf_shape.last())
                    .map(|(c, k)| vec![*k, *c])
                    .unwrap_or_default()
            } else {
                hf_shape.iter().rev().copied().collect()
            };
            let always_f32 = hf_shape.len() == 1
                || gguf.ends_with("_norm.weight")
                || transform == Transform::Squeeze;
            out.push(Entry {
                hf,
                gguf,
                hf_shape,
                dims,
                hf_dtype: dtype,
                transform,
                role: if always_f32 {
                    Role::Vector
                } else {
                    Role::Matrix
                },
                module,
            });
        };
        let (d, ff) = (self.hidden, self.ff);
        let q_rows = self.heads * self.head_dim * 2;
        let kv_rows = self.kv_heads * self.head_dim;
        let attn_out = self.heads * self.head_dim;
        let norm_plus = |hf: &str| -> Transform {
            if hf.ends_with("norm.weight") && !hf.ends_with("linear_attn.norm.weight") {
                Transform::PlusOne
            } else {
                Transform::Copy
            }
        };

        add(
            "model.language_model.embed_tokens.weight".into(),
            "token_embd.weight".into(),
            vec![self.vocab, d],
            Dtype::Bf16,
            Transform::Copy,
            None,
        );
        // One block per layer: the text layers, then the MTP ones (named `mtp.layers.N` in the
        // checkpoint and `blk.<n_layers + N>` in the GGUF).
        for layer in 0..self.n_layers + self.n_mtp {
            let (prefix, blk) = if layer < self.n_layers {
                (
                    format!("model.language_model.layers.{layer}"),
                    format!("blk.{layer}"),
                )
            } else {
                (
                    format!("mtp.layers.{}", layer - self.n_layers),
                    format!("blk.{layer}"),
                )
            };
            let mut push =
                |hf_tail: &str, gguf_tail: &str, shape: Vec<u64>, dtype, transform, module| {
                    let hf = format!("{prefix}.{hf_tail}");
                    let transform = if transform == Transform::Copy {
                        norm_plus(&hf)
                    } else {
                        transform
                    };
                    add(
                        hf,
                        format!("{blk}.{gguf_tail}"),
                        shape,
                        dtype,
                        transform,
                        module,
                    );
                };
            push(
                "input_layernorm.weight",
                "attn_norm.weight",
                vec![d],
                Dtype::Bf16,
                Transform::Copy,
                None,
            );
            push(
                "post_attention_layernorm.weight",
                "post_attention_norm.weight",
                vec![d],
                Dtype::Bf16,
                Transform::Copy,
                None,
            );
            push(
                "mlp.gate_proj.weight",
                "ffn_gate.weight",
                vec![ff, d],
                Dtype::Bf16,
                Transform::Copy,
                Some("gate_proj"),
            );
            push(
                "mlp.up_proj.weight",
                "ffn_up.weight",
                vec![ff, d],
                Dtype::Bf16,
                Transform::Copy,
                Some("up_proj"),
            );
            push(
                "mlp.down_proj.weight",
                "ffn_down.weight",
                vec![d, ff],
                Dtype::Bf16,
                Transform::Copy,
                Some("down_proj"),
            );
            if self.is_full(layer) {
                push(
                    "self_attn.q_proj.weight",
                    "attn_q.weight",
                    vec![q_rows, d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("q_proj"),
                );
                push(
                    "self_attn.k_proj.weight",
                    "attn_k.weight",
                    vec![kv_rows, d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("k_proj"),
                );
                push(
                    "self_attn.v_proj.weight",
                    "attn_v.weight",
                    vec![kv_rows, d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("v_proj"),
                );
                push(
                    "self_attn.o_proj.weight",
                    "attn_output.weight",
                    vec![d, attn_out],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("o_proj"),
                );
                push(
                    "self_attn.q_norm.weight",
                    "attn_q_norm.weight",
                    vec![self.head_dim],
                    Dtype::Bf16,
                    Transform::Copy,
                    None,
                );
                push(
                    "self_attn.k_norm.weight",
                    "attn_k_norm.weight",
                    vec![self.head_dim],
                    Dtype::Bf16,
                    Transform::Copy,
                    None,
                );
            } else {
                push(
                    "linear_attn.in_proj_qkv.weight",
                    "attn_qkv.weight",
                    vec![self.conv_dim(), d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("in_proj_qkv"),
                );
                push(
                    "linear_attn.in_proj_z.weight",
                    "attn_gate.weight",
                    vec![self.inner, d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("in_proj_z"),
                );
                push(
                    "linear_attn.in_proj_a.weight",
                    "ssm_alpha.weight",
                    vec![self.value_heads, d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("in_proj_a"),
                );
                push(
                    "linear_attn.in_proj_b.weight",
                    "ssm_beta.weight",
                    vec![self.value_heads, d],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("in_proj_b"),
                );
                push(
                    "linear_attn.conv1d.weight",
                    "ssm_conv1d.weight",
                    vec![self.conv_dim(), 1, self.conv_kernel],
                    Dtype::Bf16,
                    Transform::Squeeze,
                    None,
                );
                push(
                    "linear_attn.dt_bias",
                    "ssm_dt.bias",
                    vec![self.value_heads],
                    Dtype::Bf16,
                    Transform::Copy,
                    None,
                );
                push(
                    "linear_attn.A_log",
                    "ssm_a",
                    vec![self.value_heads],
                    Dtype::F32,
                    Transform::NegExp,
                    None,
                );
                push(
                    "linear_attn.norm.weight",
                    "ssm_norm.weight",
                    vec![self.value_head_dim()],
                    Dtype::F32,
                    Transform::Copy,
                    None,
                );
                push(
                    "linear_attn.out_proj.weight",
                    "ssm_out.weight",
                    vec![d, self.inner],
                    Dtype::Bf16,
                    Transform::Copy,
                    Some("out_proj"),
                );
            }
            if layer >= self.n_layers {
                // the rest of the MTP block: the projection and the three norms around it
                let n = self.n_layers;
                let mut mtp = |hf: &str, gguf: &str, shape: Vec<u64>, module| {
                    // the official script renames these (`enorm`, `hnorm`, `shared_head.norm`) before it
                    // decides which weights get `+ 1`, so all three do
                    let transform = if gguf.ends_with("norm.weight") {
                        Transform::PlusOne
                    } else {
                        Transform::Copy
                    };
                    add(
                        hf.to_string(),
                        format!("blk.{n}.{gguf}"),
                        shape,
                        Dtype::Bf16,
                        transform,
                        module,
                    );
                };
                mtp(
                    "mtp.fc.weight",
                    "nextn.eh_proj.weight",
                    vec![d, 2 * d],
                    None,
                );
                mtp(
                    "mtp.pre_fc_norm_embedding.weight",
                    "nextn.enorm.weight",
                    vec![d],
                    None,
                );
                mtp(
                    "mtp.pre_fc_norm_hidden.weight",
                    "nextn.hnorm.weight",
                    vec![d],
                    None,
                );
                mtp(
                    "mtp.norm.weight",
                    "nextn.shared_head_norm.weight",
                    vec![d],
                    None,
                );
            }
        }
        add(
            "model.language_model.norm.weight".into(),
            "output_norm.weight".into(),
            vec![d],
            Dtype::Bf16,
            Transform::PlusOne,
            None,
        );
        out
    }
}
