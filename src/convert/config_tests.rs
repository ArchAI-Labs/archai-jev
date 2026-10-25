//! Tests of `config.json` of the base model: the real file, and one defect at a time.

use super::config::Hparams;
use crate::error::Error;
use crate::json_strict::{self, Json};
use crate::models::families::{ArchParams, ParamValue};
use crate::models::incompat::Incompat;
use crate::models::testing::jsonedit::{arr, b, n, remove, s, set};

const REAL: &str = include_str!("../../tests/data/kev-0.8b/base-config.json");

fn real() -> Json {
    json_strict::parse(REAL.as_bytes()).unwrap()
}

fn edited(edit: impl Fn(&mut Json)) -> Json {
    let mut j = real();
    edit(&mut j);
    j
}

fn refusal(j: &Json) -> Incompat {
    match Hparams::from_json(j) {
        Err(Error::IncompatibleModel(i)) => i,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

fn mentions(i: &Incompat, fragment: &str) {
    let text = i.to_string();
    assert!(text.contains(fragment), "{text:?} lacks {fragment:?}");
}

#[test]
fn the_real_config_of_kev_0_8b_is_read() {
    let h = Hparams::from_json(&real()).unwrap();
    assert_eq!((h.n_layers, h.n_mtp, h.hidden, h.ff), (24, 1, 1024, 3584));
    assert_eq!(
        (h.heads, h.kv_heads, h.head_dim, h.vocab),
        (8, 2, 256, 248_320)
    );
    assert_eq!(h.max_position, 262_144);
    assert_eq!(h.full_attention_interval, 4);
    assert_eq!(h.linear_layers.len(), 24);
    assert_eq!(h.linear_layers.iter().filter(|l| **l).count(), 18);
    assert!(!h.linear_layers[3] && !h.linear_layers[23] && h.linear_layers[22]);
    assert_eq!(
        (
            h.conv_kernel,
            h.key_head_dim,
            h.key_heads,
            h.value_head_dim,
            h.value_heads
        ),
        (4, 128, 16, 128, 16)
    );
    assert_eq!(
        (h.rope_theta, h.partial_rotary, h.rope_dim()),
        (1e7, 0.25, 64)
    );
    assert_eq!(h.mrope_section, [11, 11, 10]);
    assert_eq!(h.rms_eps, 1e-6);
}

#[test]
fn a_config_that_is_not_what_it_should_be_is_refused() {
    let i = refusal(&Json::Null);
    mentions(&i, "not a JSON object");
    let i = refusal(&edited(|j| {
        set(j, "architectures", arr(vec![s("LlamaForCausalLM")]))
    }));
    mentions(&i, "architectures");
    let i = refusal(&edited(|j| set(j, "model_type", s("llama"))));
    mentions(&i, "model_type");
    let i = refusal(&edited(|j| remove(j, "text_config")));
    mentions(&i, "text_config");
    let i = refusal(&edited(|j| set(j, "text_config.model_type", s("qwen3"))));
    mentions(&i, "qwen3_5_text");
}

#[test]
fn every_needed_field_is_required_and_has_a_type() {
    for key in [
        "num_hidden_layers",
        "full_attention_interval",
        "layer_types",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "attn_output_gate",
        "tie_word_embeddings",
        "rope_parameters",
        "rms_norm_eps",
        "head_dim",
        "mtp_num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
    ] {
        let path = format!("text_config.{key}");
        let p = path.clone();
        let i = refusal(&edited(move |j| remove(j, &p)));
        mentions(&i, key);
        let p = path.clone();
        let i = refusal(&edited(move |j| set(j, &p, s("x"))));
        mentions(&i, key);
    }
    for key in [
        "rope_type",
        "mrope_interleaved",
        "mrope_section",
        "partial_rotary_factor",
        "rope_theta",
    ] {
        let path = format!("text_config.rope_parameters.{key}");
        let i = refusal(&edited(move |j| remove(j, &path)));
        mentions(&i, key);
    }
}

#[test]
fn layer_types_must_agree_with_the_layer_count_and_the_interval() {
    let i = refusal(&edited(|j| set(j, "text_config.num_hidden_layers", n(23))));
    mentions(&i, "layer_types has 24 entries");
    let i = refusal(&edited(|j| {
        set(j, "text_config.layer_types.3", s("linear_attention"))
    }));
    mentions(&i, "layer_types[3]");
    let i = refusal(&edited(|j| {
        set(j, "text_config.layer_types.0", s("full_attention"))
    }));
    mentions(&i, "layer_types[0]");
    let i = refusal(&edited(|j| {
        set(j, "text_config.layer_types.5", s("sliding_attention"))
    }));
    mentions(&i, "layer_types[5]");
}

#[test]
fn a_number_that_does_not_fit_a_gguf_integer_is_never_written_saturated() {
    use super::qwen35::metadata;
    use super::vocab::Vocab;
    let vocab = Vocab {
        tokens: vec!["a".to_string()],
        types: vec![1],
        merges: vec![],
        eos: 0,
        pad: 0,
        add_bos: false,
        add_eos: false,
    };
    let fine = Hparams::from_json(&real()).unwrap();
    assert!(metadata(&fine, &vocab, "bf16").is_ok());
    let big = Hparams::from_json(&edited(|j| {
        set(j, "text_config.max_position_embeddings", n(5_000_000_000))
    }))
    .unwrap();
    match metadata(&big, &vocab, "bf16") {
        Err(Error::IncompatibleModel(i)) => {
            mentions(&i, "5000000000");
            mentions(&i, "32 bits");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn what_the_converter_cannot_check_is_not_guessed() {
    // different numbers of key and value heads: the heads of V would have to be reordered
    let i = refusal(&edited(|j| {
        set(j, "text_config.linear_num_value_heads", n(32))
    }));
    assert!(matches!(i, Incompat::SourceUnsupported { .. }), "{i}");
    mentions(
        &i,
        "linear_num_key_heads is 16 and linear_num_value_heads is 32",
    );
    mentions(&i, "not supported yet");
    for (path, value) in [
        ("text_config.num_experts", n(8)),
        ("moe_intermediate_size", n(8)),
        ("quantization_config", Json::Object(vec![])),
        ("text_config.attn_output_gate", b(false)),
        ("text_config.tie_word_embeddings", b(false)),
        ("text_config.mtp_use_dedicated_embeddings", b(true)),
        ("text_config.mtp_num_hidden_layers", n(2)),
        ("text_config.mtp_num_hidden_layers", n(0)),
        ("text_config.mlp_only_layers", arr(vec![n(0)])),
        ("text_config.rope_parameters.rope_type", s("yarn")),
        ("text_config.rope_parameters.mrope_interleaved", b(false)),
    ] {
        let i = refusal(&edited(move |j| set(j, path, value.clone())));
        assert!(
            matches!(i, Incompat::SourceUnsupported { .. }),
            "{path}: {i}"
        );
    }
    let i = refusal(&edited(|j| {
        set(
            j,
            "text_config.rope_parameters.partial_rotary_factor",
            Json::Float(0.3),
        )
    }));
    mentions(&i, "whole number");
    let i = refusal(&edited(|j| {
        set(j, "text_config.rms_norm_eps", Json::Float(-1.0))
    }));
    mentions(&i, "rms_norm_eps");
}

fn params(edit: impl Fn(&mut Vec<(String, ParamValue)>)) -> ArchParams {
    let i = ParamValue::Int;
    let mut pairs = vec![
        ("n_layers".to_string(), i(24)),
        ("n_mtp_layers".to_string(), i(1)),
        ("n_embd".to_string(), i(1024)),
        ("n_ff".to_string(), i(3584)),
        ("n_heads".to_string(), i(8)),
        ("n_kv_heads".to_string(), i(2)),
        ("head_dim".to_string(), i(256)),
        ("n_vocab".to_string(), i(248_320)),
        ("full_attention_interval".to_string(), i(4)),
        ("ssm_conv_kernel".to_string(), i(4)),
        ("ssm_state_size".to_string(), i(128)),
        ("ssm_group_count".to_string(), i(16)),
        ("ssm_time_step_rank".to_string(), i(16)),
        ("ssm_inner_size".to_string(), i(2048)),
        ("rope_theta".to_string(), i(10_000_000)),
        ("rope_dim".to_string(), i(64)),
        ("tie_embeddings".to_string(), ParamValue::Bool(true)),
    ];
    edit(&mut pairs);
    ArchParams::from_pairs(pairs)
}

#[test]
fn the_manifest_must_describe_the_files() {
    let h = Hparams::from_json(&real()).unwrap();
    assert_eq!(h.check_against(&params(|_| {})), Ok(()));
    for name in [
        "n_layers",
        "n_mtp_layers",
        "n_embd",
        "n_ff",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "n_vocab",
        "full_attention_interval",
        "ssm_conv_kernel",
        "ssm_state_size",
        "ssm_group_count",
        "ssm_time_step_rank",
        "ssm_inner_size",
        "rope_theta",
        "rope_dim",
    ] {
        let wrong = params(|p| {
            for (k, v) in p.iter_mut() {
                if k == name {
                    *v = ParamValue::Int(7);
                }
            }
        });
        let Err(i) = h.check_against(&wrong) else {
            panic!("{name}: a different value was accepted")
        };
        assert!(
            matches!(i, Incompat::ArchParamMismatch { ref param, .. } if param == name),
            "{i}"
        );
        mentions(&i, "config.json");
        mentions(&i, "7");
    }
    let untied = params(|p| {
        for (k, v) in p.iter_mut() {
            if k == "tie_embeddings" {
                *v = ParamValue::Bool(false);
            }
        }
    });
    assert!(matches!(
        h.check_against(&untied),
        Err(Incompat::ArchParamMismatch { .. })
    ));
}
