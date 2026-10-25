//! Tests of `adapter_config.json`: the real file of Kev-0.8B and one unsupported option at a time.

use super::lora::{Half, LoraConfig, base_of};
use crate::error::Error;
use crate::json_strict::{self, Json};
use crate::models::incompat::Incompat;
use crate::models::testing::jsonedit::{arr, b, n, obj, remove, s, set};

const REAL: &str = include_str!("../../tests/data/kev-0.8b/adapter_config.json");
const BASE: &str = "Qwen/Qwen3.5-0.8B-Base";

fn real() -> Json {
    json_strict::parse(REAL.as_bytes()).unwrap()
}

fn edited(edit: impl Fn(&mut Json)) -> Json {
    let mut j = real();
    edit(&mut j);
    j
}

fn refusal(j: &Json) -> Incompat {
    match LoraConfig::from_json(j, BASE) {
        Err(Error::IncompatibleModel(i)) => i,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

fn mentions(i: &Incompat, fragment: &str) {
    let text = i.to_string();
    assert!(text.contains(fragment), "{text:?} lacks {fragment:?}");
}

#[test]
fn the_real_adapter_config_is_read() {
    let c = LoraConfig::from_json(&real(), BASE).unwrap();
    assert_eq!((c.r, c.alpha, c.scale()), (16, 32.0, 2.0));
    assert_eq!(c.target_modules.len(), 12);
    assert!(c.targets("q_proj") && c.targets("in_proj_qkv") && c.targets("out_proj"));
    assert!(!c.targets("embed_tokens") && !c.targets("lm_head"));
}

#[test]
fn each_unsupported_option_is_refused_by_name() {
    for (key, value) in [
        ("use_rslora", b(true)),
        ("use_dora", b(true)),
        ("use_qalora", b(true)),
        ("use_bdlora", b(true)),
        ("fan_in_fan_out", b(true)),
        ("lora_bias", b(true)),
        ("ensure_weight_tying", b(true)),
        ("trainable_token_indices", arr(vec![n(1), n(2)])),
        ("modules_to_save", arr(vec![s("lm_head")])),
        ("rank_pattern", obj(vec![("q_proj", n(8))])),
        ("alpha_pattern", obj(vec![("q_proj", n(8))])),
        ("loftq_config", obj(vec![("bits", n(4))])),
        ("target_parameters", arr(vec![s("x")])),
        ("layer_replication", arr(vec![arr(vec![n(0), n(2)])])),
        ("exclude_modules", arr(vec![s("k_proj")])),
        ("layers_to_transform", arr(vec![n(0)])),
        ("layers_pattern", s("layers")),
        ("eva_config", obj(vec![])),
        ("auto_mapping", obj(vec![])),
    ] {
        let name = key.to_string();
        let i = refusal(&edited(move |j| set(j, &name, value.clone())));
        assert!(
            matches!(i, Incompat::SourceUnsupported { .. }),
            "{key}: {i}"
        );
        mentions(&i, key);
    }
    let i = refusal(&edited(|j| set(j, "bias", s("all"))));
    assert!(matches!(i, Incompat::SourceUnsupported { .. }), "{i}");
    mentions(&i, "bias");
    // a key nobody knows might change the weights: refused, not ignored
    let i = refusal(&edited(|j| set(j, "use_something_new", b(true))));
    assert!(matches!(i, Incompat::SourceUnsupported { .. }), "{i}");
    mentions(&i, "use_something_new");
    // options that do not change the merged weights are accepted
    let ok = edited(|j| {
        set(j, "lora_dropout", Json::Float(0.3));
        set(j, "init_lora_weights", b(false));
        set(j, "inference_mode", b(false));
    });
    assert!(LoraConfig::from_json(&ok, BASE).is_ok());
}

#[test]
fn rank_alpha_modules_and_base_must_be_valid() {
    for (case, edit) in [
        (
            "peft_type",
            Box::new(|j: &mut Json| set(j, "peft_type", s("IA3"))) as Box<dyn Fn(&mut Json)>,
        ),
        ("r zero", Box::new(|j: &mut Json| set(j, "r", n(0)))),
        ("r huge", Box::new(|j: &mut Json| set(j, "r", n(100_000)))),
        ("r missing", Box::new(|j: &mut Json| remove(j, "r"))),
        (
            "alpha zero",
            Box::new(|j: &mut Json| set(j, "lora_alpha", n(0))),
        ),
        (
            "alpha text",
            Box::new(|j: &mut Json| set(j, "lora_alpha", s("32"))),
        ),
        (
            "alpha missing",
            Box::new(|j: &mut Json| remove(j, "lora_alpha")),
        ),
        (
            "modules empty",
            Box::new(|j: &mut Json| set(j, "target_modules", arr(vec![]))),
        ),
        (
            "modules pattern",
            Box::new(|j: &mut Json| set(j, "target_modules", s("all-linear"))),
        ),
        (
            "modules mixed",
            Box::new(|j: &mut Json| set(j, "target_modules", arr(vec![n(1)]))),
        ),
        ("bias missing", Box::new(|j: &mut Json| remove(j, "bias"))),
        (
            "base other",
            Box::new(|j: &mut Json| set(j, "base_model_name_or_path", s("Other/Base"))),
        ),
        (
            "base missing",
            Box::new(|j: &mut Json| remove(j, "base_model_name_or_path")),
        ),
    ] {
        let j = edited(edit);
        let i = refusal(&j);
        assert!(
            matches!(
                i,
                Incompat::SourceFile { .. } | Incompat::SourceUnsupported { .. }
            ),
            "{case}: {i}"
        );
    }
    let i = refusal(&Json::Null);
    mentions(&i, "not a JSON object");
}

#[test]
fn adapter_tensor_names_map_to_base_weights() {
    assert_eq!(
        base_of("base_model.model.layers.3.mlp.down_proj.lora_A.weight"),
        Some((
            "model.language_model.layers.3.mlp.down_proj.weight".to_string(),
            Half::A
        ))
    );
    assert_eq!(
        base_of("base_model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"),
        Some((
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            Half::B
        ))
    );
    for other in [
        "layers.3.mlp.down_proj.lora_A.weight",
        "base_model.model.layers.3.mlp.down_proj.weight",
        "base_model.model.layers.3.mlp.down_proj.lora_embedding_A",
        "base_model.model.lm_head.lora_magnitude_vector",
    ] {
        assert_eq!(base_of(other), None, "{other}");
    }
}
