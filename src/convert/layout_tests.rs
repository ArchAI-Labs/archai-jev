//! The tensor layout of Qwen3.5-0.8B against the GGUF the official script wrote for it.

use std::path::PathBuf;

use super::layout::{Dims, Transform};
use crate::models::families::{self, ArchParams, MetaExpect, ParamValue, Role};
use crate::models::gguf::{self, GgmlType, MetaValue};

pub(crate) fn kev_dims() -> Dims {
    Dims {
        n_layers: 24,
        n_mtp: 1,
        hidden: 1024,
        ff: 3584,
        heads: 8,
        kv_heads: 2,
        head_dim: 256,
        vocab: 248_320,
        interval: 4,
        conv_kernel: 4,
        key_head_dim: 128,
        key_heads: 16,
        value_heads: 16,
        inner: 2048,
    }
}

pub(crate) fn kev_params() -> ArchParams {
    let i = ParamValue::Int;
    ArchParams::from_pairs(vec![
        ("n_layers".to_string(), i(24)),
        ("n_mtp_layers".to_string(), i(1)),
        ("n_embd".to_string(), i(1024)),
        ("n_ff".to_string(), i(3584)),
        ("n_heads".to_string(), i(8)),
        ("n_kv_heads".to_string(), i(2)),
        ("head_dim".to_string(), i(256)),
        ("n_vocab".to_string(), i(248_320)),
        ("tie_embeddings".to_string(), ParamValue::Bool(true)),
        ("full_attention_interval".to_string(), i(4)),
        ("ssm_conv_kernel".to_string(), i(4)),
        ("ssm_state_size".to_string(), i(128)),
        ("ssm_group_count".to_string(), i(16)),
        ("ssm_time_step_rank".to_string(), i(16)),
        ("ssm_inner_size".to_string(), i(2048)),
        ("rope_theta".to_string(), i(10_000_000)),
        ("rope_dim".to_string(), i(64)),
    ])
}

#[test]
fn kev_0_8b_has_335_tensors_195_matrices_and_140_vectors() {
    let entries = kev_dims().entries();
    assert_eq!(entries.len(), 335);
    let matrices = entries.iter().filter(|e| e.role == Role::Matrix).count();
    assert_eq!((matrices, entries.len() - matrices), (195, 140));
    let mut names: Vec<&str> = entries.iter().map(|e| e.gguf.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), 335, "GGUF names are unique");
    let mut hf: Vec<&str> = entries.iter().map(|e| e.hf.as_str()).collect();
    hf.sort_unstable();
    hf.dedup();
    assert_eq!(hf.len(), 335, "checkpoint names are unique");
    let count = |t: Transform| entries.iter().filter(|e| e.transform == t).count();
    assert_eq!(
        (
            count(Transform::NegExp),
            count(Transform::Squeeze),
            count(Transform::PlusOne)
        ),
        (18, 18, 68)
    );
    // the weights a LoRA adapts: the 186 modules of the adapter of Kev-0.8B
    let adapted = entries
        .iter()
        .filter(|e| e.hf.starts_with("model.language_model.layers.") && e.module.is_some())
        .count();
    assert_eq!(adapted, 186);
}

#[test]
fn names_shapes_and_roles_match_the_official_script() {
    let entries = kev_dims().entries();
    let get = |name: &str| {
        entries
            .iter()
            .find(|e| e.gguf == name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
    };
    for (name, hf, dims, role) in [
        (
            "token_embd.weight",
            "model.language_model.embed_tokens.weight",
            vec![1024, 248_320],
            Role::Matrix,
        ),
        (
            "output_norm.weight",
            "model.language_model.norm.weight",
            vec![1024],
            Role::Vector,
        ),
        (
            "blk.0.attn_norm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            vec![1024],
            Role::Vector,
        ),
        (
            "blk.0.attn_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![1024, 6144],
            Role::Matrix,
        ),
        (
            "blk.0.attn_gate.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![1024, 2048],
            Role::Matrix,
        ),
        (
            "blk.0.ssm_alpha.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1024, 16],
            Role::Matrix,
        ),
        (
            "blk.0.ssm_beta.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1024, 16],
            Role::Matrix,
        ),
        (
            "blk.0.ssm_conv1d.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 6144],
            Role::Vector,
        ),
        (
            "blk.0.ssm_dt.bias",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![16],
            Role::Vector,
        ),
        (
            "blk.0.ssm_a",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![16],
            Role::Vector,
        ),
        (
            "blk.0.ssm_norm.weight",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![128],
            Role::Vector,
        ),
        (
            "blk.0.ssm_out.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2048, 1024],
            Role::Matrix,
        ),
        (
            "blk.3.attn_q.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            vec![1024, 4096],
            Role::Matrix,
        ),
        (
            "blk.3.attn_k.weight",
            "model.language_model.layers.3.self_attn.k_proj.weight",
            vec![1024, 512],
            Role::Matrix,
        ),
        (
            "blk.3.attn_output.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
            vec![2048, 1024],
            Role::Matrix,
        ),
        (
            "blk.3.attn_q_norm.weight",
            "model.language_model.layers.3.self_attn.q_norm.weight",
            vec![256],
            Role::Vector,
        ),
        (
            "blk.3.ffn_down.weight",
            "model.language_model.layers.3.mlp.down_proj.weight",
            vec![3584, 1024],
            Role::Matrix,
        ),
        (
            "blk.24.attn_q.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            vec![1024, 4096],
            Role::Matrix,
        ),
        (
            "blk.24.nextn.eh_proj.weight",
            "mtp.fc.weight",
            vec![2048, 1024],
            Role::Matrix,
        ),
        (
            "blk.24.nextn.enorm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            vec![1024],
            Role::Vector,
        ),
        (
            "blk.24.nextn.hnorm.weight",
            "mtp.pre_fc_norm_hidden.weight",
            vec![1024],
            Role::Vector,
        ),
        (
            "blk.24.nextn.shared_head_norm.weight",
            "mtp.norm.weight",
            vec![1024],
            Role::Vector,
        ),
    ] {
        let e = get(name);
        assert_eq!(
            (e.hf.as_str(), &e.dims, e.role),
            (hf, &dims, role),
            "{name}"
        );
    }
    // layer 3 is a full attention layer and has no linear attention tensors; layer 0 the opposite
    assert!(
        entries
            .iter()
            .all(|e| e.gguf != "blk.3.ssm_a" && e.gguf != "blk.0.attn_q.weight")
    );
    // which checkpoint types are expected: the two F32 ones are A_log and the norm of the DeltaNet
    let f32_in_checkpoint: Vec<&str> = entries
        .iter()
        .filter(|e| e.hf_dtype == super::safetensors::Dtype::F32)
        .map(|e| e.hf.as_str())
        .collect();
    assert_eq!(f32_in_checkpoint.len(), 36);
    assert!(
        f32_in_checkpoint
            .iter()
            .all(|n| n.ends_with(".A_log") || n.ends_with("linear_attn.norm.weight"))
    );
    // `+1` goes to the norms, not to the norm of the DeltaNet
    assert_eq!(get("blk.0.ssm_norm.weight").transform, Transform::Copy);
    assert_eq!(get("blk.0.attn_norm.weight").transform, Transform::PlusOne);
    assert_eq!(
        get("blk.3.attn_k_norm.weight").transform,
        Transform::PlusOne
    );
    assert_eq!(
        get("blk.24.nextn.shared_head_norm.weight").transform,
        Transform::PlusOne
    );
    assert_eq!(get("blk.0.ssm_conv1d.weight").transform, Transform::Squeeze);
    assert_eq!(get("blk.0.ssm_a").transform, Transform::NegExp);
}

#[test]
fn the_family_row_expects_the_same_tensors() {
    let family = families::lookup("qwen35-pointer").expect("the family exists");
    assert_eq!(
        (family.arch, family.head_kind, family.template_id),
        ("qwen35", "pointer", "kev")
    );
    let specs = (family.tensors)(&kev_params());
    assert_eq!(specs.len(), 335);
    let from_layout = kev_dims().entries();
    for (spec, entry) in specs.iter().zip(&from_layout) {
        assert_eq!(
            (&spec.name, &spec.dims, spec.role),
            (&entry.gguf, &entry.dims, entry.role)
        );
    }
}

/// The GGUF files the official script wrote (spike S4), if `ARCHAI_JEV_TEST_KEV_DIR` has them.
fn oracle(file: &str) -> Option<PathBuf> {
    let dir = std::env::var("ARCHAI_JEV_TEST_KEV_DIR").ok()?;
    let p = PathBuf::from(dir).join(file);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("SKIPPED: {file} is not in ARCHAI_JEV_TEST_KEV_DIR");
        None
    }
}

#[test]
fn the_family_row_describes_the_gguf_of_the_official_script() {
    let family = families::lookup("qwen35-pointer").unwrap();
    for (file, matrix) in [
        ("kev-0.8b-f32.gguf", GgmlType::F32),
        ("kev-0.8b-bf16.gguf", GgmlType::BF16),
    ] {
        let Some(path) = oracle(file) else { continue };
        let info = gguf::read_header(&path).unwrap();
        let params = kev_params();
        assert_eq!(info.tensors.len(), 335, "{file}");
        for spec in (family.tensors)(&params) {
            let t = info
                .tensor(&spec.name)
                .unwrap_or_else(|| panic!("{file}: no {}", spec.name));
            assert_eq!(t.dims, spec.dims, "{file}: {}", spec.name);
            let want = if spec.role == Role::Vector {
                GgmlType::F32
            } else {
                matrix
            };
            assert_eq!(t.ty, want, "{file}: {}", spec.name);
        }
        for (key, expect) in (family.metadata)(&params) {
            let got = info.get(&key).unwrap_or_else(|| panic!("{file}: no {key}"));
            match (expect, got) {
                (MetaExpect::Str(s), MetaValue::Str(g)) => assert_eq!(&s, g, "{file}: {key}"),
                (MetaExpect::UInt(n), MetaValue::UInt(g)) => assert_eq!(n, *g, "{file}: {key}"),
                (e, g) => panic!("{file}: {key}: {e:?} against {g:?}"),
            }
        }
        assert_eq!(
            info.get(family.context_key),
            Some(&MetaValue::UInt(262_144)),
            "{file}"
        );
    }
}
