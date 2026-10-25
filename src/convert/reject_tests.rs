//! Every way the original files of a checkpoint can be wrong, one at a time (spec 018, section
//! D of the acceptance criteria). Each test starts from the valid tiny checkpoint, introduces one
//! defect and checks the error, that it names the culprit, and that nothing is left behind.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::oracle_tests::{BASE_REPO, fixture, params};
use super::qwen35::convert_files;
use super::safetensors::Dtype;
use super::testing::{
    StEntry, load_entries, safetensors_header, safetensors_with_header, save_entries,
};
use crate::error::Error;
use crate::hub::events::{CancelFlag, NeverCancel, NoObserver};
use crate::models::convert::SourceFiles;
use crate::models::failures::DownloadFailure;
use crate::models::families::{ArchParams, ParamValue};
use crate::models::incompat::Incompat;

/// A private copy of the fixture's input files.
struct Case {
    dir: tempfile::TempDir,
}

impl Case {
    fn new() -> Case {
        let dir = tempfile::tempdir().unwrap();
        for sub in ["base", "adapter"] {
            std::fs::create_dir_all(dir.path().join(sub)).unwrap();
            for entry in std::fs::read_dir(fixture().join(sub)).unwrap() {
                let entry = entry.unwrap();
                std::fs::copy(entry.path(), dir.path().join(sub).join(entry.file_name())).unwrap();
            }
        }
        Case { dir }
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.dir.path().join(relative)
    }

    fn files(&self) -> SourceFiles {
        SourceFiles {
            base_config: self.path("base/config.json"),
            base_weights: self.path("base/model.safetensors"),
            base_tokenizer: self.path("base/tokenizer.json"),
            base_tokenizer_config: self.path("base/tokenizer_config.json"),
            adapter_config: self.path("adapter/adapter_config.json"),
            adapter_weights: self.path("adapter/adapter_model.safetensors"),
        }
    }

    fn edit_json(&self, relative: &str, edit: impl Fn(&mut Value)) {
        let p = self.path(relative);
        let mut v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        edit(&mut v);
        std::fs::write(p, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    }

    fn edit_entries(&self, relative: &str, edit: impl Fn(&mut Vec<StEntry>)) {
        let p = self.path(relative);
        let mut entries = load_entries(&p);
        edit(&mut entries);
        save_entries(&p, &entries);
    }

    fn run_with(&self, params: &ArchParams, dtype: &str) -> (tempfile::TempDir, Result<(), Error>) {
        let out = tempfile::tempdir().unwrap();
        let r = convert_files(
            &self.files(),
            params,
            BASE_REPO,
            dtype,
            out.path(),
            &NoObserver,
            &NeverCancel,
        )
        .map(|_| ());
        (out, r)
    }
}

/// The reason the conversion of `case` is refused; also checks that nothing is left in the output.
fn refusal(case: &Case) -> Incompat {
    let (out, r) = case.run_with(&params(), "bf16");
    let leftovers: Vec<_> = std::fs::read_dir(out.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(
        leftovers.is_empty(),
        "a refused conversion left {leftovers:?}"
    );
    match r {
        Err(Error::IncompatibleModel(i)) => i,
        other => panic!("expected an IncompatibleModelError, got {other:?}"),
    }
}

fn mentions(i: &Incompat, fragment: &str) {
    let text = i.to_string();
    assert!(text.contains(fragment), "{text:?} lacks {fragment:?}");
}

#[test]
fn the_unmodified_copy_converts() {
    let case = Case::new();
    let (out, r) = case.run_with(&params(), "bf16");
    r.unwrap();
    assert!(out.path().join("model.gguf").is_file());
    assert!(!out.path().join("model.gguf.part").exists());
}

// ---- manifest and config.json ----

#[test]
fn a_parameter_of_the_manifest_that_differs_from_config_json_is_refused() {
    let case = Case::new();
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
        let wrong = {
            let mut pairs: Vec<(String, ParamValue)> = Vec::new();
            for (k, v) in [
                ("n_layers", 4u64),
                ("n_mtp_layers", 1),
                ("n_embd", 16),
                ("n_ff", 32),
                ("n_heads", 2),
                ("n_kv_heads", 1),
                ("head_dim", 8),
                ("n_vocab", 40),
                ("full_attention_interval", 4),
                ("ssm_conv_kernel", 4),
                ("ssm_state_size", 4),
                ("ssm_group_count", 2),
                ("ssm_time_step_rank", 2),
                ("ssm_inner_size", 8),
                ("rope_theta", 10_000_000),
                ("rope_dim", 2),
            ] {
                pairs.push((
                    k.to_string(),
                    ParamValue::Int(if k == name { v + 1 } else { v }),
                ));
            }
            pairs.push(("tie_embeddings".to_string(), ParamValue::Bool(true)));
            ArchParams::from_pairs(pairs)
        };
        let (_out, r) = case.run_with(&wrong, "bf16");
        let Err(Error::IncompatibleModel(i)) = r else {
            panic!("{name}: accepted")
        };
        assert!(
            matches!(i, Incompat::ArchParamMismatch { ref param, .. } if param == name),
            "{name}: {i}"
        );
    }
}

#[test]
fn a_config_that_needs_a_reordering_of_the_heads_is_refused_before_any_weight_is_read() {
    let case = Case::new();
    case.edit_json("base/config.json", |j| {
        j["text_config"]["linear_num_value_heads"] = json!(4)
    });
    // the weights are not even opened: remove them
    std::fs::remove_file(case.path("base/model.safetensors")).unwrap();
    let i = refusal(&case);
    assert!(matches!(i, Incompat::SourceUnsupported { .. }), "{i}");
    mentions(
        &i,
        "linear_num_key_heads is 2 and linear_num_value_heads is 4",
    );
}

#[test]
fn a_dtype_the_converter_does_not_write_is_refused() {
    let case = Case::new();
    let (_out, r) = case.run_with(&params(), "q8_0");
    let Err(Error::IncompatibleModel(i)) = r else {
        panic!("accepted")
    };
    assert!(matches!(i, Incompat::SourceUnsupported { .. }), "{i}");
    mentions(&i, "q8_0");
}

// ---- safetensors files ----

#[test]
fn damaged_weight_files_are_refused_naming_which_one() {
    for (file, label) in [
        ("base/model.safetensors", "base weights"),
        ("adapter/adapter_model.safetensors", "adapter weights"),
    ] {
        let case = Case::new();
        let p = case.path(file);
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.pop();
        std::fs::write(&p, bytes).unwrap();
        let i = refusal(&case);
        assert!(matches!(i, Incompat::SourceFile { .. }), "{file}: {i}");
        mentions(&i, label);

        // another dtype than BF16 and F32
        let case = Case::new();
        let entries = load_entries(&case.path(file));
        let header = safetensors_header(&entries)
            .replacen("\"BF16\"", "\"F16\"", 1)
            .replacen("\"F32\"", "\"F16\"", 1);
        std::fs::write(case.path(file), safetensors_with_header(&header, &entries)).unwrap();
        let i = refusal(&case);
        mentions(&i, label);
        mentions(&i, "F16");
    }
}

// ---- tensors of the base model ----

#[test]
fn a_missing_tensor_of_the_base_is_named() {
    for name in [
        "model.language_model.layers.0.mlp.up_proj.weight",
        "mtp.fc.weight",
        "model.language_model.embed_tokens.weight",
    ] {
        let case = Case::new();
        case.edit_entries("base/model.safetensors", |e| e.retain(|t| t.name != name));
        let i = refusal(&case);
        mentions(&i, name);
        mentions(&i, "missing");
    }
}

#[test]
fn an_unexpected_tensor_of_the_base_is_named() {
    for name in [
        "model.language_model.layers.4.mlp.up_proj.weight", // one layer too many
        "model.visualx.patch.weight",                       // not the vision tower's prefix
        "lm_head.weight",
        "extra",
    ] {
        let case = Case::new();
        case.edit_entries("base/model.safetensors", |e| {
            e.push(StEntry::bf16(name, &[2], &[1.0, 2.0]))
        });
        let i = refusal(&case);
        mentions(&i, name);
        mentions(&i, "not expected");
    }
    // the vision tower is dropped, not an error
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        e.retain(|t| !t.name.starts_with("model.visual."))
    });
    let (_out, r) = case.run_with(&params(), "bf16");
    r.unwrap();
}

#[test]
fn a_base_tensor_of_another_shape_or_dtype_is_refused() {
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name == "model.language_model.embed_tokens.weight")
        {
            *t = StEntry::bf16(&t.name, &[41, 16], &vec![0.0; 41 * 16]);
        }
    });
    let i = refusal(&case);
    mentions(&i, "embed_tokens");
    mentions(&i, "[41, 16]");
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name == "model.language_model.layers.0.mlp.up_proj.weight")
        {
            *t = StEntry::bf16(&t.name, &[16, 32], &vec![0.0; 16 * 32]);
        }
    });
    mentions(&refusal(&case), "[16, 32]");
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name == "model.language_model.norm.weight")
        {
            *t = StEntry::bf16(&t.name, &[15], &[0.0; 15]);
        }
    });
    mentions(&refusal(&case), "[15]");
    // dtype: A_log must be F32, an embedding must be BF16
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name == "model.language_model.layers.0.linear_attn.A_log")
        {
            *t = StEntry::bf16(&t.name, &[2], &[-1.0, -2.0]);
        }
    });
    let i = refusal(&case);
    mentions(&i, "A_log");
    mentions(&i, "dtype BF16, expected F32");
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name == "model.language_model.embed_tokens.weight")
        {
            *t = StEntry::f32(&t.name, &[40, 16], &vec![0.0; 40 * 16]);
        }
    });
    mentions(&refusal(&case), "dtype F32, expected BF16");
}

#[test]
fn a_non_finite_value_is_refused_wherever_it_is() {
    // in a weight, in a norm that gets `+ 1`, in the adapter, and one that appears in the merge
    let nan_bf16 = 0x7FC0u16.to_le_bytes();
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name == "model.language_model.layers.1.mlp.down_proj.weight")
        {
            t.data[6..8].copy_from_slice(&nan_bf16);
        }
    });
    let i = refusal(&case);
    mentions(&i, "layers.1.mlp.down_proj");
    mentions(&i, "non-finite");
    let case = Case::new();
    case.edit_entries("base/model.safetensors", |e| {
        for t in e.iter_mut().filter(|t| t.name == "mtp.norm.weight") {
            t.data[0..2].copy_from_slice(&0x7F80u16.to_le_bytes()); // +inf
        }
    });
    mentions(&refusal(&case), "mtp.norm.weight");
    let case = Case::new();
    case.edit_entries("adapter/adapter_model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name.ends_with("layers.2.mlp.up_proj.lora_A.weight"))
        {
            t.data[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
    });
    let i = refusal(&case);
    mentions(&i, "lora_A");
    mentions(&i, "non-finite");
    // finite inputs whose product overflows `f32`
    let case = Case::new();
    case.edit_entries("adapter/adapter_model.safetensors", |e| {
        for t in e.iter_mut() {
            if t.name.ends_with("layers.0.mlp.gate_proj.lora_A.weight")
                || t.name.ends_with("layers.0.mlp.gate_proj.lora_B.weight")
            {
                for c in t.data.as_chunks_mut::<4>().0 {
                    *c = 1.0e30f32.to_le_bytes();
                }
            }
        }
    });
    let i = refusal(&case);
    mentions(&i, "layers.0.mlp.gate_proj");
    mentions(&i, "merged value");
}

// ---- the adapter ----

#[test]
fn an_unsupported_adapter_option_stops_the_conversion() {
    let case = Case::new();
    case.edit_json("adapter/adapter_config.json", |j| {
        j["use_dora"] = json!(true)
    });
    let i = refusal(&case);
    assert!(matches!(i, Incompat::SourceUnsupported { .. }), "{i}");
    mentions(&i, "use_dora");
    let case = Case::new();
    case.edit_json("adapter/adapter_config.json", |j| {
        j["base_model_name_or_path"] = json!("Other/Base")
    });
    mentions(&refusal(&case), "Other/Base");
}

#[test]
fn adapter_tensors_must_match_the_weights_they_adapt() {
    // a module of target_modules without its pair
    let case = Case::new();
    case.edit_entries("adapter/adapter_model.safetensors", |e| {
        e.retain(|t| !t.name.ends_with("layers.1.mlp.up_proj.lora_B.weight"));
    });
    let i = refusal(&case);
    mentions(&i, "layers.1.mlp.up_proj.lora_B");
    mentions(&i, "missing");
    // a tensor that is not a pair half, and a pair for a weight nothing adapts
    for name in [
        "base_model.model.layers.1.mlp.up_proj.lora_magnitude_vector",
        "base_model.model.embed_tokens.lora_A.weight",
        "base_model.model.layers.1.input_layernorm.lora_A.weight",
        "base_model.model.layers.9.mlp.up_proj.lora_A.weight",
    ] {
        let case = Case::new();
        case.edit_entries("adapter/adapter_model.safetensors", |e| {
            e.push(StEntry::f32(name, &[2, 16], &[0.0; 32]))
        });
        let i = refusal(&case);
        mentions(&i, name);
        mentions(&i, "does not belong");
    }
    // shapes: A must be [r, in], B must be [out, r]; r must be the one of the config
    let case = Case::new();
    case.edit_entries("adapter/adapter_model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name.ends_with("layers.0.mlp.up_proj.lora_A.weight"))
        {
            *t = StEntry::f32(&t.name, &[3, 16], &[0.0; 48]);
        }
    });
    let i = refusal(&case);
    mentions(&i, "lora_A");
    mentions(&i, "expected [2, 16]");
    let case = Case::new();
    case.edit_entries("adapter/adapter_model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name.ends_with("layers.0.mlp.up_proj.lora_B.weight"))
        {
            *t = StEntry::f32(&t.name, &[32, 3], &[0.0; 96]);
        }
    });
    mentions(&refusal(&case), "expected [32, 2]");
    // dtype
    let case = Case::new();
    case.edit_entries("adapter/adapter_model.safetensors", |e| {
        for t in e
            .iter_mut()
            .filter(|t| t.name.ends_with("layers.0.mlp.up_proj.lora_A.weight"))
        {
            *t = StEntry::bf16(&t.name, &[2, 16], &[0.0; 32]);
        }
    });
    let i = refusal(&case);
    mentions(&i, "dtype BF16, expected F32");
    assert!(Dtype::F32.size() == 4);
}

// ---- the vocabulary ----

#[test]
fn a_broken_tokenizer_is_refused() {
    let case = Case::new();
    case.edit_json("base/tokenizer.json", |j| {
        j["model"]["vocab"]["zz"] = json!(0); // a second token with id 0
    });
    mentions(&refusal(&case), "two tokens have the id 0");
    let case = Case::new();
    case.edit_json("base/tokenizer_config.json", |j| {
        j.as_object_mut().unwrap().remove("eos_token");
    });
    mentions(&refusal(&case), "eos_token");
    let case = Case::new();
    std::fs::write(case.path("base/tokenizer.json"), b"not json").unwrap();
    mentions(&refusal(&case), "not valid JSON");
}

// ---- interruption and I/O ----

#[test]
fn a_request_to_stop_ends_the_conversion_and_leaves_nothing() {
    let case = Case::new();
    let out = tempfile::tempdir().unwrap();
    let cancel = CancelFlag::default();
    cancel.cancel();
    let r = convert_files(
        &case.files(),
        &params(),
        BASE_REPO,
        "bf16",
        out.path(),
        &NoObserver,
        &cancel,
    );
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert_eq!(
        std::fs::read_dir(out.path()).unwrap().count(),
        0,
        "no partial file"
    );
}

#[test]
fn an_output_folder_that_cannot_be_written_is_an_io_error_not_a_panic() {
    let case = Case::new();
    let r = convert_files(
        &case.files(),
        &params(),
        BASE_REPO,
        "bf16",
        Path::new("/nonexistent-folder/for/the/output"),
        &NoObserver,
        &NeverCancel,
    );
    assert!(
        matches!(r, Err(Error::ModelDownload(DownloadFailure::Io { .. }))),
        "{r:?}"
    );
    // an input that vanished after it was checked
    let case = Case::new();
    std::fs::remove_file(case.path("base/model.safetensors")).unwrap();
    let (_out, r) = case.run_with(&params(), "bf16");
    assert!(
        matches!(r, Err(Error::ModelDownload(DownloadFailure::Io { .. }))),
        "{r:?}"
    );
}
