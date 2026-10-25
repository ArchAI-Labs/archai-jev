//! The `hf-lora` source of a variant (spec 018): its schema in the manifest, the verification
//! of the original files before any conversion starts, and the cache of the result.

use std::sync::Arc;

use super::Harness;
use crate::error::Error;
use crate::json_strict::Json;
use crate::models::incompat::Incompat;
use crate::models::manifest::{Manifest, ManifestOrigin, Source};
use crate::models::testing::builder::{Builder, COMMIT};
use crate::models::testing::jsonedit::{arr, n, obj, remove, s, set};
use crate::models::testing::{FakeConverter, Fixture};

fn fixture() -> (Fixture, Vec<u8>) {
    let good = Builder::tiny().build();
    let bytes = std::fs::read(good.gguf_path()).unwrap();
    (Builder::tiny().hf_lora().build(), bytes)
}

/// Why a manifest with this `hf-lora` defect is refused (no converter involved: it is stage 1).
fn schema_refusal(builder: Builder) -> Incompat {
    let f = builder.hf_lora().build();
    let h = Harness::new();
    match h.load(&h.local(f.path())) {
        Err(Error::IncompatibleModel(i)) => i,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

fn mentions(i: &Incompat, fragment: &str) {
    let text = i.to_string();
    assert!(text.contains(fragment), "{text:?} lacks {fragment:?}");
}

#[test]
fn the_valid_source_is_read_into_typed_fields() {
    let f = Builder::tiny().hf_lora().build();
    let m = Manifest::from_json(f.manifest.clone(), ManifestOrigin::Local).unwrap();
    let Source::HfLora(src) = &m.variants[0].source else {
        panic!("not an hf-lora source")
    };
    assert_eq!(src.base.repo, "test/tiny");
    assert_eq!(src.base.revision, COMMIT);
    assert_eq!(src.base.weights.path, "base/model.safetensors");
    assert_eq!(src.adapter.config.path, "adapter/adapter_config.json");
    assert_eq!(src.files().len(), 6);
}

#[test]
fn reject_a_malformed_hf_lora_schema() {
    // unknown key, missing blocks and fields
    let i = schema_refusal(Builder::tiny().edit_manifest(|j| {
        set(j, "variants.0.source.extra", n(1));
    }));
    assert!(matches!(i, Incompat::ManifestUnknownField { .. }), "{i}");
    mentions(&i, "extra");
    for (path, field) in [
        ("variants.0.source.adapter", "adapter"),
        ("variants.0.source.base", "base"),
        ("variants.0.source.base.config", "config"),
        (
            "variants.0.source.base.tokenizer_config",
            "tokenizer_config",
        ),
        ("variants.0.source.adapter.weights", "weights"),
        ("variants.0.source.base.revision", "revision"),
    ] {
        let i = schema_refusal(Builder::tiny().edit_manifest(move |j| remove(j, path)));
        assert!(
            matches!(i, Incompat::ManifestMissingField { .. }),
            "{path}: {i}"
        );
        mentions(&i, field);
    }
    // sharded or empty weights
    for count in [0usize, 2] {
        let i = schema_refusal(Builder::tiny().edit_manifest(move |j| {
            let entry = obj(vec![
                ("path", s("base/model.safetensors")),
                ("size", n(12)),
                ("sha256", s(&"0".repeat(64))),
            ]);
            set(j, "variants.0.source.base.weights", arr(vec![entry; count]));
        }));
        assert!(matches!(i, Incompat::ManifestBadValue { .. }), "{i}");
        mentions(&i, "exactly one");
    }
    // repository and revision
    let i = schema_refusal(Builder::tiny().set("variants.0.source.base.repo", s("not a repo")));
    mentions(&i, "repository id");
    let i = schema_refusal(Builder::tiny().set("variants.0.source.base.revision", s("main")));
    mentions(&i, "40-digit");
    // `origin` belongs to the registry: in a local manifest it is an unknown field
    let origin = obj(vec![("repo", s("test/tiny")), ("revision", s(COMMIT))]);
    let i = schema_refusal(Builder::tiny().edit_manifest(move |j| {
        set(j, "variants.0.source.base.config.origin", origin.clone());
    }));
    assert!(matches!(i, Incompat::ManifestUnknownField { .. }), "{i}");
    mentions(&i, "origin");
    // a file entry that is not valid
    let i =
        schema_refusal(Builder::tiny().set("variants.0.source.adapter.config.sha256", s("abc")));
    mentions(&i, "sha256");
    let i = schema_refusal(Builder::tiny().set(
        "variants.0.source.base.weights.0.path",
        s("../outside.safetensors"),
    ));
    mentions(&i, "..");
}

fn registry_json(edit: impl Fn(&mut Json)) -> Result<Manifest, Incompat> {
    let f = Builder::tiny().registry().hf_lora().build();
    let mut j = f.manifest.clone();
    edit(&mut j);
    Manifest::from_json(j, ManifestOrigin::Registry)
}

#[test]
fn in_the_registry_the_base_files_come_from_the_base_repository() {
    assert!(registry_json(|_| {}).is_ok());
    let other = "b".repeat(40);
    // each of the four base files must agree with base.repo and base.revision
    for file in ["config", "tokenizer", "tokenizer_config", "weights.0"] {
        let path = format!("variants.0.source.base.{file}.origin.revision");
        let other = other.clone();
        let err = registry_json(move |j| set(j, &path, s(&other))).unwrap_err();
        mentions(&err, "base.repo and base.revision");
        mentions(&err, &file.replace(".0", "[0]"));
        let path = format!("variants.0.source.base.{file}.origin.repo");
        let err = registry_json(move |j| set(j, &path, s("other/repo"))).unwrap_err();
        mentions(&err, "base.repo and base.revision");
    }
    // the adapter files may live in another repository (they usually do)
    assert!(
        registry_json(|j| set(
            j,
            "variants.0.source.adapter.weights.origin.repo",
            s("someone/adapters")
        ))
        .is_ok()
    );
}

/// Overwrite one source file with different bytes of the same length (an integrity defect).
fn corrupt(relative: &'static str) -> impl Fn(&std::path::Path) {
    move |dir| {
        let p = dir.join(relative);
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] ^= 0x55;
        std::fs::write(&p, bytes).unwrap();
    }
}

#[test]
fn each_original_file_is_verified_before_the_conversion_starts() {
    for file in [
        "base/config.json",
        "base/model.safetensors",
        "base/tokenizer.json",
        "base/tokenizer_config.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    ] {
        let good = Builder::tiny().build();
        let bytes = std::fs::read(good.gguf_path()).unwrap();
        let f = Builder::tiny().hf_lora().after_seal(corrupt(file)).build();
        let conv = Arc::new(FakeConverter::new(bytes));
        let mut h = Harness::new();
        h.converter = Some(conv.clone());
        let err = h.load(&h.local(f.path())).err().unwrap();
        let Error::IncompatibleModel(i @ Incompat::HashMismatch { .. }) = err else {
            panic!("{file}: {err:?}")
        };
        mentions(&i, file.rsplit('/').next().unwrap());
        mentions(&i, "delete it and retry");
        assert_eq!(conv.calls(), 0, "{file}: the conversion must not start");
        assert!(
            !h.cache.path().join("v1").join("materialized").exists(),
            "{file}: nothing may be materialised"
        );
        assert_eq!(h.backend.loads(), 0);
    }
}

#[test]
fn a_missing_original_in_a_local_folder_is_named() {
    let (f, bytes) = fixture();
    std::fs::remove_file(f.path().join("adapter/adapter_model.safetensors")).unwrap();
    let conv = Arc::new(FakeConverter::new(bytes));
    let mut h = Harness::new();
    h.converter = Some(conv.clone());
    let err = h.load(&h.local(f.path())).err().unwrap();
    let Error::IncompatibleModel(i @ Incompat::FileMissing { .. }) = err else {
        panic!("{err:?}")
    };
    mentions(&i, "adapter_model.safetensors");
    assert_eq!(conv.calls(), 0);
}

#[test]
fn without_a_converter_the_source_is_refused_before_any_file_is_read() {
    let (f, _) = fixture();
    let h = Harness::new();
    let err = h.load(&h.local(f.path())).err().unwrap();
    assert!(matches!(
        err,
        Error::IncompatibleModel(Incompat::SourceWithoutConverter { .. })
    ));
}
