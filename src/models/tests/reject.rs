//! One test per reason a checkpoint is refused (spec 005, table of "Rifiuti al caricamento").
//!
//! Each starts from a valid tiny checkpoint, introduces only that defect, and checks the
//! reason, the message, that the engine was never loaded, and that nothing was left behind.

use super::Harness;
use crate::error::Error;
use crate::json_strict::Json;
use crate::models::incompat::Incompat;
use crate::models::testing::builder::{Builder, MANIFEST_FILE};
use crate::models::testing::jsonedit::{arr, b, f, n, obj, s};

/// Load a fixture built by `builder` and return why it was refused.
fn rejects(builder: Builder) -> Incompat {
    let fixture = builder.build();
    let h = Harness::new();
    let err = h
        .load(&h.local(fixture.path()))
        .err()
        .expect("the checkpoint should be refused");
    assert_eq!(
        h.backend.loads(),
        0,
        "the engine must not be touched: {err}"
    );
    assert_eq!(
        h.records(),
        0,
        "no verification record may be written: {err}"
    );
    assert_eq!(
        h.transport.gets.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    match err {
        Error::IncompatibleModel(i) => i,
        other => panic!("expected IncompatibleModel, got {other:?}"),
    }
}

fn has(i: &Incompat, fragment: &str) {
    let text = i.to_string();
    assert!(
        text.contains(fragment),
        "{text:?} does not contain {fragment:?}"
    );
}

/// Rewrite the manifest file of a built fixture with arbitrary bytes.
fn rejects_manifest_bytes(bytes: &[u8]) -> Incompat {
    let fixture = Builder::tiny().build();
    std::fs::write(fixture.manifest_path(), bytes).unwrap();
    let h = Harness::new();
    match h.load(&h.local(fixture.path())) {
        Err(Error::IncompatibleModel(i)) => i,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

// ---- stage 1: reading the manifest -------------------------------------------------------

#[test]
fn reject_manifest_absent() {
    let fixture = Builder::tiny().build();
    std::fs::remove_file(fixture.manifest_path()).unwrap();
    let h = Harness::new();
    let err = h.load(&h.local(fixture.path())).err().unwrap();
    let Error::IncompatibleModel(i @ Incompat::ManifestAbsent { .. }) = err else {
        panic!("{err:?}")
    };
    has(&i, MANIFEST_FILE);
    // a manifest= that does not exist
    let fixture = Builder::tiny().build();
    let mut req = h.local(fixture.path());
    req.selection.manifest = Some(fixture.path().join("elsewhere.json"));
    let err = h.load(&req).err().unwrap();
    assert!(matches!(
        err,
        Error::IncompatibleModel(Incompat::ManifestAbsent { .. })
    ));
}

#[test]
fn reject_json_syntax_bom_nan() {
    for bytes in [
        &b"{"[..],
        b"not json",
        b"\xEF\xBB\xBF{}",
        b"{\"a\": NaN}",
        b"",
    ] {
        let i = rejects_manifest_bytes(bytes);
        assert!(
            matches!(
                i,
                Incompat::ManifestSyntax { .. } | Incompat::ManifestLimit { .. }
            ),
            "{i}"
        );
    }
    assert!(matches!(
        rejects_manifest_bytes(b"\xEF\xBB\xBF{}"),
        Incompat::ManifestLimit { .. }
    ));
}

#[test]
fn reject_size_and_depth() {
    let big = format!("\"{}\"", "x".repeat(8 * 1024 * 1024));
    assert!(matches!(
        rejects_manifest_bytes(big.as_bytes()),
        Incompat::ManifestLimit { .. }
    ));
    let deep = format!("{}{}", "[".repeat(17), "]".repeat(17));
    let i = rejects_manifest_bytes(deep.as_bytes());
    assert!(matches!(i, Incompat::ManifestLimit { .. }));
    has(&i, "deeper than 16");
}

#[test]
fn reject_duplicate_key() {
    let i = rejects_manifest_bytes(br#"{"schema_version": 1, "schema_version": 1}"#);
    assert!(matches!(i, Incompat::ManifestDuplicateKey { ref key } if key == "schema_version"));
    let i = rejects_manifest_bytes(br#"{"tokenizer": {"special_tokens": {"a": 1, "a": 2}}}"#);
    assert!(matches!(i, Incompat::ManifestDuplicateKey { ref key } if key == "a"));
}

#[test]
fn reject_unknown_field() {
    let i = rejects(Builder::tiny().set("variants.0.tolerance", f(0.9)));
    assert!(
        matches!(i, Incompat::ManifestUnknownField { ref path, ref field } if path == "variants[0]" && field == "tolerance"),
        "{i}"
    );
    // `origin` is for registry manifests only
    let i = rejects(Builder::tiny().set(
        "tokenizer.file.origin",
        obj(vec![("repo", s("a/b")), ("revision", s(&"a".repeat(40)))]),
    ));
    assert!(
        matches!(i, Incompat::ManifestUnknownField { ref field, .. } if field == "origin"),
        "{i}"
    );
    let i = rejects(Builder::tiny().set("surprise", n(1)));
    has(&i, "surprise");
}

#[test]
fn reject_missing_required_field() {
    // every required field of the schema, one case each
    const REQUIRED: &[&str] = &[
        "schema_version",
        "name",
        "revision",
        "family",
        "architecture",
        "architecture.name",
        "architecture.n_layers",
        "architecture.n_embd",
        "architecture.n_ff",
        "architecture.n_heads",
        "architecture.n_kv_heads",
        "architecture.n_vocab",
        "architecture.tie_embeddings",
        "head",
        "head.kind",
        "head.rule",
        "head.choice_targets",
        "head.yes_no_targets",
        "template",
        "template.id",
        "template.version",
        "tokenizer",
        "tokenizer.file",
        "tokenizer.file.path",
        "tokenizer.file.size",
        "tokenizer.file.sha256",
        "tokenizer.special_tokens",
        "max_context",
        "license",
        "license.spdx",
        "license.url",
        "tasks",
        "default_dtype",
        "variants",
        "variants.0.dtype",
        "variants.0.source",
        "variants.0.source.kind",
        "variants.0.source.file",
        "variants.0.source.file.path",
        "variants.0.source.file.size",
        "variants.0.source.file.sha256",
        "variants.0.calibration",
        "variants.0.calibration.declared",
        "variants.0.selfcheck",
        "variants.0.selfcheck.vectors",
    ];
    for path in REQUIRED {
        let b = Builder::tiny()
            .edit_manifest(move |j| crate::models::testing::jsonedit::remove(j, path));
        let i = rejects(b);
        let last = path.rsplit('.').next().unwrap();
        assert!(
            matches!(i, Incompat::ManifestMissingField { .. }),
            "removing {path}: {i}"
        );
        has(&i, last);
    }
}

#[test]
fn reject_value_out_of_domain() {
    let sha = |v: &str| Builder::tiny().set("tokenizer.file.sha256", s(v));
    for bad in ["abc", &"A".repeat(64), &"g".repeat(64)] {
        has(&rejects(sha(bad)), "sha256");
    }
    has(
        &rejects(Builder::tiny().set("tokenizer.file.size", n(0))),
        "size",
    );
    for path in ["../x", "/abs", "a\\b", "C:/x", "a//b", ""] {
        let i = rejects(Builder::tiny().edit_manifest({
            let path = path.to_string();
            move |j| crate::models::testing::jsonedit::set(j, "tokenizer.file.path", s(&path))
        }));
        assert!(
            matches!(i, Incompat::ManifestBadValue { .. }),
            "{path:?}: {i}"
        );
    }
    has(
        &rejects(Builder::tiny().set("max_context", n(0))),
        "max_context",
    );
    has(
        &rejects(
            Builder::tiny()
                .set("variants.0.dtype", s("Q8 0"))
                .set("default_dtype", s("Q8 0")),
        ),
        "dtype",
    );
    has(&rejects(Builder::tiny().set("name", s("bad name"))), "name");
    has(
        &rejects(Builder::tiny().set("revision", s("bad rev!"))),
        "revision",
    );
    has(
        &rejects(Builder::tiny().set("variants", arr(vec![]))),
        "variants",
    );
    has(
        &rejects(Builder::tiny().set("head.rule", s("full_softmax"))),
        "rule",
    );
}

#[test]
fn reject_schema_version() {
    for v in [n(2), n(0), s("1"), f(1.5)] {
        let i = rejects(Builder::tiny().set("schema_version", v));
        assert!(matches!(i, Incompat::SchemaVersion { .. }), "{i}");
        has(&i, "supports 1");
    }
}

#[test]
fn reject_local_name_collision() {
    use crate::models::registry::Registry;
    let registry_fixture = Builder::tiny().registry().build();
    let text = std::fs::read_to_string(registry_fixture.manifest_path()).unwrap();
    let index = format!(
        r#"{{"default": "test/tiny@{c}", "current": {{"test/tiny": "{c}"}}}}"#,
        c = crate::models::testing::builder::COMMIT
    );
    let reg = Registry::from_sources(&[text.as_str()], &index).unwrap();
    let h = Harness::with_registry(reg);
    let local = Builder::tiny().set("name", s("test/tiny")).build();
    let err = h.load(&h.local(local.path())).err().unwrap();
    let Error::IncompatibleModel(i @ Incompat::NameCollision { .. }) = err else {
        panic!("{err:?}")
    };
    has(&i, "test/tiny");
}

// ---- stage 2: what the manifest declares -------------------------------------------------

#[test]
fn reject_unknown_family() {
    let i = rejects(Builder::tiny().set("family", s("nope")));
    assert!(matches!(i, Incompat::UnknownFamily { ref got, .. } if got == "nope"));
    has(&i, "qwen2-letters");
}

#[test]
fn reject_architecture_mismatch() {
    let i = rejects(Builder::tiny().set("architecture.name", s("llama")));
    assert!(matches!(i, Incompat::ArchitectureMismatch { .. }), "{i}");
    has(&i, "qwen2");
    // an unknown or mistyped structural parameter
    let i = rejects(Builder::tiny().set("architecture.n_layers", s("one")));
    assert!(matches!(i, Incompat::ManifestBadValue { .. }), "{i}");
    let i = rejects(Builder::tiny().set("architecture.rope_scaling", n(2)));
    assert!(matches!(i, Incompat::ManifestUnknownField { .. }), "{i}");
}

fn pointer_head() -> Json {
    obj(vec![
        ("kind", s("pointer")),
        ("d_model", n(32)),
        ("proj_dim", n(8)),
        (
            "weights",
            obj(vec![
                ("path", s("head.pt")),
                ("size", n(10)),
                ("sha256", s(&"a".repeat(64))),
            ]),
        ),
    ])
}

#[test]
fn reject_head_mismatch() {
    let i = rejects(Builder::tiny().set("head", pointer_head()));
    assert!(matches!(i, Incompat::HeadMismatch { .. }), "{i}");
    has(&i, "letters");
    let i = rejects(Builder::tiny().set("head.kind", s("x")));
    assert!(matches!(i, Incompat::HeadMismatch { .. }), "{i}");
}

#[test]
fn reject_template_mismatch() {
    let i = rejects(Builder::tiny().set("template.id", s("kev")));
    assert!(matches!(i, Incompat::TemplateMismatch { .. }), "{i}");
    let i = rejects(Builder::tiny().set("template.version", n(99)));
    assert!(matches!(i, Incompat::TemplateMismatch { .. }), "{i}");
}

#[test]
fn reject_dtype_without_tolerance() {
    let i = rejects(Builder::tiny().dtype("q4_k_m"));
    assert!(
        matches!(i, Incompat::NoTolerance { ref dtype, .. } if dtype == "q4_k_m"),
        "{i}"
    );
    has(&i, "tolerances are in the library");
}

#[test]
fn reject_source_without_converter() {
    for kind in ["hf-lora", "hf-full"] {
        let i = rejects(Builder::tiny().set("variants.0.source", obj(vec![("kind", s(kind))])));
        assert!(matches!(i, Incompat::SourceWithoutConverter { .. }), "{i}");
        has(&i, kind);
    }
}

#[test]
fn reject_missing_vectors_for_kind() {
    let i = rejects(Builder::tiny().set("variants.0.selfcheck.vectors", arr(vec![])));
    assert!(matches!(i, Incompat::MissingVectors { .. }), "{i}");
    // a vector that only covers choice questions leaves noul uncovered
    let i = rejects(Builder::tiny().edit_manifest(|j| {
        crate::models::testing::jsonedit::remove(
            j,
            "variants.0.selfcheck.vectors.0.request.questions.angry",
        );
    }));
    assert!(
        matches!(i, Incompat::MissingVectors { ref kind, .. } if kind == "noul"),
        "{i}"
    );
}

#[test]
fn reject_malformed_vector() {
    let path = "variants.0.selfcheck.vectors.0.expected";
    let i = rejects(Builder::tiny().edit_manifest(move |j| {
        crate::models::testing::jsonedit::set(j, &format!("{path}.input_ids"), arr(vec![]));
    }));
    assert!(matches!(i, Incompat::BadVector { .. }), "{i}");
    let i = rejects(Builder::tiny().edit_manifest(move |j| {
        crate::models::testing::jsonedit::set(j, &format!("{path}.input_ids"), arr(vec![n(999)]));
    }));
    has(&i, "outside the vocabulary");
    let i = rejects(Builder::tiny().edit_manifest(move |j| {
        crate::models::testing::jsonedit::set(
            j,
            &format!("{path}.questions.angry.probabilities"),
            arr(vec![f(0.5), f(0.4)]),
        );
    }));
    has(&i, "sum to");
    let i = rejects(Builder::tiny().edit_manifest(move |j| {
        crate::models::testing::jsonedit::set(
            j,
            &format!("{path}.questions.angry.logits"),
            arr(vec![f(0.1)]),
        );
    }));
    assert!(matches!(i, Incompat::BadVector { .. }), "{i}");
}

#[test]
fn reject_invalid_calibration_declaration() {
    let cases: Vec<(&str, Json)> = vec![
        (
            "zero",
            obj(vec![
                ("declared", b(true)),
                ("temperature", f(0.0)),
                ("evidence", s("x")),
            ]),
        ),
        (
            "negative",
            obj(vec![
                ("declared", b(true)),
                ("temperature", f(-1.0)),
                ("evidence", s("x")),
            ]),
        ),
        (
            "no evidence",
            obj(vec![("declared", b(true)), ("temperature", f(2.0))]),
        ),
        (
            "blank evidence",
            obj(vec![
                ("declared", b(true)),
                ("temperature", f(2.0)),
                ("evidence", s("  ")),
            ]),
        ),
        (
            "no temperature",
            obj(vec![("declared", b(true)), ("evidence", s("x"))]),
        ),
        (
            "temperature when undeclared",
            obj(vec![("declared", b(false)), ("temperature", f(2.0))]),
        ),
    ];
    for (case, calibration) in cases {
        let i = rejects(Builder::tiny().set("variants.0.calibration", calibration));
        assert!(matches!(i, Incompat::BadCalibration { .. }), "{case}: {i}");
    }
}

#[test]
fn reject_missing_license() {
    assert!(matches!(
        rejects(Builder::tiny().set("license.spdx", s(""))),
        Incompat::MissingLicense
    ));
    assert!(matches!(
        rejects(Builder::tiny().set("license.spdx", s("  "))),
        Incompat::MissingLicense
    ));
    has(&rejects(Builder::tiny().remove("license")), "license");
}

fn task(id: &str, kind: &str) -> Json {
    obj(vec![
        ("id", s(id)),
        ("kind", s(kind)),
        ("match", obj(vec![])),
    ])
}

#[test]
fn reject_invalid_tasks() {
    let i = rejects(Builder::tiny().set("tasks", arr(vec![])));
    assert!(matches!(i, Incompat::BadTasks { index: 0, .. }), "{i}");
    let i =
        rejects(Builder::tiny().set("tasks", arr(vec![task("a", "choice"), task("a", "yes_no")])));
    assert!(matches!(i, Incompat::BadTasks { index: 1, .. }), "{i}");
    let i = rejects(Builder::tiny().set("tasks", arr(vec![task("a", "ranking")])));
    assert!(matches!(i, Incompat::BadTasks { index: 0, .. }), "{i}");
    let i = rejects(Builder::tiny().set("tasks", arr(vec![task(" ", "choice")])));
    assert!(matches!(i, Incompat::BadTasks { .. }), "{i}");
}

#[test]
fn reject_pointer_head_with_tasks() {
    let i = rejects(Builder::tiny_kev().set("tasks", arr(vec![task("a", "choice")])));
    assert!(matches!(i, Incompat::TasksNotSupported { .. }), "{i}");
    has(&i, "test-kev");
}

#[test]
fn reject_pointer_head_dimension() {
    let i = rejects(Builder::tiny_kev().set("head.d_model", n(16)));
    assert!(
        matches!(
            i,
            Incompat::HeadDimension {
                d_model: 16,
                n_embd: 32
            }
        ),
        "{i}"
    );
    has(&i, "16");
    has(&i, "32");
}

#[test]
fn reject_kev_template_on_a_family_that_is_not_kev() {
    let i = rejects(Builder::tiny_kev().set("template.id", s("chatml-letters")));
    assert!(matches!(i, Incompat::TemplateMismatch { .. }), "{i}");
}

// ---- stage 3: files ------------------------------------------------------------------------

#[test]
fn reject_file_missing() {
    let i = rejects(
        Builder::tiny().after_seal(|d| std::fs::remove_file(d.join("model.gguf")).unwrap()),
    );
    assert!(matches!(i, Incompat::FileMissing { .. }), "{i}");
    has(&i, "model.gguf");
}

#[test]
fn reject_size_mismatch() {
    let i = rejects(Builder::tiny().after_seal(|d| {
        let p = d.join("model.gguf");
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..bytes.len() - 1]).unwrap();
    }));
    assert!(matches!(i, Incompat::SizeMismatch { .. }), "{i}");
    has(&i, "delete it and retry");
}

#[test]
fn reject_sha_mismatch_local() {
    let i = rejects(Builder::tiny().after_seal(|d| {
        let p = d.join("model.gguf");
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&p, bytes).unwrap();
    }));
    assert!(matches!(i, Incompat::HashMismatch { .. }), "{i}");
    has(&i, "delete it and retry");
    // the tokenizer is checked too
    let i = rejects(Builder::tiny().after_seal(|d| {
        let p = d.join("tokenizer.json");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.push(b' ');
        std::fs::write(&p, bytes).unwrap();
    }));
    assert!(matches!(i, Incompat::SizeMismatch { .. }), "{i}");
}

// ---- stage 4: GGUF -------------------------------------------------------------------------

#[test]
fn reject_gguf_unreadable() {
    let i = rejects(Builder::tiny().edit_gguf(|b| b[0] = b'X'));
    assert!(matches!(i, Incompat::GgufUnreadable { .. }), "{i}");
    has(&i, "bad magic");
    let i = rejects(Builder::tiny().edit_gguf(|b| b[4..8].copy_from_slice(&99u32.to_le_bytes())));
    has(&i, "version 99");
    // cut inside the header: the counts then cannot be true for a file this short
    let i = rejects(Builder::tiny().edit_gguf(|b| b.truncate(40)));
    assert!(
        matches!(
            i,
            Incompat::GgufUnreadable { .. } | Incompat::GgufImpossible { .. }
        ),
        "{i}"
    );
}

#[test]
fn reject_gguf_impossible_offsets() {
    let i = rejects(
        Builder::tiny().edit_gguf(|b| b[8..16].copy_from_slice(&(1u64 << 60).to_le_bytes())),
    );
    assert!(matches!(i, Incompat::GgufImpossible { .. }), "{i}");
    // all tensors at offset 0: they overlap
    let count = crate::models::families::lookup("qwen2-letters")
        .map(|f| {
            let p = crate::models::families::ArchParams::from_pairs(vec![
                (
                    "n_layers".into(),
                    crate::models::families::ParamValue::Int(1),
                ),
                (
                    "n_embd".into(),
                    crate::models::families::ParamValue::Int(32),
                ),
                ("n_ff".into(), crate::models::families::ParamValue::Int(64)),
                (
                    "n_heads".into(),
                    crate::models::families::ParamValue::Int(2),
                ),
                (
                    "n_kv_heads".into(),
                    crate::models::families::ParamValue::Int(1),
                ),
                (
                    "n_vocab".into(),
                    crate::models::families::ParamValue::Int(32),
                ),
                (
                    "tie_embeddings".into(),
                    crate::models::families::ParamValue::Bool(true),
                ),
            ]);
            (f.tensors)(&p).len()
        })
        .unwrap();
    let i = rejects(Builder::tiny().tensor_offsets(vec![0; count]));
    has(&i, "overlap");
    // a tensor placed beyond the end of the file
    let i = rejects(Builder::tiny().tensor_offsets(vec![1 << 40; count]));
    has(&i, "past the end of the file");
}

#[test]
fn reject_gguf_metadata_mismatch() {
    use crate::models::testing::gguf_writer::Meta;
    let edit = |key: &'static str, v: Meta| {
        Builder::tiny().edit_meta(move |m| {
            for (k, val) in m.iter_mut() {
                if k == key {
                    *val = v.clone();
                }
            }
        })
    };
    let i = rejects(edit("qwen2.block_count", Meta::U32(7)));
    assert!(
        matches!(i, Incompat::GgufMetadata { ref key, .. } if key == "qwen2.block_count"),
        "{i}"
    );
    let i = rejects(edit("general.architecture", Meta::Str("llama".into())));
    assert!(
        matches!(i, Incompat::GgufMetadata { ref key, .. } if key == "general.architecture"),
        "{i}"
    );
    let i = rejects(edit("qwen2.embedding_length", Meta::U32(64)));
    assert!(matches!(i, Incompat::GgufMetadata { .. }), "{i}");
    let i = rejects(Builder::tiny().edit_meta(|m| m.retain(|(k, _)| k != "qwen2.context_length")));
    assert!(
        matches!(i, Incompat::GgufMetadata { ref key, .. } if key == "qwen2.context_length"),
        "{i}"
    );
}

#[test]
fn reject_max_context_over_gguf() {
    let i = rejects(Builder::tiny().context_length(256));
    assert!(
        matches!(
            i,
            Incompat::ContextTooLong {
                max_context: 512,
                context_length: 256
            }
        ),
        "{i}"
    );
}

#[test]
fn reject_tensor_missing() {
    let i = rejects(Builder::tiny().drop_tensor("blk.0.attn_q.weight"));
    assert!(
        matches!(i, Incompat::TensorMissing { ref name } if name == "blk.0.attn_q.weight"),
        "{i}"
    );
}

#[test]
fn reject_tensor_unexpected() {
    let i = rejects(Builder::tiny().add_tensor("blk.0.extra.weight", vec![32]));
    assert!(
        matches!(i, Incompat::TensorUnexpected { ref name } if name == "blk.0.extra.weight"),
        "{i}"
    );
}

#[test]
fn reject_tensor_shape() {
    let i = rejects(Builder::tiny().set_tensor_dims("token_embd.weight", vec![32, 31]));
    let Incompat::TensorShape {
        name,
        expected,
        got,
    } = &i
    else {
        panic!("{i}")
    };
    assert_eq!(
        (name.as_str(), expected.clone(), got.clone()),
        ("token_embd.weight", vec![32, 32], vec![32, 31])
    );
}

#[test]
fn reject_tensor_type() {
    let i = rejects(Builder::tiny().set_tensor_type("blk.0.attn_q.weight", 1));
    assert!(
        matches!(i, Incompat::TensorType { ref name, .. } if name == "blk.0.attn_q.weight"),
        "{i}"
    );
    has(&i, "Q8_0");
    // a vector that is not F32
    let i = rejects(Builder::tiny().set_tensor_type("output_norm.weight", 1));
    has(&i, "F32");
}

// ---- stage 5: tokenizer --------------------------------------------------------------------

#[test]
fn reject_tokenizer_unreadable() {
    let i = rejects(Builder::tiny().edit_tokenizer(|t| {
        crate::models::testing::jsonedit::set(t, "model.type", s("Nope"));
    }));
    assert!(matches!(i, Incompat::TokenizerUnreadable { .. }), "{i}");
}

#[test]
fn reject_tokenizer_truncation_or_padding() {
    let i = rejects(Builder::tiny().edit_tokenizer(|t| {
        crate::models::testing::jsonedit::set(
            t,
            "truncation",
            obj(vec![
                ("direction", s("Right")),
                ("max_length", n(512)),
                ("strategy", s("LongestFirst")),
                ("stride", n(0)),
            ]),
        );
    }));
    assert!(
        matches!(i, Incompat::TokenizerTruncationOrPadding { ref which } if which == "truncation"),
        "{i}"
    );
    let i = rejects(Builder::tiny().edit_tokenizer(|t| {
        crate::models::testing::jsonedit::set(
            t,
            "padding",
            obj(vec![
                ("strategy", obj(vec![("Fixed", n(8))])),
                ("direction", s("Right")),
                ("pad_to_multiple_of", Json::Null),
                ("pad_id", n(0)),
                ("pad_type_id", n(0)),
                ("pad_token", s("<unk>")),
            ]),
        );
    }));
    assert!(
        matches!(i, Incompat::TokenizerTruncationOrPadding { ref which } if which == "padding"),
        "{i}"
    );
}

#[test]
fn reject_special_roles() {
    let i = rejects(Builder::tiny().remove("tokenizer.special_tokens.im_end"));
    assert!(matches!(i, Incompat::SpecialRoles { .. }), "{i}");
    let i = rejects(Builder::tiny().set(
        "tokenizer.special_tokens.extra",
        obj(vec![("text", s("<|im_start|>")), ("id", n(1))]),
    ));
    assert!(matches!(i, Incompat::SpecialRoles { .. }), "{i}");
}

#[test]
fn reject_special_token_text_or_id() {
    let i = rejects(Builder::tiny().set(
        "tokenizer.special_tokens.im_end",
        obj(vec![("text", s("<|nope|>")), ("id", n(2))]),
    ));
    assert!(
        matches!(i, Incompat::SpecialTokenText { found: None, .. }),
        "{i}"
    );
    let i = rejects(Builder::tiny().set(
        "tokenizer.special_tokens.im_end",
        obj(vec![("text", s("<|im_end|>")), ("id", n(9))]),
    ));
    assert!(
        matches!(
            i,
            Incompat::SpecialTokenText {
                found: Some(2),
                expected_id: 9,
                ..
            }
        ),
        "{i}"
    );
}

#[test]
fn reject_special_token_not_single() {
    // "a-b" is one entry of the vocabulary but the pre-tokenizer cuts it in three
    let i = rejects(
        Builder::tiny()
            .edit_tokenizer(|t| {
                crate::models::testing::jsonedit::set(t, "model.vocab.a-b", n(20));
            })
            .set(
                "tokenizer.special_tokens.im_end",
                obj(vec![("text", s("a-b")), ("id", n(20))]),
            ),
    );
    let Incompat::SpecialTokenNotSingle { ids, .. } = &i else {
        panic!("{i}")
    };
    assert_eq!(ids.len(), 3);
}

#[test]
fn reject_special_roles_same_id() {
    let i = rejects(Builder::tiny().set(
        "tokenizer.special_tokens.im_end",
        obj(vec![("text", s("<|im_start|>")), ("id", n(1))]),
    ));
    assert!(matches!(i, Incompat::SpecialSameId { id: 1, .. }), "{i}");
}

#[test]
fn reject_letters_targets_invalid() {
    let i = rejects(Builder::tiny().set("head.choice_targets.0.id", n(99)));
    assert!(matches!(i, Incompat::LettersTarget { id: 99, .. }), "{i}");
    has(&i, "outside the vocabulary");
    let i = rejects(Builder::tiny().set("head.choice_targets.1.id", n(3)));
    assert!(matches!(i, Incompat::LettersTarget { id: 3, .. }), "{i}");
    has(&i, "already used");
    let i = rejects(Builder::tiny().set("head.choice_targets.0.label", s("B")));
    assert!(matches!(i, Incompat::LettersTarget { .. }), "{i}");
}

#[test]
fn reject_letters_label_not_single_token() {
    let i = rejects(Builder::tiny().set(
        "head.yes_no_targets.yes",
        obj(vec![("label", s("A-B")), ("id", n(7))]),
    ));
    let Incompat::LettersLabelNotSingle { label, ids } = &i else {
        panic!("{i}")
    };
    assert_eq!((label.as_str(), ids.len()), ("A-B", 3));
}

// ---- stage 6: pointer head weights are covered by head::tests::each_defect_is_a_head_weights_error
