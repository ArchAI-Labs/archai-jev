//! The letters-at-the-end family `spark2_5-letters` on a tiny fake checkpoint: it loads through
//! the generic pipeline, and every way its manifest, its tokenizer or its GGUF can be wrong is
//! one named refusal that names the field (spec 010b, AC-11 and AC-12).

use super::Harness;
use crate::error::Error;
use crate::models::gguf::GgmlType;
use crate::models::incompat::Incompat;
use crate::models::testing::builder::Builder;
use crate::models::testing::gguf_writer::Meta;
use crate::models::testing::jsonedit::{arr, n, obj, s};

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
    assert_eq!(h.records(), 0, "no record may be written: {err}");
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

// ---- it loads, and it is not calibrated -----------------------------------------------------

#[test]
fn a_tiny_spark_checkpoint_loads_through_the_generic_pipeline() {
    let fixture = Builder::tiny_spark().build();
    let h = Harness::new();
    let model = h.load(&h.local(fixture.path())).expect("loads");
    let d = &model.data;
    assert_eq!(
        (d.family.as_str(), d.head.as_str(), d.template.as_str()),
        ("spark2_5-letters", "answer_letters", "spark-decisions-v3")
    );
    assert_eq!((d.max_options, d.tasks.as_ref()), (Some(26), None));
    assert!(d.notice.as_deref().is_some_and(|n| !n.is_empty()));
    assert_eq!(h.backend.loads(), 1);
    assert_eq!(h.backend.runs(), 1, "the self-check ran once");
}

#[test]
fn an_uncalibrated_model_needs_a_temperature_or_consent() {
    let fixture = Builder::tiny_spark().build();
    let h = Harness::new();
    let mut req = h.local(fixture.path());
    req.allow_uncalibrated = false;
    let err = h.load(&req).err().expect("refused");
    let Error::IncompatibleModel(i @ Incompat::Uncalibrated { .. }) = err else {
        panic!("{err:?}")
    };
    // the message names both ways out
    has(&i, "temperature");
    has(&i, "allow_uncalibrated");
    assert_eq!(h.backend.loads(), 0);

    req.allow_uncalibrated = true;
    let model = h.load(&req).expect("consent given");
    assert_eq!(
        (
            model.data.calibrated,
            model.data.temperature,
            model.data.calibration_source
        ),
        (false, 1.0, "none")
    );

    let mut req = h.local(fixture.path());
    req.allow_uncalibrated = false;
    req.temperature = Some(1.4);
    let model = h.load(&req).expect("a temperature of the user");
    assert_eq!(
        (
            model.data.calibrated,
            model.data.temperature,
            model.data.calibration_source
        ),
        (true, 1.4, "user")
    );
}

// ---- the manifest ---------------------------------------------------------------------------

#[test]
fn the_head_and_the_template_must_be_the_ones_of_the_family() {
    // the head of the v1 model has other fields: the strict reader stops at the first it lacks
    let i = rejects(Builder::tiny_spark().set("head.kind", s("letters")));
    assert!(matches!(i, Incompat::ManifestMissingField { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set("head.kind", s("nope")));
    assert!(matches!(i, Incompat::HeadMismatch { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set("template.version", n(2)));
    assert!(matches!(i, Incompat::TemplateMismatch { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set("template.id", s("chatml-letters")));
    assert!(matches!(i, Incompat::TemplateMismatch { .. }), "{i}");
}

#[test]
fn a_closed_task_set_is_not_supported_and_null_is() {
    let i = rejects(Builder::tiny_spark().set(
        "tasks",
        arr(vec![obj(vec![
            ("id", s("t")),
            ("kind", s("choice")),
            ("match", obj(vec![])),
        ])]),
    ));
    assert!(matches!(i, Incompat::TasksNotSupported { .. }), "{i}");
    // `tasks: null` is what the fixture has, and it loads
    let fixture = Builder::tiny_spark().build();
    let h = Harness::new();
    assert!(h.load(&h.local(fixture.path())).is_ok());
}

#[test]
fn max_options_is_from_one_to_the_number_of_letters() {
    for bad in [0u64, 27, 100] {
        let i = rejects(Builder::tiny_spark().set("head.max_options", n(bad)));
        assert!(
            matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "head.max_options"),
            "{bad}: {i}"
        );
    }
    // fewer letters than max_options
    let i = rejects(Builder::tiny_spark().edit_manifest(|m| {
        let letters = crate::models::testing::jsonedit::arr(
            (0..3u64)
                .map(|k| {
                    obj(vec![
                        (
                            "label",
                            s(&char::from(b'A' + u8::try_from(k).unwrap()).to_string()),
                        ),
                        ("id", n(7 + k)),
                    ])
                })
                .collect(),
        );
        crate::models::testing::jsonedit::set(m, "head.letters", letters);
    }));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "head.max_options"),
        "{i}"
    );
    has(&i, "(3)");
}

#[test]
fn the_letters_are_a_to_z_in_order_with_distinct_ids() {
    let i = rejects(Builder::tiny_spark().set("head.letters.0.label", s("B")));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "head.letters[0].label"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().set("head.letters.4.id", n(7)));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "head.letters[4].id"),
        "{i}"
    );
    has(&i, "already");
    let i = rejects(Builder::tiny_spark().remove("head.letters.2.id"));
    assert!(matches!(i, Incompat::ManifestMissingField { .. }), "{i}");
    has(&i, "letters");
    let i = rejects(Builder::tiny_spark().set("head.letters", arr(vec![])));
    assert!(matches!(i, Incompat::ManifestBadValue { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set("head.rule", s("full_softmax")));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "head.rule"),
        "{i}"
    );
}

#[test]
fn a_letter_the_tokenizer_cannot_give_is_refused_by_name() {
    // the id is outside the vocabulary
    let i = rejects(Builder::tiny_spark().set("head.letters.0.id", n(900)));
    assert!(matches!(i, Incompat::LettersTarget { id: 900, .. }), "{i}");
    has(&i, "outside the vocabulary");
    // the tokenizer gives the letter another id than the manifest says
    let i = rejects(Builder::tiny_spark().set("head.letters.3.id", n(40)));
    assert!(matches!(i, Incompat::LettersTarget { id: 40, .. }), "{i}");
    // the letter is not one token: a pre-tokenizer that splits on "B" leaves nothing of it
    let i = rejects(Builder::tiny_spark().edit_tokenizer(|t| {
        crate::models::testing::jsonedit::set(
            t,
            "pre_tokenizer",
            obj(vec![
                ("type", s("CharDelimiterSplit")),
                ("delimiter", s("B")),
            ]),
        );
    }));
    let Incompat::LettersLabelNotSingle { label, ids } = &i else {
        panic!("{i}")
    };
    assert_eq!((label.as_str(), ids.len()), ("B", 0));
}

#[test]
fn the_roles_the_template_needs_are_all_declared() {
    for role in ["bos", "system", "user", "bot", "eos", "think_end"] {
        let path = format!("tokenizer.special_tokens.{role}");
        let i = rejects(Builder::tiny_spark().edit_manifest(move |m| {
            crate::models::testing::jsonedit::remove(m, &path);
        }));
        assert!(matches!(i, Incompat::SpecialRoles { .. }), "{role}: {i}");
        has(&i, role);
    }
}

#[test]
fn the_notice_and_the_control_tokens_are_required() {
    let i = rejects(Builder::tiny_spark().remove("notice"));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "notice"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().set("notice", s("  ")));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "notice"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().remove("tokenizer.control_tokens"));
    assert!(
        matches!(i, Incompat::ManifestMissingField { ref path } if path == "tokenizer.control_tokens"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().set("tokenizer.control_tokens", arr(vec![])));
    assert!(matches!(i, Incompat::ManifestMissingField { .. }), "{i}");
}

#[test]
fn the_control_tokens_are_exactly_the_added_tokens_of_the_tokenizer() {
    // one added token is missing from the list: text could still produce it
    let i = rejects(Builder::tiny_spark().remove("tokenizer.control_tokens.5"));
    let Incompat::ControlTokensMismatch { unlisted, unknown } = &i else {
        panic!("{i}")
    };
    assert_eq!((unlisted.len(), unknown.len()), (1, 0));
    has(&i, &unlisted[0]);
    // one token of the list is not in the tokenizer
    let i = rejects(Builder::tiny_spark().set("tokenizer.control_tokens.0", s("<not-there>")));
    let Incompat::ControlTokensMismatch { unlisted, unknown } = &i else {
        panic!("{i}")
    };
    assert_eq!(
        (unlisted.len(), unknown.as_slice()),
        (1, ["<not-there>".to_string()].as_slice())
    );
    // an added token that is not special is a control token too (the tokenizer extracts it)
    let i = rejects(Builder::tiny_spark().edit_tokenizer(|t| {
        let tokens = match t {
            crate::json_strict::Json::Object(p) => p
                .iter_mut()
                .find(|(k, _)| k == "added_tokens")
                .map(|(_, v)| v)
                .unwrap(),
            _ => unreachable!(),
        };
        if let crate::json_strict::Json::Array(a) = tokens {
            a.push(obj(vec![
                ("id", n(60)),
                ("content", s("w60")),
                ("single_word", crate::models::testing::jsonedit::b(false)),
                ("lstrip", crate::models::testing::jsonedit::b(false)),
                ("rstrip", crate::models::testing::jsonedit::b(false)),
                ("normalized", crate::models::testing::jsonedit::b(false)),
                ("special", crate::models::testing::jsonedit::b(false)),
            ]));
        }
    }));
    let Incompat::ControlTokensMismatch { unlisted, .. } = &i else {
        panic!("{i}")
    };
    assert_eq!(unlisted.as_slice(), ["w60".to_string()]);
}

#[test]
fn the_list_of_control_tokens_is_read_strictly() {
    let i = rejects(Builder::tiny_spark().set("tokenizer.control_tokens", s("<bos>")));
    assert!(matches!(i, Incompat::ManifestBadValue { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set("tokenizer.control_tokens.1", n(3)));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "tokenizer.control_tokens[1]"),
        "{i}"
    );
    // a token the neutralisation could not break: too short, or holding its own mark
    for bad in ["", "<", "a\u{a6}b"] {
        let i = rejects(Builder::tiny_spark().set("tokenizer.control_tokens.1", s(bad)));
        assert!(
            matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "tokenizer.control_tokens[1]"),
            "{bad:?}: {i}"
        );
    }
    let i = rejects(Builder::tiny_spark().set("tokenizer.control_tokens.1", s("<bos>")));
    assert!(matches!(i, Incompat::ManifestBadValue { .. }), "{i}");
    has(&i, "twice");
    // a family that does not neutralise control tokens must not carry a list nobody applies
    let i = rejects(Builder::tiny().set(
        "tokenizer.control_tokens",
        arr(vec![s("<|im_start|>"), s("<|im_end|>")]),
    ));
    assert!(
        matches!(i, Incompat::ManifestBadValue { ref path, .. } if path == "tokenizer.control_tokens"),
        "{i}"
    );
}

// ---- the GGUF: tensors and metadata ---------------------------------------------------------

#[test]
fn a_tensor_missing_unexpected_misshapen_or_mistyped_is_refused_by_name() {
    let i = rejects(Builder::tiny_spark().drop_tensor("blk.2.attn_gate.weight"));
    assert!(
        matches!(i, Incompat::TensorMissing { ref name } if name == "blk.2.attn_gate.weight"),
        "{i}"
    );
    // a model with separate q, k and v is another architecture file, not this one
    let i = rejects(Builder::tiny_spark().add_tensor("blk.0.attn_q.weight", vec![32, 32]));
    assert!(
        matches!(i, Incompat::TensorUnexpected { ref name } if name == "blk.0.attn_q.weight"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().add_tensor("output.weight", vec![32, 64]));
    assert!(matches!(i, Incompat::TensorUnexpected { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set_tensor_dims("blk.1.attn_qkv.weight", vec![32, 96]));
    assert!(
        matches!(i, Incompat::TensorShape { ref name, .. } if name == "blk.1.attn_qkv.weight"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().set_tensor_dims("blk.3.attn_gate.weight", vec![2, 32]));
    assert!(matches!(i, Incompat::TensorShape { .. }), "{i}");
    let i = rejects(Builder::tiny_spark().set_tensor_type("blk.0.ffn_up.weight", GgmlType::F16.0));
    assert!(
        matches!(i, Incompat::TensorType { ref name, .. } if name == "blk.0.ffn_up.weight"),
        "{i}"
    );
    // norms are always F32
    let i =
        rejects(Builder::tiny_spark().set_tensor_type("blk.0.attn_norm.weight", GgmlType::Q8_0.0));
    assert!(matches!(i, Incompat::TensorType { .. }), "{i}");
}

fn replace_meta(key: &'static str, value: Meta) -> impl Fn(&mut Vec<(String, Meta)>) + 'static {
    move |meta| {
        for (k, v) in meta.iter_mut() {
            if k == key {
                *v = value.clone();
            }
        }
    }
}

#[test]
fn the_metadata_of_the_architecture_is_checked_including_the_attention_pattern() {
    let key = "spark2_5.attention.sliding_window_pattern";
    // all layers sliding: the pattern of the file is not the one of the manifest
    let i = rejects(Builder::tiny_spark().edit_meta(replace_meta(
        key,
        Meta::BoolArray(vec![true, true, true, true]),
    )));
    let Incompat::GgufMetadata {
        key: k,
        expected,
        got,
    } = &i
    else {
        panic!("{i}")
    };
    assert_eq!(k, key);
    assert_eq!(expected, "[true, true, true, false]");
    assert!(got.contains("BoolArray"), "{got}");
    // the pattern is missing, or is not an array of booleans
    let i = rejects(Builder::tiny_spark().edit_meta(move |m| m.retain(|(k, _)| k != key)));
    assert!(matches!(i, Incompat::GgufMetadata { .. }), "{i}");
    has(&i, "missing");
    let i = rejects(Builder::tiny_spark().edit_meta(replace_meta(key, Meta::U32(1))));
    assert!(matches!(i, Incompat::GgufMetadata { .. }), "{i}");
    // a shorter pattern than the number of layers
    let i = rejects(
        Builder::tiny_spark().edit_meta(replace_meta(key, Meta::BoolArray(vec![true, true, true]))),
    );
    assert!(matches!(i, Incompat::GgufMetadata { .. }), "{i}");
}

#[test]
fn every_other_key_of_the_architecture_is_compared_with_the_manifest() {
    for (key, wrong) in [
        ("spark2_5.block_count", 5u32),
        ("spark2_5.embedding_length", 48),
        ("spark2_5.feed_forward_length", 80),
        ("spark2_5.attention.head_count", 4),
        ("spark2_5.attention.head_count_kv", 2),
        ("spark2_5.attention.key_length", 64),
        ("spark2_5.attention.value_length", 64),
        ("spark2_5.vocab_size", 65),
        ("spark2_5.attention.sliding_window", 512),
    ] {
        let i = rejects(Builder::tiny_spark().edit_meta(replace_meta(key, Meta::U32(wrong))));
        assert!(
            matches!(i, Incompat::GgufMetadata { key: ref k, .. } if k == key),
            "{key}: {i}"
        );
    }
    let i = rejects(Builder::tiny_spark().edit_meta(replace_meta(
        "general.architecture",
        Meta::Str("qwen2".to_string()),
    )));
    assert!(matches!(i, Incompat::GgufMetadata { .. }), "{i}");
}

#[test]
fn a_manifest_that_asks_for_more_context_than_the_file_has_is_refused() {
    let i = rejects(
        Builder::tiny_spark()
            .context_length(256)
            .set("max_context", n(512)),
    );
    assert!(matches!(i, Incompat::ContextTooLong { .. }), "{i}");
}

// ---- what the strict reader needs to accept a manifest without a calibration (AC-12) ---------

#[test]
fn a_manifest_without_license_or_vectors_for_both_kinds_is_not_accepted() {
    let i = rejects(Builder::tiny_spark().set("license.spdx", s("")));
    assert!(matches!(i, Incompat::MissingLicense), "{i}");
    let i = rejects(Builder::tiny_spark().set("variants.0.selfcheck.vectors", arr(vec![])));
    assert!(matches!(i, Incompat::MissingVectors { .. }), "{i}");
    // a vector that covers only the choice question leaves the yes/no kind uncovered
    let i = rejects(Builder::tiny_spark().edit_manifest(|m| {
        crate::models::testing::jsonedit::remove(
            m,
            "variants.0.selfcheck.vectors.0.request.questions.angry",
        );
    }));
    assert!(
        matches!(i, Incompat::MissingVectors { ref kind, .. } if kind == "noul"),
        "{i}"
    );
    let i = rejects(Builder::tiny_spark().edit_manifest(|m| {
        crate::models::testing::jsonedit::remove(
            m,
            "variants.0.selfcheck.vectors.0.request.questions.team",
        );
    }));
    assert!(
        matches!(i, Incompat::MissingVectors { ref kind, .. } if kind == "choice"),
        "{i}"
    );
}

#[test]
fn a_calibration_that_is_declared_is_read_like_any_other() {
    // the family may be declared calibrated by a manifest that has the evidence; this is the
    // same reader as for every family, so a half-written declaration is refused as always
    let i = rejects(Builder::tiny_spark().set(
        "variants.0.calibration",
        obj(vec![(
            "declared",
            crate::models::testing::jsonedit::b(true),
        )]),
    ));
    assert!(matches!(i, Incompat::BadCalibration { .. }), "{i}");
}

#[test]
fn a_dtype_that_was_not_measured_has_no_tolerance_and_is_refused() {
    for dtype in ["f32", "bf16"] {
        let i = rejects(Builder::tiny_spark().dtype(dtype));
        assert!(
            matches!(i, Incompat::NoTolerance { ref dtype, .. } if !dtype.is_empty()),
            "{dtype}: {i}"
        );
    }
}

#[test]
fn a_tokenizer_that_finds_its_control_tokens_in_another_way_than_the_text_is_refused() {
    use crate::models::testing::jsonedit::{b, set};
    // a normalizer that rewrites text (full-width bars become ASCII) before tokens are found
    let i = rejects(Builder::tiny_spark().edit_tokenizer(|t| {
        set(t, "normalizer", obj(vec![("type", s("NFKC"))]));
    }));
    assert!(matches!(i, Incompat::ControlTokenMatching { .. }), "{i}");
    has(&i, "NFKC");
    // canonical composition and an empty sequence do not
    for normalizer in [
        obj(vec![("type", s("NFC"))]),
        obj(vec![("type", s("Sequence")), ("normalizers", arr(vec![]))]),
    ] {
        let fixture = Builder::tiny_spark()
            .edit_tokenizer(move |t| set(t, "normalizer", normalizer.clone()))
            .build();
        let h = Harness::new();
        assert!(h.load(&h.local(fixture.path())).is_ok());
    }
    // an added token matched after normalisation, or with the spaces around it eaten
    for flag in ["normalized", "lstrip", "rstrip", "single_word"] {
        let i = rejects(Builder::tiny_spark().edit_tokenizer(move |t| {
            set(t, &format!("added_tokens.0.{flag}"), b(true));
        }));
        assert!(
            matches!(i, Incompat::ControlTokenMatching { .. }),
            "{flag}: {i}"
        );
    }
}
