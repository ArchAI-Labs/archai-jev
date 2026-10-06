//! Parity of the prompt ids with Kev (spec 006a, test level T1): the library must build **exactly**
//! the ids Kev builds, for every request of the golden, with the real `tokenizer.json` of Kev.
//!
//! Needs the tokenizer (19 MiB, not in the repository): see `models::testing::assets`. In CI a
//! missing file fails the test (`ARCHAI_JEV_REQUIRE_ASSETS=1`); locally the test says what is
//! missing and is skipped.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use _core::Error;
use _core::models::testing::golden::{self, GoldenCase};
use _core::models::testing::{assets, jsonedit};
use _core::prompt::Prompt;
use _core::prompt::{NoLimits, PromptTokenizer, RequestLimits, Roles, lookup};

const ALL: [&str; 3] = ["choice", "noul", "score"];

/// Ids of the five delimiters in the tokenizer of Kev-0.8B (spec 006a section 2).
fn kev_roles() -> Roles {
    Roles::new([
        ("fim_prefix", 248060u32),
        ("fim_middle", 248061),
        ("fim_suffix", 248062),
        ("box_start", 248049),
        ("box_end", 248050),
    ])
}

fn tokenizer() -> Option<PromptTokenizer> {
    let path = assets::kev_tokenizer()?;
    Some(PromptTokenizer::from_file(&path, kev_roles()).expect("kev tokenizer.json"))
}

fn limits() -> RequestLimits<'static> {
    RequestLimits {
        model: "kev-0.8b",
        max_context: u64::MAX,
        question_types: &ALL,
        support: &NoLimits,
    }
}

fn build(case: &GoldenCase, tok: &PromptTokenizer) -> Result<Prompt, String> {
    let (state, questions) = golden::request_to_domain(&case.request)?;
    lookup("kev", 1)
        .expect("kev v1")
        .build(&state, &questions, tok, &limits())
        .map_err(|e: Error| e.to_string())
}

/// Requests Kev accepts and the library refuses **on purpose** (stricter than Kev: D30, D32),
/// with the reason. Anything else that fails to convert or to build is a bug.
const REFUSED_ON_PURPOSE: &[(&str, &str)] = &[
    (
        "choice-desc-structured",
        "a description that is a number or a boolean (001/003: text, object or list only)",
    ),
    (
        "noul-true-plus-extra-keys",
        "unknown key in the criteria of a yes/no question (D31/D32)",
    ),
    (
        "noul-extra-keys",
        "unknown key in the criteria of a yes/no question (D31/D32)",
    ),
    (
        "score-nonstring-levels",
        "a Score level that is a number (001/003: text, object or list only)",
    ),
    (
        "state-null",
        "a null state (D30: root must be a string, an object or a list)",
    ),
    ("state-int", "a number as the whole state (D30)"),
    ("state-bool", "a boolean as the whole state (D30)"),
    (
        "instr-empty-or-odd",
        "null instructions (D30: instructions must not be empty)",
    ),
];

fn all_cases() -> Vec<GoldenCase> {
    let mut cases = golden::load("golden.jsonl");
    cases.extend(golden::load("golden_extra.jsonl"));
    cases
}

fn first_difference(got: &[u32], want: &[u32]) -> String {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    format!(
        "first difference at position {at}: got {:?}, expected {:?} (lengths {} and {})",
        got.get(at),
        want.get(at),
        got.len(),
        want.len()
    )
}

#[test]
fn packed_ids_equal_golden() {
    let Some(tok) = tokenizer() else { return };
    let mut failures = Vec::new();
    let mut refused = Vec::new();
    let mut checked = 0;
    for case in all_cases() {
        match build(&case, &tok) {
            Err(reason) => refused.push((case.id.clone(), reason)),
            Ok(prompt) => {
                checked += 1;
                let got = prompt.packed_ids();
                if got != case.ids {
                    failures.push(format!(
                        "{}: {}",
                        case.id,
                        first_difference(&got, &case.ids)
                    ));
                    continue;
                }
                // The positions the head will read from are the golden's.
                assert_eq!(case.state_len, prompt.state().len(), "{}", case.id);
                for (i, gq) in case.questions.iter().enumerate() {
                    let start = prompt.branch_start(i).unwrap();
                    let b = &prompt.branches()[i];
                    assert_eq!(
                        start, gq.branch_start,
                        "{} {}: branch start",
                        case.id, gq.name
                    );
                    assert_eq!(
                        b.len(),
                        gq.branch_len,
                        "{} {}: branch length",
                        case.id,
                        gq.name
                    );
                    assert_eq!(
                        start + b.decide(),
                        gq.decide_pos,
                        "{} {}: decide",
                        case.id,
                        gq.name
                    );
                    let ends: Vec<usize> = b.option_ends().iter().map(|e| start + e).collect();
                    assert_eq!(ends, gq.opt_pos, "{} {}: option ends", case.id, gq.name);
                }
                assert_eq!(
                    prompt.packed_len(),
                    case.usage,
                    "{}: usage.input_tokens",
                    case.id
                );
            }
        }
    }
    assert!(
        failures.is_empty(),
        "id mismatches:\n{}",
        failures.join("\n")
    );
    let expected: Vec<&str> = REFUSED_ON_PURPOSE.iter().map(|(id, _)| *id).collect();
    let got: Vec<&str> = refused.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(got, expected, "requests the library refuses: {refused:#?}");
    eprintln!(
        "ids identical for {checked} requests; refused on purpose: {}",
        refused.len()
    );
    assert!(checked >= 60);
}

#[test]
fn selfcheck_vector_ids() {
    let Some(tok) = tokenizer() else { return };
    let vectors = golden::load("selfcheck.jsonl");
    assert_eq!(vectors.len(), 8);
    for case in vectors {
        let prompt = build(&case, &tok).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(prompt.packed_ids(), case.ids, "{}", case.id);
    }
}

fn question(options: Vec<(&str, Option<&str>)>) -> _core::Questions {
    use _core::{Choice, ChoiceOption, Question, Questions, TextEntry};
    let opts = options
        .into_iter()
        .map(|(k, d)| match d {
            Some(d) => ChoiceOption::described(k, d).unwrap(),
            None => ChoiceOption::key(k),
        })
        .collect();
    let q = Choice::new(TextEntry::text("Which?").unwrap(), opts).unwrap();
    Questions::new([("q", Question::from(q))]).unwrap()
}

fn state(text: &str) -> _core::State {
    _core::State::new(_core::StateValue::string(text)).unwrap()
}

fn ids_of(tok: &PromptTokenizer, st: &_core::State, qs: &_core::Questions) -> Vec<u32> {
    lookup("kev", 1)
        .unwrap()
        .build(st, qs, tok, &limits())
        .unwrap()
        .packed_ids()
}

#[test]
fn nfd_equals_nfc_and_added_tokens_outside_the_sanitizer_stay_single() {
    let Some(tok) = tokenizer() else { return };
    let qs = question(vec![("a", None)]);
    // NFC is done by the tokenizer, after the sanitizer.
    assert_eq!(
        ids_of(&tok, &state("caf\u{e9}"), &qs),
        ids_of(&tok, &state("cafe\u{301}"), &qs)
    );
    // `<think>` is an added token that the sanitizer does not touch (as in Kev): one id, 248068.
    let with_think = ids_of(&tok, &state("<think>"), &qs);
    let without = ids_of(&tok, &state(""), &qs);
    assert_eq!(with_think.len(), without.len() + 1);
    assert!(with_think.contains(&248_068));
    // `<|im_start|>` is sanitized, so it never becomes a control token.
    let sanitized = ids_of(&tok, &state("<|im_start|>"), &qs);
    assert!(!sanitized.contains(&248_045), "im_start must not appear");
}

#[test]
fn options_that_are_the_same_tokens_are_refused_with_the_real_tokenizer() {
    let Some(tok) = tokenizer() else { return };
    let st = state("s");
    for (a, b) in [("\u{e9}", "e\u{301}"), ("<|x|>", "<\u{a6}x\u{a6}>")] {
        let qs = question(vec![(a, None), (b, None)]);
        let err = lookup("kev", 1)
            .unwrap()
            .build(&st, &qs, &tok, &limits())
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{a:?} {b:?}: {err}");
    }
    for (a, b) in [("a", "A"), ("a", "a ")] {
        let qs = question(vec![(a, None), (b, None)]);
        assert!(
            lookup("kev", 1)
                .unwrap()
                .build(&st, &qs, &tok, &limits())
                .is_ok()
        );
    }
}

#[test]
fn json_helpers_are_reachable() {
    // Keeps `jsonedit` linked for the other tests of this file (compile-time check only).
    let _ = jsonedit::s("x");
}
