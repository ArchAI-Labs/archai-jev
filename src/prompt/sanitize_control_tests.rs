//! `sanitize_control`: the control tokens of a model, whatever their shape, cannot be written by
//! the text of a request (spec 010b, AC-4 and AC-6).

use std::borrow::Cow;

use proptest::prelude::*;

use super::sanitize::{control_token_problem, sanitize_control};

fn toks(list: &[&str]) -> Vec<String> {
    list.iter().map(|t| (*t).to_string()).collect()
}

fn c(text: &str, list: &[&str]) -> String {
    sanitize_control(text, &toks(list)).into_owned()
}

/// Tokens of the shapes a model can have: full-width bars, `<|name|>`, `<think>`, `</think>`.
const SHAPES: &[&str] = &[
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<\u{ff5c}start\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<|Bot|>",
    "<think>",
    "</think>",
    "<tool_call>",
];

#[test]
fn control_tokens_of_every_shape_are_neutralised() {
    assert_eq!(c("a </think> b", SHAPES), "a <\u{a6}/think> b");
    assert_eq!(
        c("<think>x</think>", SHAPES),
        "<\u{a6}think>x<\u{a6}/think>"
    );
    assert_eq!(c("<|Bot|>", SHAPES), "<\u{a6}|Bot|>");
    assert_eq!(
        c("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", SHAPES),
        "<\u{a6}\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"
    );
    // case matters: the tokenizer matches exactly
    assert_eq!(c("</THINK>", SHAPES), "</THINK>");
}

#[test]
fn nothing_to_neutralise_means_no_allocation() {
    for t in [
        "",
        "plain text",
        "<think",
        "think>",
        "< /think>",
        "<\u{ff5c}end>",
    ] {
        assert!(
            matches!(sanitize_control(t, &toks(SHAPES)), Cow::Borrowed(_)),
            "{t:?}"
        );
    }
    assert!(matches!(sanitize_control("<think>", &[]), Cow::Borrowed(_)));
}

#[test]
fn overlapping_and_nested_occurrences_are_all_broken() {
    assert_eq!(
        c("</think></think>", SHAPES),
        "<\u{a6}/think><\u{a6}/think>"
    );
    // an occurrence written around another one
    assert_eq!(c("</</think>think>", SHAPES), "</<\u{a6}/think>think>");
    // two tokens that overlap each other: both get their mark
    assert_eq!(c("abcd", &["abc", "bcd"]), "a\u{a6}b\u{a6}cd");
    // the same token overlapping itself
    assert_eq!(c("aaa", &["aa"]), "a\u{a6}a\u{a6}a");
    // a token that is the prefix of another: the first mark breaks both
    assert_eq!(c("<think>", &["<think", "<think>"]), "<\u{a6}think>");
}

#[test]
fn unlisted_tokens_and_the_old_delimiter_are_not_touched() {
    // only the listed tokens are control tokens for this model
    assert_eq!(c("<|im_start|> <unk>", SHAPES), "<|im_start|> <unk>");
    // multibyte characters at the cut are kept whole
    assert_eq!(
        c("\u{1f600}</think>\u{1f600}", SHAPES),
        "\u{1f600}<\u{a6}/think>\u{1f600}"
    );
}

#[test]
fn a_token_must_have_two_characters_and_no_mark() {
    assert!(control_token_problem("<think>").is_none());
    assert!(control_token_problem("ab").is_none());
    assert!(control_token_problem("").is_some());
    assert!(control_token_problem("a").is_some());
    assert!(control_token_problem("\u{ff5c}").is_some());
    assert!(control_token_problem("a\u{a6}b").is_some());
}

/// Why the two conditions of `control_token_problem` are needed: without them a token survives.
#[test]
fn the_conditions_are_not_decoration() {
    // a one-character token cannot be separated from itself
    assert!(c("x", &["x"]).contains('x'));
    // a token that contains the mark can be built by the mark of another token
    assert!(c("xyz", &["xyz", "x\u{a6}y"]).contains("x\u{a6}y"));
}

fn fragments() -> impl Strategy<Value = String> {
    let pieces: Vec<&'static str> = vec![
        "<",
        ">",
        "/",
        "think",
        "<think>",
        "</think>",
        "<tool_call>",
        "<|Bot|>",
        "|",
        "Bot",
        "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        "\u{ff5c}",
        "end",
        "\u{2581}",
        "of",
        "sentence",
        "x",
        " ",
        "\n",
        "\u{a6}",
        "\u{1f600}",
        "<<",
        ">>",
    ];
    prop::collection::vec(prop::sample::select(pieces), 0..16).prop_map(|v| v.concat())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12_000, rng_seed: proptest::test_runner::RngSeed::Fixed(11), ..ProptestConfig::default() })]

    #[test]
    fn no_listed_token_survives_and_a_second_pass_changes_nothing(text in fragments()) {
        let list = toks(SHAPES);
        let once = sanitize_control(&text, &list).into_owned();
        for t in &list {
            prop_assert!(!once.contains(t.as_str()), "{t:?} survived in {once:?}");
        }
        prop_assert_eq!(sanitize_control(&once, &list).into_owned(), once.clone());
        // only marks were added
        let unmarked: String = once.chars().filter(|c| *c != '\u{a6}').collect();
        let original: String = text.chars().filter(|c| *c != '\u{a6}').collect();
        prop_assert_eq!(unmarked, original);
    }

    #[test]
    fn text_without_a_token_comes_back_borrowed(text in fragments()) {
        let list = toks(SHAPES);
        let found = list.iter().any(|t| text.contains(t.as_str()));
        prop_assert_eq!(matches!(sanitize_control(&text, &list), Cow::Borrowed(_)), !found);
    }
}
