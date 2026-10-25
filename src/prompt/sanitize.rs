//! Sanitization of the user's text before tokenization (spec 006a section 3).
//!
//! Kev rewrites every `<|name|>` (name = one or more ASCII letters, digits or `_`) as
//! `<¦name¦>` (U+00A6) so that nobody can write a delimiter or a control token by hand. This is
//! the same rule, written as a plain scan so that no regex crate is needed. Text that does not
//! match is left alone, **including** the added tokens that do not have the `<|..|>` shape
//! (`<think>`, `<tool_call>`, ...): Kev leaves them, and parity with Kev is the point (D36).

use std::borrow::Cow;

/// The character that replaces the two bars: BROKEN BAR.
const BROKEN_BAR: char = '\u{a6}';

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Rewrite every `<|name|>` of `text` as `<¦name¦>`; the rest is unchanged.
///
/// Matches are leftmost and do not overlap, so `<|x|><|y|>` gives `<¦x¦><¦y¦>` and
/// `<|<|x|>|>` gives `<|<¦x¦>|>`. Returns the input untouched (no allocation) when there is
/// nothing to rewrite.
pub fn sanitize(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut copied = 0usize;
    let mut i = 0usize;
    while let Some(rel) = text.get(i..).and_then(|t| t.find("<|")) {
        let start = i + rel;
        let name_start = start + 2;
        let mut end = name_start;
        while bytes.get(end).copied().is_some_and(is_name_byte) {
            end += 1;
        }
        let closed = end > name_start
            && bytes.get(end).copied() == Some(b'|')
            && bytes.get(end + 1).copied() == Some(b'>');
        if closed {
            let buf = out.get_or_insert_with(|| String::with_capacity(text.len() + 8));
            buf.push_str(text.get(copied..start).unwrap_or_default());
            buf.push('<');
            buf.push(BROKEN_BAR);
            buf.push_str(text.get(name_start..end).unwrap_or_default());
            buf.push(BROKEN_BAR);
            buf.push('>');
            i = end + 2;
            copied = i;
        } else {
            // `<` is one byte, so `start + 1` is a character boundary.
            i = start + 1;
        }
    }
    match out {
        Some(mut buf) => {
            buf.push_str(text.get(copied..).unwrap_or_default());
            Cow::Owned(buf)
        }
        None => Cow::Borrowed(text),
    }
}

/// Smallest number of characters of a control token: neutralising puts a mark after the first
/// character, which only separates the others from it if there is at least one.
pub const MIN_CONTROL_TOKEN_CHARS: usize = 2;

/// Why a control token cannot be used by [`sanitize_control`], if it cannot.
///
/// The proof that no token survives (and that a second pass changes nothing) needs every token
/// to have at least [`MIN_CONTROL_TOKEN_CHARS`] characters and none to contain the mark itself,
/// which would let the mark put by one token build another.
pub fn control_token_problem(token: &str) -> Option<&'static str> {
    if token.chars().count() < MIN_CONTROL_TOKEN_CHARS {
        Some("must have at least 2 characters")
    } else if token.contains(BROKEN_BAR) {
        Some("must not contain the character U+00A6 (it is what neutralises a token)")
    } else {
        None
    }
}

/// Neutralise every occurrence of every token of `tokens` in `text` (spec 010b).
///
/// A model whose special tokens do not look like `<|name|>` (full-width bars, `<think>`) cannot
/// use [`sanitize`]: the text of a request would still be able to write one. This puts
/// U+00A6 after the first character of **every** occurrence, overlapping ones included
/// (`</think>` becomes `<¦/think>`), so what the tokenizer sees is ordinary text. With tokens
/// that satisfy [`control_token_problem`] no occurrence is left, so a second pass changes
/// nothing. Returns the input untouched (no allocation) when no token occurs.
pub fn sanitize_control<'a>(text: &'a str, tokens: &[String]) -> Cow<'a, str> {
    // Only a position that holds the first character of some token can start one: most tokens
    // begin with the same `<`, so almost every position is dismissed by a comparison.
    let mut firsts: Vec<char> = tokens.iter().filter_map(|t| t.chars().next()).collect();
    firsts.sort_unstable();
    firsts.dedup();
    let hits = |rest: &str| {
        tokens
            .iter()
            .any(|t| !t.is_empty() && rest.starts_with(t.as_str()))
    };
    let mut out: Option<String> = None;
    let mut copied = 0usize;
    for (i, c) in text.char_indices() {
        if firsts.binary_search(&c).is_err() || !hits(text.get(i..).unwrap_or_default()) {
            continue;
        }
        let after_first = i + c.len_utf8();
        let buf = out.get_or_insert_with(|| String::with_capacity(text.len() + 8));
        buf.push_str(text.get(copied..after_first).unwrap_or_default());
        buf.push(BROKEN_BAR);
        copied = after_first;
    }
    match out {
        Some(mut buf) => {
            buf.push_str(text.get(copied..).unwrap_or_default());
            Cow::Owned(buf)
        }
        None => Cow::Borrowed(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn s(x: &str) -> String {
        sanitize(x).into_owned()
    }

    #[test]
    fn rewrites_any_ascii_name_in_any_case() {
        assert_eq!(s("<|fim_prefix|>"), "<¦fim_prefix¦>");
        assert_eq!(s("<|FIM_PREFIX|>"), "<¦FIM_PREFIX¦>");
        assert_eq!(s("<|im_start|>"), "<¦im_start¦>");
        assert_eq!(s("<|x_1|>"), "<¦x_1¦>");
        assert_eq!(s("a <|box_end|> b"), "a <¦box_end¦> b");
    }

    #[test]
    fn near_misses_are_left_alone() {
        for t in [
            "<|fim-prefix|>",
            "<|fim_prefix|",
            "<|a|b|>",
            "<||>",
            "<|>",
            "<|é|>",
            "<| fim_prefix |>",
            "<think>",
            "<tool_call>",
            "plain text",
            "",
        ] {
            assert_eq!(s(t), t, "{t:?}");
            assert!(
                matches!(sanitize(t), Cow::Borrowed(_)),
                "{t:?} must not allocate"
            );
        }
    }

    #[test]
    fn adjacent_and_nested_matches() {
        assert_eq!(s("<|x|><|y|>"), "<¦x¦><¦y¦>");
        assert_eq!(s("<|<|x|>|>"), "<|<¦x¦>|>");
        assert_eq!(s("<<|x|>>"), "<<¦x¦>>");
        assert_eq!(s("<|a|><|"), "<¦a¦><|");
    }

    /// An independent check: does `text` still contain `<|name|>` with an ASCII name?
    fn has_delimiter(text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len() {
            if chars[i] == '<' && chars.get(i + 1) == Some(&'|') {
                let mut j = i + 2;
                while chars
                    .get(j)
                    .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
                {
                    j += 1;
                }
                if j > i + 2 && chars.get(j) == Some(&'|') && chars.get(j + 1) == Some(&'>') {
                    return true;
                }
            }
        }
        false
    }

    fn fragments() -> impl Strategy<Value = String> {
        let pieces: Vec<&'static str> = vec![
            "<|",
            "|>",
            "<",
            ">",
            "|",
            "_",
            "a",
            "Z",
            "9",
            "x_y",
            "fim_prefix",
            "box_end",
            "im_start",
            "endoftext",
            "<|fim_prefix|>",
            "<|im_end|>",
            "<|x|>",
            " ",
            "\n",
            "é",
            "\u{212a}",
            "\u{37e}",
            "\u{1fef}",
            "\u{a6}",
            "<think>",
            "-",
            "ñ",
        ];
        prop::collection::vec(prop::sample::select(pieces), 0..14).prop_map(|v| v.concat())
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 12_000, rng_seed: proptest::test_runner::RngSeed::Fixed(7), ..ProptestConfig::default() })]

        #[test]
        fn idempotent_and_leaves_no_delimiter(text in fragments()) {
            let once = sanitize(&text).into_owned();
            prop_assert!(!has_delimiter(&once), "{once:?}");
            prop_assert_eq!(sanitize(&once).into_owned(), once);
        }

        #[test]
        fn only_bars_inside_matches_change(text in fragments()) {
            let out = sanitize(&text).into_owned();
            // Same text apart from the replaced bars: removing every BROKEN BAR and every '|'
            // gives the same string (the input has no BROKEN BAR unless the fragments add it).
            let strip = |t: &str| t.chars().filter(|c| *c != '|' && *c != '\u{a6}').collect::<String>();
            prop_assert_eq!(strip(&out), strip(&text));
        }
    }
}
