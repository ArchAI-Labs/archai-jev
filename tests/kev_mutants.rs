//! Do the 8 self-check vectors of Kev notice a prompt builder that is wrong in one way?
//! (spec 006a, mutants of spike S2 section 10). Nine deliberately wrong builders; each must give
//! other ids than Kev on at least one vector. Needs the Kev `tokenizer.json` (test level T1).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use _core::QuestionKind;
use _core::json_strict::Json;
use _core::models::testing::{assets, golden};
use _core::prompt::{PromptTokenizer, Roles, sanitize};
use tokenizers::Tokenizer;

fn roles() -> Roles {
    Roles::new([
        ("fim_prefix", 248060u32),
        ("fim_middle", 248061),
        ("fim_suffix", 248062),
        ("box_start", 248049),
        ("box_end", 248050),
    ])
}

#[derive(Clone, Copy, Debug)]
enum Mutant {
    BaseTokenizer,
    NoSanitization,
    HardenedSanitization,
    YesNoFalseTrue,
    ScorePrefix,
    StateAsJson,
    ListWithoutDashes,
    NullDescriptionAsNone,
    NoNfc,
}

const ALL: [Mutant; 9] = [
    Mutant::BaseTokenizer,
    Mutant::NoSanitization,
    Mutant::HardenedSanitization,
    Mutant::YesNoFalseTrue,
    Mutant::ScorePrefix,
    Mutant::StateAsJson,
    Mutant::ListWithoutDashes,
    Mutant::NullDescriptionAsNone,
    Mutant::NoNfc,
];

/// The tokenizer of Kev with one defect, or untouched.
fn tokenizer(path: &std::path::Path, mutant: Option<Mutant>) -> PromptTokenizer {
    let text = std::fs::read_to_string(path).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
    match mutant {
        Some(Mutant::BaseTokenizer) => {
            // The pre-tokenizer of the official Qwen3.5 tokenizer.json keeps combining marks
            // (`\p{M}`) inside words; the one Kev really uses (and archai-jev) does not.
            let mut raw = serde_json::to_string(&json).unwrap();
            let before = raw.clone();
            raw = raw.replace("\\\\p{L}+|", "[\\\\p{L}\\\\p{M}]+|");
            raw = raw.replace(
                "[^\\\\s\\\\p{L}\\\\p{N}]+",
                "[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+",
            );
            assert_ne!(raw, before, "the regex of the pre-tokenizer was not found");
            json = serde_json::from_str(&raw).unwrap();
        }
        Some(Mutant::NoNfc) => {
            assert!(json["normalizer"]["type"] == "NFC", "no NFC to remove");
            json["normalizer"] = serde_json::Value::Null;
        }
        _ => {}
    }
    let tk = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
    PromptTokenizer::new(tk, roles())
}

const TOOLS: [&str; 10] = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
];

fn clean(m: Option<Mutant>, tok: &PromptTokenizer, text: &str) -> Vec<u32> {
    let text = match m {
        Some(Mutant::NoSanitization) => text.to_string(),
        Some(Mutant::HardenedSanitization) => {
            let mut t = sanitize(text).into_owned();
            for tool in TOOLS {
                t = t.replace(tool, &tool.replace('<', "<\u{a6}"));
            }
            t
        }
        _ => sanitize(text).into_owned(),
    };
    tok.encode_raw(&text, "t").unwrap()
}

/// Kev's prompt, written again from the rules, with the chosen defect.
fn mutated_ids(case: &golden::GoldenCase, tok: &PromptTokenizer, m: Option<Mutant>) -> Vec<u32> {
    let (state, questions) = golden::request_to_domain(&case.request).unwrap();
    let [prefix, middle, suffix, bstart, bend] = tok
        .roles()
        .require([
            "fim_prefix",
            "fim_middle",
            "fim_suffix",
            "box_start",
            "box_end",
        ])
        .unwrap();
    let mut state_text = state.canonical_text().to_string();
    if matches!(m, Some(Mutant::StateAsJson))
        && !matches!(case.request.get("state"), Some(Json::Str(_)))
    {
        state_text = case.request.get("state").unwrap().to_canonical_string();
    }
    if matches!(m, Some(Mutant::ListWithoutDashes)) {
        state_text = state_text
            .lines()
            .map(|l| {
                let indent = l.len() - l.trim_start().len();
                format!(
                    "{}{}",
                    &l[..indent],
                    l.trim_start().trim_start_matches("- ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    let mut ids = vec![prefix];
    ids.extend(clean(m, tok, &state_text));
    for (_, q) in questions.iter() {
        ids.push(middle);
        ids.extend(clean(m, tok, q.instructions().rendered()));
        for (j, text) in q.option_texts().iter().enumerate() {
            let mut text = text.clone();
            match (m, q.kind()) {
                (Some(Mutant::YesNoFalseTrue), QuestionKind::YesNo) => {
                    text = if j == 0 {
                        text.replacen("no", "false", 1)
                    } else {
                        text.replacen("yes", "true", 1)
                    };
                }
                (Some(Mutant::ScorePrefix), QuestionKind::Score) => text = format!("{j}: {text}"),
                (Some(Mutant::NullDescriptionAsNone), QuestionKind::Choice)
                    if !text.contains(": ") =>
                {
                    text = format!("{text}: None");
                }
                _ => {}
            }
            ids.push(bstart);
            ids.extend(clean(m, tok, &text));
            ids.push(bend);
        }
        ids.push(suffix);
    }
    ids
}

#[test]
fn the_unmutated_rebuild_is_kevs_prompt() {
    let Some(path) = assets::kev_tokenizer() else {
        return;
    };
    let tok = tokenizer(&path, None);
    for case in golden::load("selfcheck.jsonl") {
        assert_eq!(mutated_ids(&case, &tok, None), case.ids, "{}", case.id);
    }
}

#[test]
fn every_wrong_prompt_builder_is_noticed_by_the_eight_vectors() {
    let Some(path) = assets::kev_tokenizer() else {
        return;
    };
    let cases = golden::load("selfcheck.jsonl");
    assert_eq!(cases.len(), 8);
    for m in ALL {
        let tok = tokenizer(&path, Some(m));
        let caught: Vec<&str> = cases
            .iter()
            .filter(|c| mutated_ids(c, &tok, Some(m)) != c.ids)
            .map(|c| c.id.as_str())
            .collect();
        assert!(
            !caught.is_empty(),
            "{m:?} goes unnoticed by every self-check vector"
        );
        eprintln!("{m:?}: caught by {caught:?}");
    }
}
