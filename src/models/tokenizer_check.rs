//! Stage 5: the tokenizer and the tokens the manifest relies on (spec 005, 6.3 and 6.4).
//!
//! The hash pins the file; this checks its *content*, because a pinned wrong file (such as an
//! adapter's `tokenizer.json` with truncation to 512 tokens) would be accepted by the hash alone.

use std::path::Path;

use tokenizers::Tokenizer;

use super::families::Family;
use super::incompat::Incompat;
use super::manifest::{HeadSpec, Manifest, Target};

fn ids_of(tk: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    tk.encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| e.to_string())
}

/// Check `tokenizer.json` at `path` against the manifest and the family.
///
/// # Errors
/// [`Incompat`] for an unreadable tokenizer, truncation or padding, wrong special-token roles,
/// tokens that are missing, have another id, or are not a single token, and invalid answer targets.
pub fn check(
    path: &Path,
    manifest: &Manifest,
    family: &Family,
    n_vocab: u64,
) -> Result<(), Incompat> {
    let shown = path.display().to_string();
    let unreadable = |detail: String| Incompat::TokenizerUnreadable {
        path: shown.clone(),
        detail,
    };
    let bytes = std::fs::read(path).map_err(|e| unreadable(format!("cannot read: {e}")))?;
    let tk = Tokenizer::from_bytes(&bytes).map_err(|e| unreadable(e.to_string()))?;

    if tk.get_truncation().is_some() {
        return Err(Incompat::TokenizerTruncationOrPadding {
            which: "truncation".to_string(),
        });
    }
    if tk.get_padding().is_some() {
        return Err(Incompat::TokenizerTruncationOrPadding {
            which: "padding".to_string(),
        });
    }

    let mut got: Vec<String> = manifest
        .tokenizer
        .special_tokens
        .iter()
        .map(|(r, _)| r.clone())
        .collect();
    let mut expected: Vec<String> = family
        .special_roles
        .iter()
        .map(|r| (*r).to_string())
        .collect();
    got.sort();
    expected.sort();
    if got != expected {
        return Err(Incompat::SpecialRoles { expected, got });
    }

    for (role, tok) in &manifest.tokenizer.special_tokens {
        match tk.token_to_id(&tok.text) {
            None => {
                return Err(Incompat::SpecialTokenText {
                    role: role.clone(),
                    text: tok.text.clone(),
                    expected_id: tok.id,
                    found: None,
                });
            }
            Some(real) if real != tok.id => {
                return Err(Incompat::SpecialTokenText {
                    role: role.clone(),
                    text: tok.text.clone(),
                    expected_id: tok.id,
                    found: Some(real),
                });
            }
            Some(_) => {}
        }
        let ids = ids_of(&tk, &tok.text).map_err(unreadable)?;
        if ids != [tok.id] {
            return Err(Incompat::SpecialTokenNotSingle {
                role: role.clone(),
                text: tok.text.clone(),
                ids,
            });
        }
    }
    let toks = &manifest.tokenizer.special_tokens;
    for (i, (a, ta)) in toks.iter().enumerate() {
        if let Some((b, _)) = toks.iter().take(i).find(|(_, tb)| tb.id == ta.id) {
            return Err(Incompat::SpecialSameId {
                a: b.clone(),
                b: a.clone(),
                id: ta.id,
            });
        }
    }

    let targets: Vec<&Target> = match &manifest.head {
        HeadSpec::Letters {
            choice_targets,
            yes_no,
            ..
        } => {
            let mut all: Vec<&Target> = choice_targets.iter().collect();
            if let Some((no, yes)) = yes_no {
                all.push(no);
                all.push(yes);
            }
            all
        }
        HeadSpec::AnswerLetters { letters, .. } => letters.iter().collect(),
        HeadSpec::KeyLogProb { .. } | HeadSpec::Pointer { .. } => Vec::new(),
    };
    check_targets(&tk, &targets, n_vocab, &shown)?;
    check_control_tokens(&tk, &manifest.tokenizer.control_tokens)
}

/// Every answer target is below the vocabulary size, has an id of its own, and is encoded alone
/// (no leading space, no special token) as exactly the one token the manifest says.
fn check_targets(
    tk: &Tokenizer,
    targets: &[&Target],
    n_vocab: u64,
    shown: &str,
) -> Result<(), Incompat> {
    for (i, t) in targets.iter().enumerate() {
        if u64::from(t.id) >= n_vocab {
            return Err(Incompat::LettersTarget {
                label: t.label.clone(),
                id: t.id,
                detail: format!("the id is outside the vocabulary (size {n_vocab})"),
            });
        }
        if let Some(other) = targets.iter().take(i).find(|o| o.id == t.id) {
            return Err(Incompat::LettersTarget {
                label: t.label.clone(),
                id: t.id,
                detail: format!("the id is already used by '{}'", other.label),
            });
        }
    }
    for t in targets {
        let ids = ids_of(tk, &t.label).map_err(|detail| Incompat::TokenizerUnreadable {
            path: shown.to_string(),
            detail,
        })?;
        if ids.len() != 1 {
            return Err(Incompat::LettersLabelNotSingle {
                label: t.label.clone(),
                ids,
            });
        }
        if ids.first() != Some(&t.id) {
            return Err(Incompat::LettersTarget {
                label: t.label.clone(),
                id: t.id,
                detail: format!(
                    "the tokenizer encodes the label to id {:?}, not {}",
                    ids.first(),
                    t.id
                ),
            });
        }
    }
    Ok(())
}

/// Whether a normalizer (as JSON) leaves the text as it is for the purpose of finding control
/// tokens: none, canonical composition (which cannot build the characters they are made of from
/// others), or a sequence of those.
fn keeps_text(normalizer: &serde_json::Value) -> bool {
    match normalizer.get("type").and_then(serde_json::Value::as_str) {
        Some("NFC") => true,
        Some("Sequence") => normalizer
            .get("normalizers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| items.iter().all(keeps_text)),
        _ => normalizer.is_null(),
    }
}

/// The list of control tokens of the manifest is the set of added tokens of the tokenizer,
/// special or not: the `tokenizers` crate extracts every added token from the text it encodes,
/// so one missing from the list would still be able to forge the structure of a prompt.
fn check_control_tokens(tk: &Tokenizer, listed: &[String]) -> Result<(), Incompat> {
    if listed.is_empty() {
        return Ok(());
    }
    // The neutralisation looks for the exact text of a token. It only protects if the tokenizer
    // looks for the same: no normalizer that rewrites text before the tokens are found, and no
    // added token matched after normalisation or with the spaces around it eaten.
    let normalizer = tk
        .get_normalizer()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| Incompat::ControlTokenMatching {
            detail: format!("the normalizer cannot be read: {e}"),
        })?;
    if let Some(n) = normalizer.as_ref().filter(|n| !keeps_text(n)) {
        return Err(Incompat::ControlTokenMatching {
            detail: format!("the normalizer {n} may rewrite text before tokens are found"),
        });
    }
    if let Some(t) = tk
        .get_added_tokens_decoder()
        .into_values()
        .find(|t| t.normalized || t.lstrip || t.rstrip || t.single_word)
    {
        return Err(Incompat::ControlTokenMatching {
            detail: format!(
                "the added token {:?} is matched with normalized={}, lstrip={}, rstrip={}, single_word={}",
                t.content, t.normalized, t.lstrip, t.rstrip, t.single_word
            ),
        });
    }
    let mut added: Vec<String> = tk
        .get_added_tokens_decoder()
        .into_values()
        .map(|t| t.content)
        .collect();
    added.sort();
    let mut listed: Vec<String> = listed.to_vec();
    listed.sort();
    let unlisted: Vec<String> = added
        .iter()
        .filter(|t| listed.binary_search(t).is_err())
        .cloned()
        .collect();
    let unknown: Vec<String> = listed
        .iter()
        .filter(|t| added.binary_search(t).is_err())
        .cloned()
        .collect();
    if unlisted.is_empty() && unknown.is_empty() {
        Ok(())
    } else {
        Err(Incompat::ControlTokensMismatch { unlisted, unknown })
    }
}
