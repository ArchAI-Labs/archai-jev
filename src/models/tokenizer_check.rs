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

    if let HeadSpec::Letters {
        choice_targets,
        yes_no,
        ..
    } = &manifest.head
    {
        let mut targets: Vec<&Target> = choice_targets.iter().collect();
        if let Some((no, yes)) = yes_no {
            targets.push(no);
            targets.push(yes);
        }
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
            let ids = ids_of(&tk, &t.label).map_err(unreadable)?;
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
    }
    Ok(())
}
