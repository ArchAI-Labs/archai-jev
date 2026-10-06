//! The tokenizer of a model as the prompt builders use it (spec 006a section 4).
//!
//! Text is sanitized (section 3) and tokenized with the pinned `tokenizer.json` **without** special
//! tokens added. Delimiters never come from text: they are looked up by role and inserted as ids.
//! The backend's own tokenizer is never used (D24).

use std::path::Path;

use tokenizers::Tokenizer;

use super::sanitize::sanitize;
use super::unsupported::Unsupported;
use crate::error::{Error, Result};
use crate::models::incompat::Incompat;

/// The special-token roles of a template and their ids (from the manifest, already validated by
/// stage 5 of model loading).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roles(Vec<(String, u32)>);

impl Roles {
    /// Build from `(role, id)` pairs.
    pub fn new(pairs: impl IntoIterator<Item = (impl Into<String>, u32)>) -> Self {
        Roles(pairs.into_iter().map(|(r, id)| (r.into(), id)).collect())
    }

    /// The id of `role`.
    pub fn id(&self, role: &str) -> Option<u32> {
        self.0.iter().find(|(r, _)| r == role).map(|(_, id)| *id)
    }

    /// The ids of exactly `wanted`, in that order.
    ///
    /// # Errors
    /// [`Incompat::SpecialRoles`] if a role is missing (model loading checks this first; this is
    /// the defence in depth for a template used with the wrong tokenizer).
    pub fn require<const N: usize>(&self, wanted: [&str; N]) -> Result<[u32; N]> {
        let mut out = [0u32; N];
        for (slot, role) in out.iter_mut().zip(wanted) {
            *slot = self.id(role).ok_or_else(|| {
                Error::IncompatibleModel(Incompat::SpecialRoles {
                    expected: wanted.iter().map(|r| (*r).to_string()).collect(),
                    got: self.0.iter().map(|(r, _)| r.clone()).collect(),
                })
            })?;
        }
        Ok(out)
    }
}

/// A tokenizer plus the special-token ids of the model.
pub struct PromptTokenizer {
    tk: Tokenizer,
    roles: Roles,
}

impl PromptTokenizer {
    /// Wrap a loaded tokenizer.
    pub fn new(tk: Tokenizer, roles: Roles) -> Self {
        PromptTokenizer { tk, roles }
    }

    /// Load `tokenizer.json` from `path` (already pinned and checked by model loading).
    ///
    /// # Errors
    /// [`Incompat::TokenizerUnreadable`] if the file cannot be read or parsed.
    pub fn from_file(path: &Path, roles: Roles) -> Result<Self> {
        let unreadable = |detail: String| {
            Error::IncompatibleModel(Incompat::TokenizerUnreadable {
                path: path.display().to_string(),
                detail,
            })
        };
        let bytes = std::fs::read(path).map_err(|e| unreadable(format!("cannot read: {e}")))?;
        let tk = Tokenizer::from_bytes(&bytes).map_err(|e| unreadable(e.to_string()))?;
        Ok(PromptTokenizer { tk, roles })
    }

    /// The special-token ids.
    pub fn roles(&self) -> &Roles {
        &self.roles
    }

    /// Size of the vocabulary, added tokens included.
    pub fn vocab_size(&self) -> usize {
        self.tk.get_vocab_size(true)
    }

    /// Sanitize `text` and tokenize it. `segment` names the text for error messages.
    ///
    /// # Errors
    /// [`Unsupported::Untokenizable`] if the tokenizer fails (an internal fault).
    pub fn encode_user(&self, text: &str, segment: &str) -> Result<Vec<u32>> {
        self.encode_raw(&sanitize(text), segment)
    }

    /// Tokenize `text` as it is: for text that is ours (a fixed system prompt), never the user's.
    ///
    /// # Errors
    /// [`Unsupported::Untokenizable`] if the tokenizer fails (an internal fault).
    pub fn encode_raw(&self, text: &str, segment: &str) -> Result<Vec<u32>> {
        self.tk
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| {
                Error::Unsupported(Unsupported::Untokenizable {
                    segment: segment.to_string(),
                    cause: e.to_string(),
                })
            })
    }
}

/// Tokenizers for tests: byte-level with no merges (one token per byte), NFC normalization like
/// Qwen, and the given special tokens. Any text tokenizes, different byte strings give different
/// ids, and `é` written two ways gives the same ids: enough to test prompts without a 20 MiB file.
#[cfg(any(test, feature = "testing"))]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
pub mod testkit {
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, Tokenizer};

    use super::{PromptTokenizer, Roles};

    /// A byte-level tokenizer with `specials` (`(role, text)`) added after the 256 byte tokens.
    pub fn byte_level(specials: &[(&str, &str)]) -> PromptTokenizer {
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        assert_eq!(
            alphabet.len(),
            256,
            "the byte-level alphabet has 256 symbols"
        );
        let vocab: [(String, u32); 256] =
            std::array::from_fn(|i| (alphabet[i].to_string(), i as u32));
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .expect("byte-level BPE");
        let mut tk = Tokenizer::new(bpe);
        tk.with_normalizer(Some(NFC));
        tk.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        let added: Vec<AddedToken> = specials
            .iter()
            .map(|(_, text)| AddedToken::from(*text, true))
            .collect();
        tk.add_special_tokens(&added);
        let roles = Roles::new(
            specials
                .iter()
                .map(|(role, text)| (*role, tk.token_to_id(text).expect("special token id"))),
        );
        PromptTokenizer::new(tk, roles)
    }

    /// The five delimiters of the `kev` template.
    pub fn kev() -> PromptTokenizer {
        byte_level(&[
            ("fim_prefix", "<|fim_prefix|>"),
            ("fim_middle", "<|fim_middle|>"),
            ("fim_suffix", "<|fim_suffix|>"),
            ("box_start", "<|box_start|>"),
            ("box_end", "<|box_end|>"),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_testkit_tokenizes_any_text_one_token_per_byte() {
        let tok = testkit::kev();
        assert_eq!(tok.encode_raw("abc", "t").unwrap().len(), 3);
        assert_eq!(tok.encode_raw("é", "t").unwrap().len(), 2);
        // NFC: the decomposed form gives the same ids as the composed one.
        assert_eq!(
            tok.encode_raw("e\u{301}", "t").unwrap(),
            tok.encode_raw("\u{e9}", "t").unwrap()
        );
        assert_eq!(tok.encode_raw("", "t").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn user_text_cannot_become_a_delimiter() {
        let tok = testkit::kev();
        let id = tok.roles().id("fim_prefix").unwrap();
        // Raw text does become the special token; user text is sanitized first.
        assert!(tok.encode_raw("<|fim_prefix|>", "t").unwrap().contains(&id));
        assert!(
            !tok.encode_user("<|fim_prefix|>", "t")
                .unwrap()
                .contains(&id)
        );
    }

    #[test]
    fn roles_require_reports_missing_roles() {
        let roles = Roles::new([("a", 1u32)]);
        assert_eq!(roles.require(["a"]).unwrap(), [1]);
        assert!(matches!(
            roles.require(["a", "b"]),
            Err(Error::IncompatibleModel(Incompat::SpecialRoles { .. }))
        ));
    }
}
