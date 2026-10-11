//! What identifies a saved prefix: who computed it (the fingerprint) and which tokens it is.
//!
//! A saved prefix is only valid for the same model, the same template and the same engine
//! settings (D20: never serve a cache of another model or template). Every component is hashed
//! with its length, so two different splits of the same bytes never give the same fingerprint.

use sha2::{Digest, Sha256};

/// What makes two engines produce the same numbers for the same tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FingerprintParts<'a> {
    /// The model and its revision (the SHA-256 of the canonical manifest, or name + revision).
    pub model: &'a str,
    /// SHA-256 of the GGUF the engine runs (of the converted file for converted models).
    pub gguf_sha256: &'a str,
    /// The supported family.
    pub family: &'a str,
    /// The weights variant (`q8_0`, `bf16`...).
    pub dtype: &'a str,
    /// Id of the prompt template.
    pub template_id: &'a str,
    /// Version of the prompt template.
    pub template_version: u64,
    /// SHA-256 of the pinned `tokenizer.json`.
    pub tokenizer_sha256: &'a str,
    /// The engine and its settings as they really are (library version, cache types, context
    /// size...): built from the parameters used, not from a constant.
    pub engine: &'a str,
}

/// The identity of a loaded model and engine, hashed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint([u8; 32]);

fn field(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

impl Fingerprint {
    /// Hash the parts.
    pub fn new(parts: &FingerprintParts<'_>) -> Self {
        let mut h = Sha256::new();
        for text in [
            parts.model,
            parts.gguf_sha256,
            parts.family,
            parts.dtype,
            parts.template_id,
            parts.tokenizer_sha256,
            parts.engine,
        ] {
            field(&mut h, text.as_bytes());
        }
        field(&mut h, &parts.template_version.to_le_bytes());
        Fingerprint(h.finalize().into())
    }
}

/// The key of a saved prefix: SHA-256 of the fingerprint and the token ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixKey([u8; 32]);

impl PrefixKey {
    /// Its first eight bytes, to remember a key cheaply.
    pub fn short(&self) -> u64 {
        let [a, b, c, d, e, f, g, h, ..] = self.0;
        u64::from_le_bytes([a, b, c, d, e, f, g, h])
    }
}

/// Keys of the prefixes of one row, in one pass over its ids.
#[derive(Clone)]
pub struct KeyHasher {
    inner: Sha256,
    tokens: usize,
}

impl KeyHasher {
    /// Start from a fingerprint.
    pub fn new(fingerprint: &Fingerprint) -> Self {
        let mut inner = Sha256::new();
        inner.update(fingerprint.0);
        KeyHasher { inner, tokens: 0 }
    }

    /// Add the next token ids of the prefix.
    pub fn update(&mut self, ids: &[u32]) {
        for id in ids {
            self.inner.update(id.to_le_bytes());
        }
        self.tokens += ids.len();
    }

    /// The key of everything added so far (the hasher can keep going).
    pub fn key(&self) -> PrefixKey {
        let mut h = self.inner.clone();
        h.update((self.tokens as u64).to_le_bytes());
        PrefixKey(h.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: FingerprintParts<'static> = FingerprintParts {
        model: "model-a@rev1",
        gguf_sha256: "aaaa",
        family: "qwen2-letters",
        dtype: "q8_0",
        template_id: "chatml-letters",
        template_version: 1,
        tokenizer_sha256: "tttt",
        engine: "llama-cpp-2=0.1.157; n_ctx=1024",
    };

    fn key(p: &FingerprintParts<'_>, ids: &[u32]) -> PrefixKey {
        let mut k = KeyHasher::new(&Fingerprint::new(p));
        k.update(ids);
        k.key()
    }

    #[test]
    fn each_component_changes_the_key() {
        let ids = [1, 2, 3];
        let base = key(&BASE, &ids);
        assert_eq!(base, key(&BASE, &ids), "deterministic");
        let others = [
            (
                "model",
                FingerprintParts {
                    model: "model-a@rev2",
                    ..BASE
                },
            ),
            (
                "gguf",
                FingerprintParts {
                    gguf_sha256: "bbbb",
                    ..BASE
                },
            ),
            (
                "family",
                FingerprintParts {
                    family: "qwen35-pointer",
                    ..BASE
                },
            ),
            (
                "dtype",
                FingerprintParts {
                    dtype: "bf16",
                    ..BASE
                },
            ),
            (
                "template id",
                FingerprintParts {
                    template_id: "kev",
                    ..BASE
                },
            ),
            (
                "template version",
                FingerprintParts {
                    template_version: 2,
                    ..BASE
                },
            ),
            (
                "tokenizer",
                FingerprintParts {
                    tokenizer_sha256: "uuuu",
                    ..BASE
                },
            ),
            (
                "engine",
                FingerprintParts {
                    engine: "llama-cpp-2=0.1.157; n_ctx=2048",
                    ..BASE
                },
            ),
        ];
        for (what, parts) in others {
            assert_ne!(key(&parts, &ids), base, "{what}");
        }
        // And the prefix itself: other ids, other length.
        assert_ne!(key(&BASE, &[1, 2, 4]), base);
        assert_ne!(key(&BASE, &[1, 2]), base);
        assert_ne!(key(&BASE, &[1, 2, 3, 0]), base);
    }

    #[test]
    fn components_are_not_confused_by_moving_a_boundary() {
        let a = FingerprintParts {
            model: "ab",
            gguf_sha256: "c",
            ..BASE
        };
        let b = FingerprintParts {
            model: "a",
            gguf_sha256: "bc",
            ..BASE
        };
        assert_ne!(Fingerprint::new(&a), Fingerprint::new(&b));
    }

    #[test]
    fn the_hasher_gives_the_key_of_every_cut_in_one_pass() {
        let fp = Fingerprint::new(&BASE);
        let mut k = KeyHasher::new(&fp);
        k.update(&[1, 2]);
        let first = k.key();
        k.update(&[3]);
        let second = k.key();
        assert_eq!(first, key(&BASE, &[1, 2]));
        assert_eq!(second, key(&BASE, &[1, 2, 3]));
        // The same ids split differently are the same prefix.
        let mut other = KeyHasher::new(&fp);
        other.update(&[1]);
        other.update(&[2, 3]);
        assert_eq!(other.key(), second);
    }
}
