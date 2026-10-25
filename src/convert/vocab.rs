//! The vocabulary a Qwen3.5 GGUF carries (spec 018, 3.3).
//!
//! llama.cpp refuses a model without a vocabulary, but we never use it to tokenize (decision
//! D24: ids come from our own tokenizer). Still the GGUF must be a standard one, so we rebuild
//! the arrays the official script writes, **from the base model's** `tokenizer.json` and
//! `tokenizer_config.json`: tokens by id (with `[PAD<i>]` filling the gaps up to `vocab_size`),
//! their types, the BPE merges, and the ids of the end and padding tokens. The rules were
//! checked against the GGUF of the official script: the arrays are identical.

use std::path::Path;

use serde_json::Value;

use super::safetensors::io_error;
use crate::error::{Error, Result};
use crate::models::incompat::Incompat;

const TOKENIZER: &str = "tokenizer.json of the base model";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json of the base model";
const MAX_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;

/// Token types of the GGUF format.
pub mod token_type {
    /// A normal token.
    pub const NORMAL: i32 = 1;
    /// A control token (special).
    pub const CONTROL: i32 = 3;
    /// A user-defined token (added, not special).
    pub const USER_DEFINED: i32 = 4;
    /// A slot that no token uses.
    pub const UNUSED: i32 = 5;
}

/// The vocabulary arrays and special ids of the GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vocab {
    /// The token text of every id `0..vocab_size`.
    pub tokens: Vec<String>,
    /// The type of every token.
    pub types: Vec<i32>,
    /// The BPE merges, each `"left right"`.
    pub merges: Vec<String>,
    /// Id of the end-of-sequence token.
    pub eos: u32,
    /// Id of the padding token.
    pub pad: u32,
    /// Whether a tokenizer adds a BOS token (always `false` here).
    pub add_bos: bool,
    /// Whether a tokenizer adds an EOS token (always `false` here).
    pub add_eos: bool,
}

fn invalid(file: &str, detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::SourceFile {
        file: file.to_string(),
        detail: detail.into(),
    })
}

fn unsupported(what: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::SourceUnsupported {
        what: what.into(),
        detail: detail.into(),
    })
}

fn looks_special(t: &str) -> bool {
    (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<｜") && t.ends_with("｜>"))
        || (t.starts_with("<unused") && t.ends_with('>'))
        || matches!(t, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
}

fn read_json(path: &Path, file: &str) -> Result<Value> {
    let len = std::fs::metadata(path)
        .map_err(|e| io_error(path, &e))?
        .len();
    if len > MAX_TOKENIZER_BYTES {
        return Err(invalid(
            file,
            format!("it is {len} bytes; the maximum is {MAX_TOKENIZER_BYTES}"),
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| io_error(path, &e))?;
    serde_json::from_slice(&bytes).map_err(|e| invalid(file, format!("not valid JSON: {e}")))
}

/// One added token: `(id, content, special)`.
type Added = (u64, String, bool);

fn added_of_tokenizer(j: &Value) -> Result<Vec<Added>> {
    let Some(list) = j.get("added_tokens").and_then(Value::as_array) else {
        return Err(invalid(
            TOKENIZER,
            "'added_tokens' is missing or not a list",
        ));
    };
    list.iter()
        .enumerate()
        .map(|(i, t)| {
            let id = t.get("id").and_then(Value::as_u64);
            let content = t.get("content").and_then(Value::as_str);
            let special = t.get("special").and_then(Value::as_bool);
            match (id, content, special) {
                (Some(id), Some(c), Some(sp)) => Ok((id, c.to_string(), sp)),
                _ => Err(invalid(
                    TOKENIZER,
                    format!("added_tokens[{i}] needs an integer id, a string content and a boolean special"),
                )),
            }
        })
        .collect()
}

fn added_of_config(j: &Value) -> Result<Vec<Added>> {
    let Some(map) = j.get("added_tokens_decoder").and_then(Value::as_object) else {
        return Err(invalid(
            TOKENIZER_CONFIG,
            "'added_tokens_decoder' is missing or not an object",
        ));
    };
    map.iter()
        .map(|(k, t)| {
            let id = k.parse::<u64>().ok();
            let content = t.get("content").and_then(Value::as_str);
            let special = t.get("special").and_then(Value::as_bool);
            match (id, content, special) {
                (Some(id), Some(c), Some(sp)) => Ok((id, c.to_string(), sp)),
                _ => Err(invalid(
                    TOKENIZER_CONFIG,
                    format!("added_tokens_decoder[{k:?}] needs a numeric key, a string content and a boolean special"),
                )),
            }
        })
        .collect()
}

fn merges_of(j: &Value) -> Result<Vec<String>> {
    let Some(list) = j
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(Value::as_array)
    else {
        return Err(invalid(
            TOKENIZER,
            "'model.merges' is missing or not a list",
        ));
    };
    list.iter()
        .enumerate()
        .map(|(i, m)| match m {
            Value::String(s) if s.split(' ').count() == 2 => Ok(s.clone()),
            Value::Array(pair) => match pair.as_slice() {
                [Value::String(a), Value::String(b)] => {
                    let enc = |x: &str| x.replace(' ', "\u{120}");
                    Ok(format!("{} {}", enc(a), enc(b)))
                }
                _ => Err(invalid(
                    TOKENIZER,
                    format!("merges[{i}] is not a pair of strings"),
                )),
            },
            _ => Err(invalid(
                TOKENIZER,
                format!("merges[{i}] is not \"left right\""),
            )),
        })
        .collect()
}

fn id_of(tokens: &[String], types: &[i32], text: &str, role: &str) -> Result<u32> {
    let mut found = tokens
        .iter()
        .enumerate()
        .filter(|(i, t)| t.as_str() == text && types.get(*i) != Some(&token_type::UNUSED));
    match (found.next(), found.next()) {
        (Some((i, _)), None) => u32::try_from(i)
            .map_err(|_| invalid(TOKENIZER_CONFIG, format!("the id of {role} is too large"))),
        (None, _) => Err(invalid(
            TOKENIZER_CONFIG,
            format!("the {role} token {text:?} is not in the vocabulary"),
        )),
        (Some(_), Some(_)) => Err(invalid(
            TOKENIZER_CONFIG,
            format!("the {role} token {text:?} appears twice in the vocabulary"),
        )),
    }
}

impl Vocab {
    /// Build the vocabulary of `vocab_size` rows from the two files of the base model.
    ///
    /// # Errors
    /// `IncompatibleModelError` for ids that are not contiguous, duplicated or out of range,
    /// added tokens that disagree between the two files, malformed merges, special tokens that
    /// are missing, and for tokenizer settings the converter does not reproduce.
    pub fn read(tokenizer: &Path, tokenizer_config: &Path, vocab_size: u64) -> Result<Vocab> {
        let tj = read_json(tokenizer, TOKENIZER)?;
        let cj = read_json(tokenizer_config, TOKENIZER_CONFIG)?;
        Vocab::build(&tj, &cj, vocab_size)
    }

    /// Like [`Vocab::read`] from parsed JSON.
    ///
    /// # Errors
    /// See [`Vocab::read`].
    pub fn build(tj: &Value, cj: &Value, vocab_size: u64) -> Result<Vocab> {
        let n = usize::try_from(vocab_size)
            .ok()
            .filter(|n| (1..=4_000_000).contains(n))
            .ok_or_else(|| invalid(TOKENIZER, "vocab_size is out of range"))?;
        let model = tj.get("model");
        if model.and_then(|m| m.get("type")).and_then(Value::as_str) != Some("BPE") {
            return Err(invalid(TOKENIZER, "model.type is not \"BPE\""));
        }
        let Some(vocab) = model
            .and_then(|m| m.get("vocab"))
            .and_then(Value::as_object)
        else {
            return Err(invalid(
                TOKENIZER,
                "'model.vocab' is missing or not an object",
            ));
        };
        let mut slots: Vec<Option<String>> = vec![None; n];
        for (token, id) in vocab {
            let Some(id) = id.as_u64().and_then(|i| usize::try_from(i).ok()) else {
                return Err(invalid(
                    TOKENIZER,
                    format!("the id of token {token:?} is not an integer"),
                ));
            };
            let Some(slot) = slots.get_mut(id) else {
                return Err(invalid(
                    TOKENIZER,
                    format!("token {token:?} has id {id}, outside vocab_size {vocab_size}"),
                ));
            };
            if slot.replace(token.clone()).is_some() {
                return Err(invalid(TOKENIZER, format!("two tokens have the id {id}")));
            }
        }
        let base_tokens = vocab.len();
        if slots.iter().take(base_tokens).any(Option::is_none) {
            return Err(invalid(
                TOKENIZER,
                "the ids of model.vocab are not 0..N without holes",
            ));
        }

        // Added tokens: from both files; where both name an id they must agree.
        let mut added: Vec<Added> = added_of_tokenizer(tj)?;
        for (id, content, special) in added_of_config(cj)? {
            match added.iter().find(|(i, _, _)| *i == id) {
                None => added.push((id, content, special)),
                Some((_, c, s)) if *c == content && *s == special => {}
                Some((_, c, s)) => {
                    return Err(invalid(
                        TOKENIZER_CONFIG,
                        format!(
                            "added token {id} is {content:?} (special: {special}) here but {c:?} (special: {s}) in tokenizer.json"
                        ),
                    ));
                }
            }
        }
        let mut types = vec![token_type::NORMAL; n];
        for (id, content, special) in &added {
            let Some(idx) = usize::try_from(*id).ok().filter(|i| *i < n) else {
                return Err(invalid(
                    TOKENIZER,
                    format!("added token {content:?} has id {id}, outside vocab_size {vocab_size}"),
                ));
            };
            if let Some(slot) = slots.get_mut(idx) {
                match slot {
                    Some(existing) if existing != content => {
                        return Err(invalid(
                            TOKENIZER,
                            format!(
                                "added token {id} is {content:?} but model.vocab has {existing:?} at that id"
                            ),
                        ));
                    }
                    _ => *slot = Some(content.clone()),
                }
            }
            if let Some(t) = types.get_mut(idx) {
                *t = if *special || looks_special(content) {
                    token_type::CONTROL
                } else {
                    token_type::USER_DEFINED
                };
            }
        }
        let mut tokens = Vec::with_capacity(n);
        for (i, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(t) => tokens.push(t),
                None => {
                    tokens.push(format!("[PAD{i}]"));
                    if let Some(t) = types.get_mut(i) {
                        *t = token_type::UNUSED;
                    }
                }
            }
        }

        // What a tokenizer would add and which tokens are special: only what we reproduce.
        for key in [
            "bos_token",
            "unk_token",
            "cls_token",
            "sep_token",
            "mask_token",
        ] {
            if !matches!(cj.get(key), None | Some(Value::Null)) {
                return Err(unsupported(
                    format!("a '{key}' in tokenizer_config.json"),
                    "the GGUF would need its id, and the converter writes only eos and padding",
                ));
            }
        }
        let post = tj
            .get("post_processor")
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str);
        if post != Some("ByteLevel") {
            return Err(unsupported(
                format!("a post_processor of type {post:?}"),
                "only the ByteLevel post-processor (no BOS, no EOS added) is reproduced",
            ));
        }
        if cj.get("add_bos_token") == Some(&Value::Bool(true)) {
            return Err(unsupported("add_bos_token", "it is true"));
        }
        let special_text = |key: &str| -> Result<&str> {
            cj.get(key).and_then(Value::as_str).ok_or_else(|| {
                invalid(
                    TOKENIZER_CONFIG,
                    format!("'{key}' is missing or not a string"),
                )
            })
        };
        let eos = id_of(&tokens, &types, special_text("eos_token")?, "eos")?;
        let pad = id_of(&tokens, &types, special_text("pad_token")?, "padding")?;

        Ok(Vocab {
            tokens,
            types,
            merges: merges_of(tj)?,
            eos,
            pad,
            add_bos: false,
            add_eos: false,
        })
    }
}
