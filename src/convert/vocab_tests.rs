//! Tests of the vocabulary: a tiny tokenizer for every rule, the real one against the digests of
//! the official script's GGUF.

use serde_json::{Value, json};

use super::vocab::{Vocab, token_type};
use crate::error::Error;
use crate::models::hash::sha256_hex;
use crate::models::incompat::Incompat;

/// A tokenizer.json with 4 BPE tokens (ids 0-3), 2 added ones, and a ByteLevel post-processor.
fn tokenizer() -> Value {
    json!({
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2, "\u{120}c": 3},
            "merges": ["a b", "\u{120} c"]
        },
        "added_tokens": [
            {"id": 4, "content": "<|endoftext|>", "special": true},
            {"id": 5, "content": "<think>", "special": false}
        ],
        "post_processor": {"type": "ByteLevel"}
    })
}

/// A tokenizer_config.json that adds one more token (id 6) and names eos and padding.
fn config() -> Value {
    json!({
        "added_tokens_decoder": {
            "4": {"content": "<|endoftext|>", "special": true},
            "6": {"content": "<|im_start|>", "special": false}
        },
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": null,
        "add_bos_token": false
    })
}

fn build(t: &Value, c: &Value, size: u64) -> Result<Vocab, Incompat> {
    Vocab::build(t, c, size).map_err(|e| match e {
        Error::IncompatibleModel(i) => i,
        other => panic!("not a refusal: {other:?}"),
    })
}

fn refusal(t: &Value, c: &Value, size: u64, fragment: &str) {
    let Err(i) = build(t, c, size) else {
        panic!("accepted, but {fragment:?} was expected")
    };
    let text = i.to_string();
    assert!(text.contains(fragment), "{text:?} lacks {fragment:?}");
}

#[test]
fn tokens_types_merges_and_special_ids_follow_the_rules() {
    let v = build(&tokenizer(), &config(), 9).unwrap();
    assert_eq!(
        v.tokens,
        [
            "a",
            "b",
            "ab",
            "\u{120}c",
            "<|endoftext|>",
            "<think>",
            "<|im_start|>",
            "[PAD7]",
            "[PAD8]"
        ]
    );
    assert_eq!(
        v.types,
        [
            token_type::NORMAL,
            token_type::NORMAL,
            token_type::NORMAL,
            token_type::NORMAL,
            token_type::CONTROL,      // special
            token_type::USER_DEFINED, // added, not special, not shaped like <|...|>
            token_type::CONTROL,      // not special but shaped like <|...|>
            token_type::UNUSED,
            token_type::UNUSED
        ]
    );
    assert_eq!(v.merges, ["a b", "\u{120} c"]);
    assert_eq!((v.eos, v.pad, v.add_bos, v.add_eos), (4, 4, false, false));
}

#[test]
fn merges_given_as_pairs_are_written_as_strings_with_encoded_spaces() {
    let mut t = tokenizer();
    t["model"]["merges"] = json!([["a", "b"], ["x y", "z"]]);
    let v = build(&t, &config(), 9).unwrap();
    assert_eq!(v.merges, ["a b", "x\u{120}y z"]);
}

#[test]
fn ids_must_be_contiguous_unique_and_inside_the_vocabulary() {
    let mut t = tokenizer();
    t["model"]["vocab"] = json!({"a": 0, "b": 1, "ab": 3});
    refusal(&t, &config(), 9, "without holes");
    let mut t = tokenizer();
    t["model"]["vocab"] = json!({"a": 0, "b": 0});
    refusal(&t, &config(), 9, "two tokens have the id 0");
    refusal(&tokenizer(), &config(), 3, "outside vocab_size");
    let mut t = tokenizer();
    t["model"]["vocab"] = json!({"a": "zero"});
    refusal(&t, &config(), 9, "not an integer");
    refusal(&tokenizer(), &config(), 0, "out of range");
}

#[test]
fn the_two_files_must_agree_about_added_tokens() {
    let mut c = config();
    c["added_tokens_decoder"]["4"] = json!({"content": "<|other|>", "special": true});
    refusal(&tokenizer(), &c, 9, "tokenizer.json");
    let mut c = config();
    c["added_tokens_decoder"]["4"] = json!({"content": "<|endoftext|>", "special": false});
    refusal(&tokenizer(), &c, 9, "special");
    // an added token that would overwrite a different base token
    let mut t = tokenizer();
    t["added_tokens"][1] = json!({"id": 2, "content": "zzz", "special": false});
    refusal(&t, &config(), 9, "model.vocab has");
    // an added token outside the vocabulary
    let mut c = config();
    c["added_tokens_decoder"]["60"] = json!({"content": "<|far|>", "special": true});
    refusal(&tokenizer(), &c, 9, "outside vocab_size");
    // malformed entries
    let mut t = tokenizer();
    t["added_tokens"][0] = json!({"id": "four"});
    refusal(&t, &config(), 9, "added_tokens[0]");
    let mut c = config();
    c["added_tokens_decoder"]["x"] = json!({"content": "<|x|>", "special": true});
    refusal(&tokenizer(), &c, 9, "numeric key");
}

#[test]
fn malformed_files_and_merges_are_refused() {
    let mut t = tokenizer();
    t["model"]["type"] = json!("WordPiece");
    refusal(&t, &config(), 9, "BPE");
    let mut t = tokenizer();
    t["model"]["merges"] = json!(["only-one-part"]);
    refusal(&t, &config(), 9, "merges[0]");
    let mut t = tokenizer();
    t["model"]["merges"] = json!([[1, 2]]);
    refusal(&t, &config(), 9, "pair of strings");
    let mut t = tokenizer();
    t["model"].as_object_mut().unwrap().remove("merges");
    refusal(&t, &config(), 9, "merges");
    let mut t = tokenizer();
    t.as_object_mut().unwrap().remove("added_tokens");
    refusal(&t, &config(), 9, "added_tokens");
    let mut c = config();
    c.as_object_mut().unwrap().remove("added_tokens_decoder");
    refusal(&tokenizer(), &c, 9, "added_tokens_decoder");
}

#[test]
fn settings_the_converter_does_not_reproduce_are_refused() {
    for key in [
        "bos_token",
        "unk_token",
        "cls_token",
        "sep_token",
        "mask_token",
    ] {
        let mut c = config();
        c[key] = json!("<|x|>");
        refusal(&tokenizer(), &c, 9, key);
    }
    let mut t = tokenizer();
    t["post_processor"] = json!({"type": "TemplateProcessing"});
    refusal(&t, &config(), 9, "post_processor");
    let mut t = tokenizer();
    t["post_processor"] = Value::Null;
    refusal(&t, &config(), 9, "post_processor");
    let mut c = config();
    c["add_bos_token"] = json!(true);
    refusal(&tokenizer(), &c, 9, "add_bos_token");
    // eos and padding must name tokens of the vocabulary
    let mut c = config();
    c["eos_token"] = json!("<|nope|>");
    refusal(&tokenizer(), &c, 9, "not in the vocabulary");
    let mut c = config();
    c["pad_token"] = json!("[PAD7]");
    refusal(&tokenizer(), &c, 9, "not in the vocabulary");
    let mut c = config();
    c.as_object_mut().unwrap().remove("pad_token");
    refusal(&tokenizer(), &c, 9, "pad_token");
}

/// Digests of the arrays of the GGUF the official script wrote for Kev-0.8B (spike S4):
/// SHA-256 over each token followed by a NUL, over each type as a little-endian i32, and over
/// each merge followed by a NUL.
const TOKENS_SHA: &str = "49930cc8a75188cf69a07c57914f0aabe383db11494aa03ed15629adee4c7e33";
const TYPES_SHA: &str = "d54439ef48197cc1a23022aa95794bfd0c5094988f43ed1eac6e5facc0a406d4";
const MERGES_SHA: &str = "2d0d7c583fcdbbccbebdfb69eb5bee91a3b8eb95cb8dc37f44ff645c484ae6e8";

#[test]
fn the_real_vocabulary_equals_the_one_of_the_official_script() {
    use crate::models::testing::assets::{kev_base_tokenizer, kev_base_tokenizer_config};
    let (Some(t), Some(c)) = (kev_base_tokenizer(), kev_base_tokenizer_config()) else {
        return;
    };
    let v = Vocab::read(&t, &c, 248_320).unwrap();
    assert_eq!(
        (v.tokens.len(), v.types.len(), v.merges.len()),
        (248_320, 248_320, 247_587)
    );
    let joined = |items: &[String]| {
        items
            .iter()
            .flat_map(|s| s.bytes().chain(std::iter::once(0u8)))
            .collect::<Vec<u8>>()
    };
    assert_eq!(sha256_hex(&joined(&v.tokens)), TOKENS_SHA);
    assert_eq!(
        sha256_hex(
            &v.types
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<u8>>()
        ),
        TYPES_SHA
    );
    assert_eq!(sha256_hex(&joined(&v.merges)), MERGES_SHA);
    assert_eq!((v.eos, v.pad), (248_044, 248_044));
    let count = |ty: i32| v.types.iter().filter(|t| **t == ty).count();
    assert_eq!(
        (
            count(token_type::NORMAL),
            count(token_type::CONTROL),
            count(token_type::USER_DEFINED),
            count(token_type::UNUSED)
        ),
        (248_044, 27, 6, 243)
    );
}

#[test]
fn a_config_without_its_added_tokens_gives_a_different_vocabulary() {
    // the 11 tokens that only tokenizer_config.json names are not optional
    use crate::models::testing::assets::{kev_base_tokenizer, kev_base_tokenizer_config};
    let (Some(t), Some(c)) = (kev_base_tokenizer(), kev_base_tokenizer_config()) else {
        return;
    };
    let tj: Value = serde_json::from_slice(&std::fs::read(t).unwrap()).unwrap();
    let mut cj: Value = serde_json::from_slice(&std::fs::read(c).unwrap()).unwrap();
    let full = Vocab::build(&tj, &cj, 248_320).unwrap();
    let decoder = cj["added_tokens_decoder"].as_object_mut().unwrap();
    let only_here: Vec<String> = decoder
        .keys()
        .filter(|k| {
            tj["added_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .all(|a| a["id"].as_u64() != k.parse::<u64>().ok())
        })
        .cloned()
        .collect();
    assert_eq!(only_here.len(), 11);
    for k in only_here {
        decoder.remove(&k);
    }
    let reduced = Vocab::build(&tj, &cj, 248_320).unwrap();
    assert_ne!(full.tokens, reduced.tokens);
}
