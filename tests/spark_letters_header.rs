//! The tensor table and the metadata of the `spark2_5-letters` family against the header of a real
//! file (spec 010b, task "family", test level T2: by hand).
//!
//! Point `ARCHAI_JEV_TEST_SPARK_GGUF` at a GGUF of the family (run it once per size). The
//! parameters are read from the file itself; what is checked is that **every** tensor the table
//! expects is there with its shape and type, that the file has nothing else, and that every
//! metadata key the family compares equals what the file says. Without the variable the test
//! says what is missing and passes, unless `ARCHAI_JEV_REQUIRE_ASSETS=1`, which makes it fail.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;

use _core::models::families::{self, ArchParams, MetaExpect, ParamValue, Role};
use _core::models::gguf::{GgmlType, MetaValue, read_header};

fn real_gguf() -> Option<PathBuf> {
    match std::env::var("ARCHAI_JEV_TEST_SPARK_GGUF") {
        Ok(p) if PathBuf::from(&p).is_file() => Some(PathBuf::from(p)),
        _ => {
            let require = std::env::var("ARCHAI_JEV_REQUIRE_ASSETS").is_ok_and(|v| v == "1");
            assert!(
                !require,
                "ARCHAI_JEV_TEST_SPARK_GGUF must point at a GGUF of the family"
            );
            eprintln!("SKIPPED: set ARCHAI_JEV_TEST_SPARK_GGUF to a GGUF of the family");
            None
        }
    }
}

fn uint(info: &_core::models::gguf::GgufInfo, key: &str) -> u64 {
    match info.get(key) {
        Some(MetaValue::UInt(n)) => *n,
        other => panic!("{key} is {other:?}"),
    }
}

#[test]
#[ignore = "needs a GGUF of the family: ARCHAI_JEV_TEST_SPARK_GGUF"]
fn the_table_of_the_family_is_the_header_of_the_real_file() {
    let Some(path) = real_gguf() else { return };
    let info = read_header(&path).expect("a valid GGUF");
    let family = families::lookup("spark2_5-letters").expect("the family");

    let int = |k: &str, v: u64| (k.to_string(), ParamValue::Int(v));
    let params = ArchParams::from_pairs(vec![
        int("n_layers", uint(&info, "spark2_5.block_count")),
        int("n_embd", uint(&info, "spark2_5.embedding_length")),
        int("n_ff", uint(&info, "spark2_5.feed_forward_length")),
        int("n_heads", uint(&info, "spark2_5.attention.head_count")),
        int(
            "n_kv_heads",
            uint(&info, "spark2_5.attention.head_count_kv"),
        ),
        int("head_dim", uint(&info, "spark2_5.attention.key_length")),
        int("n_vocab", uint(&info, "spark2_5.vocab_size")),
        (
            "tie_embeddings".to_string(),
            ParamValue::Bool(info.tensor("output.weight").is_none()),
        ),
        int(
            "sliding_window",
            uint(&info, "spark2_5.attention.sliding_window"),
        ),
        int("full_attention_every", 4),
    ]);

    for (key, expect) in (family.metadata)(&params) {
        let got = info
            .get(&key)
            .unwrap_or_else(|| panic!("no {key} in the file"));
        match (&expect, got) {
            (MetaExpect::Str(s), MetaValue::Str(g)) => assert_eq!(s, g, "{key}"),
            (MetaExpect::UInt(n), MetaValue::UInt(g)) => assert_eq!(n, g, "{key}"),
            (MetaExpect::BoolArray(want), MetaValue::BoolArray(g)) => assert_eq!(want, g, "{key}"),
            (e, g) => panic!("{key}: expected {e:?}, the file has {g:?}"),
        }
    }

    let expected = (family.tensors)(&params);
    assert_eq!(
        expected.len(),
        info.tensors.len(),
        "the file has {} tensors, the table {}",
        info.tensors.len(),
        expected.len()
    );
    for spec in &expected {
        let t = info
            .tensor(&spec.name)
            .unwrap_or_else(|| panic!("{} is not in the file", spec.name));
        assert_eq!(t.dims, spec.dims, "{}", spec.name);
        let want = match spec.role {
            Role::Matrix => GgmlType::Q8_0,
            Role::Vector => GgmlType::F32,
        };
        assert_eq!(t.ty, want, "{}", spec.name);
    }
    assert_eq!(
        (family.snapshot_max_tokens)(&params),
        Some(uint(&info, "spark2_5.attention.sliding_window"))
    );
    eprintln!(
        "{}: {} layers, {} tensors, all as the table says",
        path.display(),
        uint(&info, "spark2_5.block_count"),
        info.tensors.len()
    );
}
