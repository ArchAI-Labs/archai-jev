//! The golden of the default model stays small, redistributable and untouched (spec 006c, AC on
//! the fixture): no weights, no tokenizer, provenance, notice and checksums that match.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use _core::json_strict::{Json, parse};
use _core::models::hash::sha256_hex;
use _core::models::testing::golden::default_data_dir as dir;

#[test]
fn size_checksums_and_no_weights() {
    let dir = dir();
    let prov = parse(&std::fs::read(dir.join("PROVENANCE.json")).unwrap()).unwrap();
    let sums = prov.get("file_sha256").expect("file_sha256");
    let bytes = std::fs::read(dir.join("golden.jsonl")).unwrap();
    let want = sums
        .get("golden.jsonl")
        .and_then(|e| e.get("sha256"))
        .expect("golden.jsonl");
    assert_eq!(
        want,
        &Json::Str(sha256_hex(&bytes)),
        "golden.jsonl was changed: regenerate it with the script of the fixture and update PROVENANCE.json"
    );
    // The whole folder, not only the data file, is under the limit of the spec.
    let mut total = 0u64;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(
            !name.ends_with(".gguf") && !name.ends_with(".safetensors") && name != "tokenizer.json",
            "{name} must not be in the repository"
        );
        total += entry.metadata().unwrap().len();
    }
    assert!(
        total <= 256 * 1024,
        "the fixture weighs {total} bytes, the limit is 256 KiB"
    );
}

#[test]
fn provenance_and_notice_carry_the_attribution() {
    let dir = dir();
    let prov = std::fs::read_to_string(dir.join("PROVENANCE.json")).unwrap();
    let notice = std::fs::read_to_string(dir.join("NOTICE")).unwrap();
    // The revisions the numbers come from are named in both.
    for text in [&prov, &notice] {
        assert!(
            text.contains("e4f3964abbc746f164f6cd009104e5213decba80"),
            "model revision"
        );
        assert!(
            text.contains("989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
            "base revision"
        );
    }
    assert!(
        notice.contains("Apache License, Version 2.0"),
        "licence missing"
    );
    assert!(
        notice.contains("written by the authors of archai-jev"),
        "inputs are ours"
    );
    assert!(
        notice.contains("outputs of the model"),
        "outputs are the model's"
    );
}
