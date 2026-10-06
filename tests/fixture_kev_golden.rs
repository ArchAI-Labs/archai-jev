//! The Kev golden stays small, redistributable and untouched (spec 006a, AC on the fixture).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use _core::json_strict::{Json, parse};
use _core::models::hash::sha256_hex;
use _core::models::testing::golden::data_dir;

const FILES: [&str; 3] = ["golden.jsonl", "golden_extra.jsonl", "selfcheck.jsonl"];

#[test]
fn size_checksums_and_dropped_fields() {
    let dir = data_dir();
    let prov = parse(&std::fs::read(dir.join("PROVENANCE.json")).unwrap()).unwrap();
    let sums = prov.get("file_sha256").expect("file_sha256");
    let mut total = 0usize;
    for name in FILES {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        total += bytes.len();
        let want = sums.get(name).and_then(|e| e.get("sha256")).expect(name);
        assert_eq!(
            want,
            &Json::Str(sha256_hex(&bytes)),
            "{name} was changed: regenerate it with the script of the fixture and update PROVENANCE.json"
        );
        let text = String::from_utf8(bytes).unwrap();
        for dropped in [
            "prompt_text",
            "xcheck",
            "answer",
            "state_text",
            "confidence_legacy",
        ] {
            assert!(
                !text.contains(&format!("\"{dropped}\"")),
                "{name} still has the dropped field {dropped}"
            );
        }
    }
    assert!(
        total <= 512 * 1024,
        "the fixture weighs {total} bytes, the limit is 512 KiB"
    );
    // No weights and no tokenizer next to the data.
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.ends_with(".gguf") && !name.ends_with(".safetensors") && name != "tokenizer.json",
            "{name} must not be in the repository"
        );
    }
}

#[test]
fn readme_and_notice_carry_the_attribution() {
    let dir = data_dir();
    let readme = std::fs::read_to_string(dir.join("README.md")).unwrap();
    let notice = std::fs::read_to_string(dir.join("NOTICE")).unwrap();
    for text in [&readme, &notice] {
        assert!(text.contains("Apache"), "licence missing");
        assert!(text.contains("9a45d25e"), "Kev revision missing");
    }
    assert!(notice.contains("dc7cdfe2"), "base revision missing");
    assert!(readme.contains("Inputs") && readme.contains("Outputs"));
}
