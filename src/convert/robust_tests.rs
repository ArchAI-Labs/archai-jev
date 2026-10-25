//! Random damage to the original files: the converter answers `Ok` or an `IncompatibleModelError`
//! (or an I/O error), and never panics or hangs (spec 018, AC-54). The `head.pt` reader has its
//! own test in `headpt/tests.rs`.

use std::path::{Path, PathBuf};

use super::oracle_tests::{BASE_REPO, fixture, params};
use super::qwen35::convert_files;
use crate::error::Error;
use crate::hub::events::{NeverCancel, NoObserver};
use crate::models::convert::SourceFiles;

/// SplitMix64, fixed seed: a deterministic stream of changes.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn files(dir: &Path) -> SourceFiles {
    SourceFiles {
        base_config: dir.join("base/config.json"),
        base_weights: dir.join("base/model.safetensors"),
        base_tokenizer: dir.join("base/tokenizer.json"),
        base_tokenizer_config: dir.join("base/tokenizer_config.json"),
        adapter_config: dir.join("adapter/adapter_config.json"),
        adapter_weights: dir.join("adapter/adapter_model.safetensors"),
    }
}

const TARGETS: [&str; 6] = [
    "base/config.json",
    "base/model.safetensors",
    "base/tokenizer.json",
    "base/tokenizer_config.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
];

/// How much of the start of a file the damage goes to: headers and configurations are where
/// the readers do their work; the data of a safetensors file only changes numbers.
fn reach(file: &str, len: usize) -> usize {
    if file.ends_with(".safetensors") {
        len.min(4096)
    } else {
        len
    }
}

#[test]
fn damaged_inputs_give_a_refusal_or_a_file_never_a_panic() {
    let pristine: Vec<(PathBuf, Vec<u8>)> = TARGETS
        .iter()
        .map(|f| (PathBuf::from(f), std::fs::read(fixture().join(f)).unwrap()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    for sub in ["base", "adapter"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
    }
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let (mut refused, mut converted) = (0usize, 0usize);
    for case in 0..10_000 {
        // restore everything, then damage one file
        for (rel, bytes) in &pristine {
            std::fs::write(dir.path().join(rel), bytes).unwrap();
        }
        let which = rng.below(TARGETS.len());
        let (rel, bytes) = &pristine[which];
        let mut damaged = bytes.clone();
        match case % 4 {
            0 => {
                // some bytes of the start of the file change
                for _ in 0..=rng.below(3) {
                    let at = rng.below(reach(TARGETS[which], damaged.len()));
                    if let Some(b) = damaged.get_mut(at) {
                        *b = rng.next() as u8;
                    }
                }
            }
            1 => damaged.truncate(rng.below(damaged.len())),
            2 => {
                // one byte flips one bit
                let at = rng.below(reach(TARGETS[which], damaged.len()));
                if let Some(b) = damaged.get_mut(at) {
                    *b ^= 1 << rng.below(8);
                }
            }
            _ => {
                // a stretch of the start is overwritten with a repeated byte
                let n = rng.below(16) + 1;
                let at = rng.below(reach(TARGETS[which], damaged.len()));
                let v = rng.next() as u8;
                for b in damaged.iter_mut().skip(at).take(n) {
                    *b = v;
                }
            }
        }
        std::fs::write(dir.path().join(rel), damaged).unwrap();
        let out = tempfile::tempdir().unwrap();
        let started = std::time::Instant::now();
        let r = convert_files(
            &files(dir.path()),
            &params(),
            BASE_REPO,
            if case % 2 == 0 { "bf16" } else { "f32" },
            out.path(),
            &NoObserver,
            &NeverCancel,
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "case {case} ({rel:?}) took too long"
        );
        match r {
            Ok(_) => converted += 1,
            Err(Error::IncompatibleModel(_) | Error::ModelDownload(_)) => refused += 1,
            Err(other) => panic!("case {case} ({rel:?}): unexpected error {other:?}"),
        }
        // a refused conversion leaves nothing behind
        if r_is_err(&out) {
            assert_eq!(
                std::fs::read_dir(out.path()).unwrap().count(),
                0,
                "case {case}"
            );
        }
    }
    // most damage is noticed; some only changes numbers inside a tensor, and that is fine
    assert!(
        refused > 1000,
        "only {refused} of 10000 damaged inputs were refused"
    );
    assert!(refused + converted == 10_000);
}

/// Whether the output folder holds a model (a successful conversion).
fn r_is_err(out: &tempfile::TempDir) -> bool {
    !out.path().join("model.gguf").exists()
}
