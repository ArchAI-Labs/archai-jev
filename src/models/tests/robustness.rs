//! Hostile input must produce an error or a success, never a panic (spec 005, AC on robustness).

use crate::json_strict;
use crate::models::gguf;
use crate::models::manifest::{Manifest, ManifestOrigin};
use crate::models::testing::builder::Builder;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap()
    }
}

#[test]
fn gguf_header_mutations_never_panic() {
    let fixture = Builder::tiny().build();
    let original = std::fs::read(fixture.gguf_path()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.gguf");
    let header_len = 2048.min(original.len());
    let mut rng = Rng(0xA11CE);
    let started = std::time::Instant::now();
    let mut accepted = 0;
    for case in 0..6000 {
        let mut bytes = original.clone();
        match case % 4 {
            0 => {
                // flip a few random bytes inside the header
                for _ in 0..=rng.below(4) {
                    let i = rng.below(header_len);
                    bytes[i] ^= u8::try_from(1 + rng.below(255)).unwrap();
                }
            }
            1 => bytes.truncate(rng.below(original.len())),
            2 => {
                // overwrite eight bytes with a random value (counts, lengths, offsets)
                let i = rng.below(header_len - 8);
                bytes[i..i + 8].copy_from_slice(&rng.next().to_le_bytes());
            }
            _ => {
                // an enormous value in one of the counts or sizes
                let i = rng.below(header_len - 8);
                bytes[i..i + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        if gguf::read_header(&path).is_ok() {
            accepted += 1;
        }
        assert!(
            started.elapsed().as_secs() < 120,
            "too slow: a case may be hanging"
        );
    }
    assert!(accepted < 6000, "some mutations must be rejected");
}

#[test]
fn manifests_cut_at_every_length_never_panic() {
    let fixture = Builder::tiny().build();
    let text = std::fs::read(fixture.manifest_path()).unwrap();
    for len in 0..text.len() {
        let _ = Manifest::from_bytes(&text[..len], ManifestOrigin::Local);
    }
    assert!(Manifest::from_bytes(&text, ManifestOrigin::Local).is_ok());
}

#[test]
fn manifests_with_random_byte_changes_never_panic() {
    let fixture = Builder::tiny().build();
    let text = std::fs::read(fixture.manifest_path()).unwrap();
    let mut rng = Rng(0xBEEF);
    for _ in 0..4000 {
        let mut bytes = text.clone();
        for _ in 0..=rng.below(3) {
            let i = rng.below(bytes.len());
            bytes[i] = u8::try_from(rng.below(256)).unwrap();
        }
        if let Ok(m) = Manifest::from_bytes(&bytes, ManifestOrigin::Local) {
            let _ = crate::models::validate::semantics(&m, &[]);
        }
    }
}

#[test]
fn json_of_arbitrary_bytes_never_panics() {
    let mut rng = Rng(7);
    for _ in 0..3000 {
        let n = rng.below(200);
        let bytes: Vec<u8> = (0..n)
            .map(|_| u8::try_from(rng.below(256)).unwrap())
            .collect();
        let _ = json_strict::parse(&bytes);
    }
}
