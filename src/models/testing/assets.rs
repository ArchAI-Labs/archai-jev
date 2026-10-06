//! Large files the parity tests need but the repository cannot hold (spec 006a, level T1).
//!
//! The rule against false greens: when `ARCHAI_JEV_REQUIRE_ASSETS=1` (CI) a missing asset **fails**
//! the test; without it the test says what is missing and is skipped. A file with the wrong
//! SHA-256 always fails: a wrong tokenizer would give plausible, wrong ids.

use std::path::{Path, PathBuf};

use crate::models::hash::sha256_file;

/// SHA-256 of the `tokenizer.json` of `jaredpalmer/kev-0.8b` @ `9a45d25e` (19,1 MiB).
pub const KEV_TOKENIZER_SHA256: &str =
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";

/// Where to find the Kev tokenizer and how to get it (for the message).
pub const KEV_TOKENIZER_HELP: &str = "set ARCHAI_JEV_TEST_TOKENIZER to the tokenizer.json of \
jaredpalmer/kev-0.8b @ 9a45d25eb2ab761841196625383fa1dff0e56c1e";

/// Decide what to do with an asset.
///
/// `given` is the value of the environment variable that points at it.
///
/// # Errors
/// A message when the asset is required and missing, or present with the wrong hash.
pub fn resolve(
    given: Option<&str>,
    expected_sha256: &str,
    name: &str,
    help: &str,
    require: bool,
) -> Result<Option<PathBuf>, String> {
    let Some(path) = given.filter(|p| !p.is_empty()) else {
        let msg = format!("asset {name} is missing: {help}");
        return if require { Err(msg) } else { Ok(None) };
    };
    let path = Path::new(path);
    if !path.is_file() {
        let msg = format!("asset {name} is not a file: {}", path.display());
        return if require { Err(msg) } else { Ok(None) };
    }
    let (sha, _) = sha256_file(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if sha != expected_sha256 {
        return Err(format!(
            "asset {name} at {} has sha256 {sha}, expected {expected_sha256}",
            path.display()
        ));
    }
    Ok(Some(path.to_path_buf()))
}

/// SHA-256 of the `tokenizer.json` of `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798` (the base of the
/// default model; the adapter's copy has truncation and padding baked in and must not be used).
pub const DEFAULT_TOKENIZER_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";

/// How to get the default model's tokenizer.
pub const DEFAULT_TOKENIZER_HELP: &str = "set ARCHAI_JEV_TEST_DEFAULT_TOKENIZER to the tokenizer.json of Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306";

/// Whether CI wants missing assets to fail.
pub fn require_assets() -> bool {
    std::env::var("ARCHAI_JEV_REQUIRE_ASSETS").is_ok_and(|v| v == "1")
}

/// The Kev tokenizer, or `None` when it is missing and not required (the caller prints `SKIPPED`).
///
/// # Panics
/// When the asset is required and missing, or has the wrong hash: that is the test failing.
pub fn kev_tokenizer() -> Option<PathBuf> {
    let given = std::env::var("ARCHAI_JEV_TEST_TOKENIZER").ok();
    match resolve(
        given.as_deref(),
        KEV_TOKENIZER_SHA256,
        "kev tokenizer.json",
        KEV_TOKENIZER_HELP,
        require_assets(),
    ) {
        Ok(Some(p)) => Some(p),
        Ok(None) => {
            eprintln!("SKIPPED: kev tokenizer.json is missing: {KEV_TOKENIZER_HELP}");
            None
        }
        Err(msg) => panic!("{msg}"),
    }
}

/// The tokenizer of the default model, or `None` when missing and not required.
///
/// # Panics
/// When the asset is required and missing, or has the wrong hash.
pub fn default_tokenizer() -> Option<PathBuf> {
    let given = std::env::var("ARCHAI_JEV_TEST_DEFAULT_TOKENIZER").ok();
    match resolve(
        given.as_deref(),
        DEFAULT_TOKENIZER_SHA256,
        "default model tokenizer.json",
        DEFAULT_TOKENIZER_HELP,
        require_assets(),
    ) {
        Ok(Some(p)) => Some(p),
        Ok(None) => {
            eprintln!("SKIPPED: default model tokenizer.json is missing: {DEFAULT_TOKENIZER_HELP}");
            None
        }
        Err(msg) => panic!("{msg}"),
    }
}

/// The GGUF of the default model (1.65 GB), for the manual engine tests (level T2).
///
/// # Panics
/// Never: a missing file is `None` and the test prints what is missing.
pub fn default_gguf() -> Option<PathBuf> {
    let given = std::env::var("ARCHAI_JEV_TEST_DEFAULT_GGUF").ok()?;
    let p = PathBuf::from(given);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("SKIPPED: ARCHAI_JEV_TEST_DEFAULT_GGUF does not point at a file");
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(bytes: &[u8]) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("asset.bin");
        std::fs::write(&p, bytes).unwrap();
        let s = p.to_string_lossy().into_owned();
        (dir, s)
    }

    #[test]
    fn missing_fails_when_required() {
        assert!(
            resolve(None, "x", "a", "help", true)
                .unwrap_err()
                .contains("help")
        );
        assert!(resolve(Some(""), "x", "a", "help", true).is_err());
        assert!(resolve(Some("/no/such/file"), "x", "a", "help", true).is_err());
    }

    #[test]
    fn missing_skips_otherwise() {
        assert_eq!(resolve(None, "x", "a", "help", false), Ok(None));
        assert_eq!(
            resolve(Some("/no/such/file"), "x", "a", "help", false),
            Ok(None)
        );
    }

    #[test]
    fn a_wrong_hash_always_fails() {
        let (_dir, path) = temp_file(b"not the tokenizer");
        for require in [true, false] {
            let err =
                resolve(Some(&path), "0".repeat(64).as_str(), "a", "help", require).unwrap_err();
            assert!(err.contains("expected"), "{err}");
        }
    }

    #[test]
    fn the_right_hash_is_accepted() {
        let (_dir, path) = temp_file(b"abc");
        // sha256("abc")
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(
            resolve(Some(&path), sha, "a", "help", true)
                .unwrap()
                .is_some()
        );
    }
}
