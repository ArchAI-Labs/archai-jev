//! The guard of the `metal` feature (spec 011, AC-10). Standard library only, because `build.rs`
//! includes this very file (`#[path]`) and the tests below run in the library.
//!
//! A feature of Cargo is a wish, not a proof: `--features metal` is accepted on every platform,
//! and `.cargo/config.toml` switches `GGML_METAL` off for every build of this crate (the wheels
//! on PyPI are CPU-only, and a Metal build that finds no GPU hung for minutes on a CI machine).
//! Without this guard a build with `--features metal` would succeed, `compiled_backends()` would
//! say `metal`, and the binary would contain no Metal at all: the worst kind of fault here,
//! because nothing fails until a user asks for the GPU.

/// Values CMake reads as "on".
fn is_on(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_uppercase().as_str(),
        "ON" | "1" | "TRUE" | "YES" | "Y"
    )
}

/// Whether the build may go on: `metal_feature` says the feature is enabled, `target_os` is
/// `CARGO_CFG_TARGET_OS` and `ggml_metal` the environment variable `GGML_METAL`.
///
/// # Errors
/// The message to show when the build must stop.
pub fn check(metal_feature: bool, target_os: &str, ggml_metal: Option<&str>) -> Result<(), String> {
    if !metal_feature {
        return Ok(());
    }
    if target_os != "macos" {
        return Err(format!(
            "the 'metal' feature builds Apple's Metal backend and needs a macOS target, this build is for '{target_os}'; use 'cuda' or 'vulkan' elsewhere"
        ));
    }
    match ggml_metal {
        Some(value) if is_on(value) => Ok(()),
        other => Err(format!(
            "the 'metal' feature needs GGML_METAL=ON in the environment (it is {}): .cargo/config.toml switches Metal off for every build so that the wheels on PyPI stay CPU-only. Set it for this build: `export GGML_METAL=ON` (or `$env:GGML_METAL = 'ON'` in PowerShell)",
            other.map_or_else(|| "not set".to_string(), |v| format!("'{v}'"))
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_metal_feature_nothing_to_check() {
        for os in ["windows", "linux", "macos"] {
            for env in [None, Some("OFF"), Some("ON")] {
                assert_eq!(check(false, os, env), Ok(()), "{os} {env:?}");
            }
        }
    }

    #[test]
    fn the_metal_feature_needs_macos() {
        for os in ["windows", "linux", "android", "ios"] {
            let err = check(true, os, Some("ON")).unwrap_err();
            assert!(
                err.contains("needs a macOS target") && err.contains(os),
                "{err}"
            );
        }
    }

    #[test]
    fn the_metal_feature_needs_ggml_metal_on() {
        // .cargo/config.toml puts OFF in the environment of every build: this is the trap
        for env in [None, Some("OFF"), Some("0"), Some(""), Some("no")] {
            let err = check(true, "macos", env).unwrap_err();
            assert!(err.contains("GGML_METAL=ON"), "{env:?}: {err}");
        }
        assert!(
            check(true, "macos", Some("OFF"))
                .unwrap_err()
                .contains("'OFF'")
        );
        assert!(check(true, "macos", None).unwrap_err().contains("not set"));
        for env in ["ON", "on", "1", "TRUE", "yes", " ON "] {
            assert_eq!(check(true, "macos", Some(env)), Ok(()), "{env:?}");
        }
    }
}
