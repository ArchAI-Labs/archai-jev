//! Where the cache lives, per operating system, and its layout.
//!
//! Environment variables are read through [`CacheEnv`] so tests can inject them and cover all
//! three OS conventions on any machine.

use std::path::{Path, PathBuf};

use crate::models::failures::DownloadFailure;

/// Name of the folder inside the OS cache directory.
pub const APP_DIR: &str = "archai-jev";
/// Environment variable that overrides the cache root.
pub const CACHE_VARIABLE: &str = "ARCHAI_JEV_CACHE";
/// Version of the cache layout (`<root>/v1`).
pub const LAYOUT: &str = "v1";

/// Operating system conventions we know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    /// Linux and other Unix-like systems.
    Linux,
    /// macOS.
    MacOs,
    /// Windows.
    Windows,
}

impl Os {
    /// The system we are running on.
    pub fn current() -> Os {
        if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "macos") {
            Os::MacOs
        } else {
            Os::Linux
        }
    }
}

/// A source of environment variables.
pub trait CacheEnv {
    /// The value of `name`, if set.
    fn get(&self, name: &str) -> Option<String>;
}

fn non_empty(env: &dyn CacheEnv, name: &str) -> Option<String> {
    env.get(name).filter(|v| !v.is_empty())
}

/// The cache root: `cache_dir` if given, else `ARCHAI_JEV_CACHE`, else the OS default.
///
/// # Errors
/// [`DownloadFailure::NoCacheDir`] if nothing tells where the home or cache directory is.
pub fn resolve_root(
    cache_dir: Option<&Path>,
    env: &dyn CacheEnv,
    os: Os,
) -> Result<PathBuf, DownloadFailure> {
    if let Some(dir) = cache_dir {
        return Ok(dir.to_path_buf());
    }
    if let Some(dir) = non_empty(env, CACHE_VARIABLE) {
        return Ok(PathBuf::from(dir));
    }
    let root = match os {
        Os::Linux => non_empty(env, "XDG_CACHE_HOME")
            .filter(|p| p.starts_with('/') || Path::new(p).is_absolute())
            .map(|p| PathBuf::from(p).join(APP_DIR))
            .or_else(|| {
                non_empty(env, "HOME").map(|h| PathBuf::from(h).join(".cache").join(APP_DIR))
            }),
        Os::MacOs => non_empty(env, "HOME").map(|h| {
            PathBuf::from(h)
                .join("Library")
                .join("Caches")
                .join(APP_DIR)
        }),
        Os::Windows => {
            non_empty(env, "LOCALAPPDATA").map(|p| PathBuf::from(p).join(APP_DIR).join("Cache"))
        }
    };
    root.ok_or(DownloadFailure::NoCacheDir)
}

/// The three folders of the layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Content-addressed files: `blobs/<sha256>`.
    pub blobs: PathBuf,
    /// Verification records.
    pub verified: PathBuf,
    /// Results of conversions (018).
    pub materialized: PathBuf,
}

impl Layout {
    /// The layout under `root`.
    pub fn new(root: &Path) -> Layout {
        let base = root.join(LAYOUT);
        Layout {
            blobs: base.join("blobs"),
            verified: base.join("verified"),
            materialized: base.join("materialized"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Env(HashMap<&'static str, &'static str>);
    impl CacheEnv for Env {
        fn get(&self, name: &str) -> Option<String> {
            self.0.get(name).map(|v| (*v).to_string())
        }
    }

    fn env(pairs: &[(&'static str, &'static str)]) -> Env {
        Env(pairs.iter().copied().collect())
    }

    #[test]
    fn linux_defaults() {
        let with_xdg = env(&[("XDG_CACHE_HOME", "/xdg"), ("HOME", "/home/u")]);
        assert_eq!(
            resolve_root(None, &with_xdg, Os::Linux).unwrap(),
            PathBuf::from("/xdg/archai-jev")
        );
        let home_only = env(&[("HOME", "/home/u")]);
        assert_eq!(
            resolve_root(None, &home_only, Os::Linux).unwrap(),
            PathBuf::from("/home/u/.cache/archai-jev")
        );
        let relative_xdg = env(&[("XDG_CACHE_HOME", "rel"), ("HOME", "/home/u")]);
        assert_eq!(
            resolve_root(None, &relative_xdg, Os::Linux).unwrap(),
            PathBuf::from("/home/u/.cache/archai-jev")
        );
    }

    #[test]
    fn macos_and_windows_defaults() {
        assert_eq!(
            resolve_root(None, &env(&[("HOME", "/Users/u")]), Os::MacOs).unwrap(),
            PathBuf::from("/Users/u/Library/Caches/archai-jev")
        );
        let win = resolve_root(
            None,
            &env(&[("LOCALAPPDATA", "C:\\Users\\u\\AppData\\Local")]),
            Os::Windows,
        )
        .unwrap();
        assert!(
            win.ends_with("archai-jev\\Cache") || win.ends_with("archai-jev/Cache"),
            "{win:?}"
        );
    }

    #[test]
    fn precedence_argument_then_variable_then_default() {
        let e = env(&[("ARCHAI_JEV_CACHE", "/from/env"), ("HOME", "/home/u")]);
        assert_eq!(
            resolve_root(Some(Path::new("/arg")), &e, Os::Linux).unwrap(),
            PathBuf::from("/arg")
        );
        assert_eq!(
            resolve_root(None, &e, Os::Linux).unwrap(),
            PathBuf::from("/from/env")
        );
        let empty = env(&[("ARCHAI_JEV_CACHE", ""), ("HOME", "/home/u")]);
        assert_eq!(
            resolve_root(None, &empty, Os::Linux).unwrap(),
            PathBuf::from("/home/u/.cache/archai-jev")
        );
    }

    #[test]
    fn nothing_to_go_on_is_an_error_with_a_hint() {
        for os in [Os::Linux, Os::MacOs, Os::Windows] {
            let err = resolve_root(None, &env(&[]), os).unwrap_err();
            assert!(err.to_string().contains("ARCHAI_JEV_CACHE"), "{err}");
        }
    }

    #[test]
    fn layout_is_versioned() {
        let l = Layout::new(Path::new("/r"));
        assert_eq!(l.blobs, PathBuf::from("/r/v1/blobs"));
        assert_eq!(l.verified, PathBuf::from("/r/v1/verified"));
        assert_eq!(l.materialized, PathBuf::from("/r/v1/materialized"));
    }
}
