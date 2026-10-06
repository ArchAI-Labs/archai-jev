//! File entries of a manifest: portable relative paths, size and SHA-256, optional origin.

use super::incompat::Incompat;
use crate::json_strict::{Json, Obj};

/// Where a file can be downloaded from (registry manifests only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOrigin {
    /// Hugging Face repo id, e.g. `Qwen/Qwen2.5-1.5B-Instruct`.
    pub repo: String,
    /// Full commit hash (40 lowercase hex digits).
    pub revision: String,
}

/// A file the model needs, with the hash it must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Portable relative path (inside the repo, or relative to the manifest's folder).
    pub path: String,
    /// Exact size in bytes.
    pub size: u64,
    /// SHA-256, 64 lowercase hex digits.
    pub sha256: String,
    /// Download source; `None` for local manifests.
    pub origin: Option<FileOrigin>,
}

/// True for 64 lowercase hex digits.
pub fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// True for a 40-digit lowercase hex commit hash.
pub fn is_commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Why a path is not portable, if it is not.
pub fn path_problem(path: &str) -> Option<&'static str> {
    if path.is_empty() {
        return Some("is empty");
    }
    if path.starts_with('/') {
        return Some("must be relative, not absolute");
    }
    if path.contains('\\') {
        return Some("must use '/' as separator, not '\\'");
    }
    if path.contains('\0') {
        return Some("contains a NUL character");
    }
    let bytes = path.as_bytes();
    if bytes.get(1) == Some(&b':') && bytes.first().is_some_and(u8::is_ascii_alphabetic) {
        return Some("must not start with a drive letter");
    }
    for part in path.split('/') {
        if part.is_empty() {
            return Some("has an empty path component");
        }
        if part == "." || part == ".." {
            return Some("must not contain '.' or '..' components");
        }
        if part.len() > 255 {
            return Some("has a component longer than 255 bytes");
        }
    }
    None
}

impl FileEntry {
    /// Read a file entry from the object `json` called `path_in_manifest`.
    ///
    /// `registry` says whether `origin` is allowed: in a local manifest it is an unknown field.
    pub(crate) fn from_json(
        json: &Json,
        path_in_manifest: &str,
        registry: bool,
    ) -> Result<FileEntry, Incompat> {
        let mut o = Obj::new(json, path_in_manifest)?;
        let path = o.str("path")?;
        let bad = |field: &str, detail: String| Incompat::ManifestBadValue {
            path: format!("{path_in_manifest}.{field}"),
            detail,
        };
        if let Some(problem) = path_problem(path) {
            return Err(bad("path", format!("the path {path:?} {problem}")));
        }
        let size = o.u64("size")?;
        if size == 0 {
            return Err(bad("size", "must be greater than 0".to_string()));
        }
        let sha256 = o.str("sha256")?;
        if !is_sha256(sha256) {
            return Err(bad(
                "sha256",
                format!("must be 64 lowercase hex digits, got {sha256:?}"),
            ));
        }
        let origin = if registry {
            match o.req("origin")? {
                Json::Null => None,
                other => {
                    let mut oo = Obj::new(other, &format!("{path_in_manifest}.origin"))?;
                    let repo = oo.str("repo")?;
                    let revision = oo.str("revision")?;
                    oo.finish()?;
                    let ok_repo = !repo.is_empty()
                        && repo.bytes().all(|b| {
                            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/')
                        });
                    if !ok_repo {
                        return Err(bad(
                            "origin.repo",
                            format!("is not a repository id: {repo:?}"),
                        ));
                    }
                    if !is_commit(revision) {
                        return Err(bad(
                            "origin.revision",
                            format!("must be a full 40-digit commit hash, got {revision:?}"),
                        ));
                    }
                    Some(FileOrigin {
                        repo: repo.to_string(),
                        revision: revision.to_string(),
                    })
                }
            }
        } else {
            None
        };
        o.finish()?;
        Ok(FileEntry {
            path: path.to_string(),
            size,
            sha256: sha256.to_string(),
            origin,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_rules() {
        for ok in [
            "model.gguf",
            "adapter/model.safetensors",
            "a/b/c.json",
            "é/ü.txt",
        ] {
            assert_eq!(path_problem(ok), None, "{ok}");
        }
        for bad in [
            "",
            "/etc/passwd",
            "a\\b",
            "../x",
            "a/../b",
            "./a",
            "a//b",
            "C:/x",
            "c:x",
            "a/",
            "a\0b",
        ] {
            assert!(path_problem(bad).is_some(), "{bad:?}");
        }
        assert!(path_problem(&"x".repeat(256)).is_some());
        assert!(path_problem(&"x".repeat(255)).is_none());
    }

    #[test]
    fn hash_shapes() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256(&"A".repeat(64)));
        assert!(!is_sha256(&"a".repeat(63)));
        assert!(is_commit(&"0".repeat(40)));
        assert!(!is_commit("e4f3964a"));
    }
}
