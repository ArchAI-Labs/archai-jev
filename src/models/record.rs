//! The verification record: proof, kept in the cache, that these exact files passed every
//! check with this exact library on this platform (spec 005, 7.2).
//!
//! A record is written only after integrity, structure and self-check all passed. Anything
//! unreadable, truncated or inconsistent counts as *no record*, never as a pass.

use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use super::hash::sha256_hex;
use crate::json_strict::{self, Json, Obj};

/// Version of the library that wrote the record.
pub const LIB_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Identifier that changes with the source code (from `build.rs`).
pub const BUILD_ID: &str = env!("ARCHAI_JEV_BUILD_ID");

/// `os-arch`, e.g. `windows-x86_64`.
pub fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Size and modification time of a file, as the fast path compares them.
///
/// # Errors
/// Any I/O error reading the metadata.
pub fn stat(path: &Path) -> io::Result<(u64, u128)> {
    let m = std::fs::metadata(path)?;
    let mtime = m
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    Ok((m.len(), mtime))
}

/// What was known about one file when the record was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    /// Path as shown in messages (manifest path).
    pub name: String,
    /// Size in bytes.
    pub size: u64,
    /// Modification time, nanoseconds since the epoch.
    pub mtime_ns: u128,
    /// SHA-256 verified at the time.
    pub sha256: String,
}

/// A verification record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// Its key (also its file name).
    pub key: String,
    /// The files, in a fixed order.
    pub files: Vec<Stamp>,
}

/// The key: a hash of everything that must be the same for a record to apply.
pub fn key(manifest_sha256: &str, dtype: &str) -> String {
    sha256_hex(
        format!(
            "v1|{LIB_VERSION}|{BUILD_ID}|{}|{manifest_sha256}|{dtype}",
            platform()
        )
        .as_bytes(),
    )
}

fn path_of(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.json"))
}

impl Record {
    fn to_json(&self) -> Json {
        let text = |x: &str| Json::Str(x.to_string());
        let files = self
            .files
            .iter()
            .map(|f| {
                Json::Object(vec![
                    ("name".to_string(), text(&f.name)),
                    ("size".to_string(), Json::Int(i128::from(f.size))),
                    ("mtime_ns".to_string(), text(&f.mtime_ns.to_string())),
                    ("sha256".to_string(), text(&f.sha256)),
                ])
            })
            .collect();
        Json::Object(vec![
            ("key".to_string(), text(&self.key)),
            ("passed".to_string(), Json::Bool(true)),
            ("files".to_string(), Json::Array(files)),
        ])
    }

    fn from_json(j: &Json) -> Option<Record> {
        let mut o = Obj::new(j, "record").ok()?;
        let key = o.str("key").ok()?.to_string();
        if !o.bool("passed").ok()? {
            return None;
        }
        let mut files = Vec::new();
        for f in o.arr("files").ok()? {
            let mut fo = Obj::new(f, "file").ok()?;
            files.push(Stamp {
                name: fo.str("name").ok()?.to_string(),
                size: fo.u64("size").ok()?,
                mtime_ns: fo.str("mtime_ns").ok()?.parse().ok()?,
                sha256: fo.str("sha256").ok()?.to_string(),
            });
            fo.finish().ok()?;
        }
        o.finish().ok()?;
        Some(Record { key, files })
    }
}

/// Read the record for `key` from `dir`; any problem means `None`.
pub fn load(dir: &Path, key: &str) -> Option<Record> {
    let bytes = std::fs::read(path_of(dir, key)).ok()?;
    let rec = Record::from_json(&json_strict::parse(&bytes).ok()?)?;
    (rec.key == key).then_some(rec)
}

/// Write the record atomically (temporary file, then rename).
///
/// # Errors
/// Any I/O error; the caller treats it as "no caching", not as a failure to load.
pub fn save(dir: &Path, record: &Record) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let target = path_of(dir, &record.key);
    let tmp = dir.join(format!("{}.{}.tmp", record.key, std::process::id()));
    std::fs::write(&tmp, record.to_json().to_canonical_string())?;
    std::fs::rename(&tmp, &target).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// True if every file in `current` is the one the record vouches for: same name, same size,
/// same modification time, and the hash the manifest expects.
pub fn matches(record: &Record, current: &[(String, PathBuf, String)]) -> bool {
    record.files.len() == current.len()
        && record
            .files
            .iter()
            .zip(current)
            .all(|(stamp, (name, path, sha))| {
                stamp.name == *name
                    && stamp.sha256 == *sha
                    && stat(path)
                        .is_ok_and(|(size, mtime)| size == stamp.size && mtime == stamp.mtime_ns)
            })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(key: &str, f: &Path) -> Record {
        let (size, mtime_ns) = stat(f).unwrap();
        Record {
            key: key.to_string(),
            files: vec![Stamp {
                name: "f".into(),
                size,
                mtime_ns,
                sha256: "a".repeat(64),
            }],
        }
    }

    #[test]
    fn roundtrip_and_matching() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.bin");
        std::fs::write(&f, b"hello").unwrap();
        let r = rec(&key(&"b".repeat(64), "q8_0"), &f);
        save(dir.path(), &r).unwrap();
        let back = load(dir.path(), &r.key).unwrap();
        assert_eq!(back, r);
        assert!(matches(&back, &[("f".into(), f.clone(), "a".repeat(64))]));
        // other hash, other name, other size: no match
        assert!(!matches(&back, &[("f".into(), f.clone(), "c".repeat(64))]));
        assert!(!matches(&back, &[("g".into(), f.clone(), "a".repeat(64))]));
        std::fs::write(&f, b"hello!").unwrap();
        assert!(!matches(&back, &[("f".into(), f, "a".repeat(64))]));
    }

    #[test]
    fn key_depends_on_manifest_and_dtype() {
        let a = key(&"1".repeat(64), "q8_0");
        assert_ne!(a, key(&"2".repeat(64), "q8_0"));
        assert_ne!(a, key(&"1".repeat(64), "bf16"));
        assert_eq!(a, key(&"1".repeat(64), "q8_0"));
    }

    #[test]
    fn every_kind_of_damage_means_no_record() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.bin");
        std::fs::write(&f, b"x").unwrap();
        let r = rec("k1", &f);
        save(dir.path(), &r).unwrap();
        let p = dir.path().join("k1.json");
        let good = std::fs::read(&p).unwrap();
        for damaged in [
            b"".to_vec(),
            good[..good.len() / 2].to_vec(),
            b"{}".to_vec(),
            b"not json".to_vec(),
            String::from_utf8(good.clone())
                .unwrap()
                .replace("\"passed\":true", "\"passed\":false")
                .into_bytes(),
            String::from_utf8(good.clone())
                .unwrap()
                .replace("\"key\":\"k1\"", "\"key\":\"k2\"")
                .into_bytes(),
        ] {
            std::fs::write(&p, damaged).unwrap();
            assert_eq!(load(dir.path(), "k1"), None);
        }
        assert_eq!(load(dir.path(), "missing"), None);
    }

    #[test]
    fn save_into_a_file_instead_of_a_folder_fails_softly() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let r = Record {
            key: "k".into(),
            files: vec![],
        };
        assert!(save(&blocker.join("sub"), &r).is_err());
    }
}
