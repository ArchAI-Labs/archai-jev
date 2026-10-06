//! SHA-256 of bytes and of files (streaming, 1 MiB blocks).

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// Lowercase hex of the SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(digest: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// An incremental SHA-256, for hashing while downloading.
#[derive(Default)]
pub struct Hasher(Sha256);

impl Hasher {
    /// A fresh hasher.
    pub fn new() -> Self {
        Hasher(Sha256::new())
    }
    /// Feed more bytes.
    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
    /// The lowercase hex digest.
    pub fn finish_hex(self) -> String {
        hex(&self.0.finalize())
    }
}

/// SHA-256 and size of the file at `path`, read in blocks.
///
/// # Errors
/// Any I/O error opening or reading the file.
pub fn sha256_file(path: &Path) -> io::Result<(String, u64)> {
    let mut f = File::open(path)?;
    let mut h = Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(buf.get(..n).unwrap_or_default());
        total += n as u64;
    }
    Ok((h.finish_hex(), total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn file_hash_matches_bytes_hash_across_block_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.bin");
        let data: Vec<u8> = (0..(3 * (1 << 20) + 17)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        let (h, n) = sha256_file(&p).unwrap();
        assert_eq!(n, data.len() as u64);
        assert_eq!(h, sha256_hex(&data));
    }
}
