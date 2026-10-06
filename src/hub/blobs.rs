//! The content-addressed store: a file is `blobs/<sha256>`, so the name proves what it is.
//! Writes go to `<sha256>.part` under an exclusive file lock and are renamed only after the
//! hash matched, so a partial or unverified file never has its final name.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

/// The folder holding the blobs.
#[derive(Debug, Clone)]
pub struct BlobStore {
    dir: PathBuf,
}

/// An exclusive lock on a blob; released when dropped (or when the process dies).
pub struct Guard {
    _file: File,
}

impl BlobStore {
    /// A store in `dir` (created on demand).
    pub fn new(dir: PathBuf) -> Self {
        BlobStore { dir }
    }

    /// The folder.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Final path of the blob with this SHA-256.
    pub fn path(&self, sha256: &str) -> PathBuf {
        self.dir.join(sha256)
    }

    /// Path of the partial download.
    pub fn part_path(&self, sha256: &str) -> PathBuf {
        self.dir.join(format!("{sha256}.part"))
    }

    /// Create the folder if needed.
    ///
    /// # Errors
    /// Any I/O error.
    pub fn ensure_dir(&self) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)
    }

    /// Wait for and take the exclusive lock of the blob. The lock is an operating-system file
    /// lock: it disappears with the process, so a crash never leaves it stuck.
    ///
    /// # Errors
    /// Any I/O error creating or locking the lock file.
    pub fn lock(&self, sha256: &str) -> io::Result<Guard> {
        self.ensure_dir()?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.dir.join(format!("{sha256}.lock")))?;
        file.lock()?;
        Ok(Guard { _file: file })
    }

    /// True if the final blob exists with exactly `size` bytes.
    pub fn is_complete(&self, sha256: &str, size: u64) -> bool {
        std::fs::metadata(self.path(sha256)).is_ok_and(|m| m.is_file() && m.len() == size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_completeness() {
        let dir = tempfile::tempdir().unwrap();
        let s = BlobStore::new(dir.path().join("blobs"));
        let sha = "ab".repeat(32);
        assert_eq!(s.path(&sha), dir.path().join("blobs").join(&sha));
        assert_eq!(
            s.part_path(&sha),
            dir.path().join("blobs").join(format!("{sha}.part"))
        );
        assert!(!s.is_complete(&sha, 3));
        s.ensure_dir().unwrap();
        std::fs::write(s.path(&sha), b"abc").unwrap();
        assert!(s.is_complete(&sha, 3));
        assert!(!s.is_complete(&sha, 4));
    }

    #[test]
    fn the_lock_is_exclusive_between_threads() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = tempfile::tempdir().unwrap();
        let s = Arc::new(BlobStore::new(dir.path().to_path_buf()));
        let inside = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..6)
            .map(|_| {
                let (s, inside, max_seen) = (s.clone(), inside.clone(), max_seen.clone());
                std::thread::spawn(move || {
                    let _g = s.lock("x").unwrap();
                    let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    max_seen.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    inside.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(max_seen.load(Ordering::SeqCst), 1);
    }
}
