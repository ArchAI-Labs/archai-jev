//! A minimal, strict reader of the zip container of `torch.save` files (spec 018, section 4).
//!
//! `torch.save` writes an **uncompressed** zip: stored entries, a central directory at the end.
//! We accept exactly that and nothing else: no compression, no encryption, no zip64, no split
//! archives, no overlapping or duplicate entries, every offset inside the file. Entry names are
//! only ever used as lookup keys, never as paths.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// Largest file we accept: a head is a few MiB, the limit is generous but finite.
pub const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_ENTRIES: usize = 64;
const EOCD_SIG: u32 = 0x0605_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

/// A stored entry of the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Its name inside the archive (a lookup key, never a path).
    pub name: String,
    /// Where its data starts in the file.
    pub offset: u64,
    /// Its length in bytes.
    pub len: u64,
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    let s = b.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([*s.first()?, *s.get(1)?]))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([
        *s.first()?,
        *s.get(1)?,
        *s.get(2)?,
        *s.get(3)?,
    ]))
}

fn io(e: &std::io::Error) -> String {
    format!("cannot read the file: {e}")
}

fn read_at(file: &mut File, at: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(at))
        .and_then(|_| file.read_exact(&mut buf))
        .map_err(|e| io(&e))?;
    Ok(buf)
}

/// Read and check the central directory of the zip file `file` of `file_len` bytes.
///
/// # Errors
/// A description of what is not acceptable (the caller turns it into `IncompatibleModelError`).
pub fn read_entries(file: &mut File, file_len: u64) -> Result<Vec<Entry>, String> {
    if file_len > MAX_FILE_BYTES {
        return Err(format!(
            "the file is {file_len} bytes; the maximum is {MAX_FILE_BYTES}"
        ));
    }
    if file_len < 22 {
        return Err("the file is too short to be a zip archive".to_string());
    }
    let tail_len = file_len.min(22 + 65_535);
    let tail_start = file_len - tail_len;
    let tail = read_at(
        file,
        tail_start,
        usize::try_from(tail_len).map_err(|_| "the file is too large")?,
    )?;
    let eocd = (0..=tail.len() - 22)
        .rev()
        .find(|&p| {
            u32_at(&tail, p) == Some(EOCD_SIG)
                && u16_at(&tail, p + 20).is_some_and(|c| p + 22 + usize::from(c) == tail.len())
        })
        .ok_or("it is not a zip archive (no end-of-central-directory record)")?;
    let field16 = |at: usize| u16_at(&tail, eocd + at).ok_or("a truncated end record");
    let field32 = |at: usize| u32_at(&tail, eocd + at).ok_or("a truncated end record");
    let (disk, cd_disk, here, total) = (field16(4)?, field16(6)?, field16(8)?, field16(10)?);
    let (cd_size, cd_offset) = (field32(12)?, field32(16)?);
    if disk != 0 || cd_disk != 0 || here != total {
        return Err("split zip archives are not supported".to_string());
    }
    if total == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF {
        return Err("zip64 archives are not supported".to_string());
    }
    let (cd_size, cd_offset) = (u64::from(cd_size), u64::from(cd_offset));
    let cd_end = cd_offset
        .checked_add(cd_size)
        .ok_or("the central directory overflows")?;
    if cd_end > tail_start + eocd as u64 {
        return Err("the central directory is outside the file".to_string());
    }
    if usize::from(total) > MAX_ENTRIES {
        return Err(format!(
            "the archive has {total} entries; the maximum is {MAX_ENTRIES}"
        ));
    }
    let cd = read_at(
        file,
        cd_offset,
        usize::try_from(cd_size).map_err(|_| "the central directory is too large")?,
    )?;

    let mut entries: Vec<Entry> = Vec::new();
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut at = 0usize;
    for _ in 0..total {
        if u32_at(&cd, at) != Some(CENTRAL_SIG) {
            return Err("the central directory is damaged".to_string());
        }
        let get16 = |o: usize| u16_at(&cd, at + o).ok_or("a truncated directory entry");
        let get32 = |o: usize| u32_at(&cd, at + o).ok_or("a truncated directory entry");
        let (flags, method) = (get16(8)?, get16(10)?);
        let (csize, usize_) = (get32(20)?, get32(24)?);
        let (name_len, extra_len, comment_len) = (
            usize::from(get16(28)?),
            usize::from(get16(30)?),
            usize::from(get16(32)?),
        );
        let (disk_start, local) = (get16(34)?, get32(42)?);
        let name_bytes = cd
            .get(at + 46..at + 46 + name_len)
            .ok_or("a truncated directory entry")?;
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|_| "an entry name is not valid UTF-8")?;
        at += 46 + name_len + extra_len + comment_len;
        if flags & 0x0041 != 0 {
            return Err(format!("entry '{name}' is encrypted"));
        }
        if method != 0 {
            return Err(format!(
                "entry '{name}' is compressed (method {method}); only stored entries are accepted"
            ));
        }
        if csize != usize_ || csize == 0xFFFF_FFFF || local == 0xFFFF_FFFF || disk_start != 0 {
            return Err(format!("entry '{name}' has inconsistent sizes or offsets"));
        }
        if entries.iter().any(|e| e.name == name) {
            return Err(format!("entry '{name}' appears twice"));
        }
        let local = u64::from(local);
        let header = read_at(file, local, 30)?;
        if u32_at(&header, 0) != Some(LOCAL_SIG) {
            return Err(format!("entry '{name}' has no local header where expected"));
        }
        let l_flags = u16_at(&header, 6).ok_or("a truncated local header")?;
        let l_method = u16_at(&header, 8).ok_or("a truncated local header")?;
        let l_name = usize::from(u16_at(&header, 26).ok_or("a truncated local header")?);
        let l_extra = u64::from(u16_at(&header, 28).ok_or("a truncated local header")?);
        if l_method != 0 || l_flags & 0x0041 != 0 {
            return Err(format!("entry '{name}' is compressed or encrypted"));
        }
        if l_flags & 0x0008 == 0
            && (u32_at(&header, 18) != Some(csize) || u32_at(&header, 22) != Some(usize_))
        {
            return Err(format!(
                "entry '{name}' has different sizes in its local header and in the directory"
            ));
        }
        if read_at(file, local + 30, l_name)? != name_bytes {
            return Err(format!(
                "entry '{name}' has a different name in its local header"
            ));
        }
        let offset = local + 30 + l_name as u64 + l_extra;
        let end = offset
            .checked_add(u64::from(usize_))
            .ok_or("an entry overflows")?;
        if end > cd_offset {
            return Err(format!("entry '{name}' runs into the central directory"));
        }
        spans.push((local, end));
        entries.push(Entry {
            name,
            offset,
            len: u64::from(usize_),
        });
    }
    spans.sort_unstable();
    if spans.windows(2).any(|w| match w {
        [a, b] => b.0 < a.1,
        _ => false,
    }) {
        return Err("two entries overlap".to_string());
    }
    Ok(entries)
}

/// Read the whole data of `entry`.
///
/// # Errors
/// An I/O failure.
pub fn read_entry(file: &mut File, entry: &Entry) -> Result<Vec<u8>, String> {
    read_at(
        file,
        entry.offset,
        usize::try_from(entry.len).map_err(|_| "an entry is too large")?,
    )
}
