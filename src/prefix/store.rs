//! The memory of saved prefixes: a map by key, least recently used out first, limited in **bytes**.
//!
//! Entries are immutable and shared (`Arc`), so a request can keep using one while another
//! evicts it. A lookup only counts as a hit when the ids and the fingerprint are equal too, so a
//! collision of the hash can never serve the wrong prefix.

use std::collections::HashMap;
use std::sync::Arc;

use super::admission::Seen;
use super::key::{Fingerprint, PrefixKey};
use crate::engine::Snapshot;

/// A saved prefix.
#[derive(Debug)]
pub struct Entry {
    /// The tokens of the prefix.
    pub ids: Box<[u32]>,
    /// The saved state of the engine after decoding them.
    pub snapshot: Snapshot,
    /// Who computed it.
    pub fingerprint: Fingerprint,
}

impl Entry {
    /// Memory the entry takes: the snapshot, the ids and a fixed overhead for the bookkeeping.
    pub fn bytes(&self) -> usize {
        self.snapshot.bytes() + 4 * self.ids.len() + 128
    }
}

/// Counters that only grow (until reset), kept by the scorer once per request.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// Requests that found a saved prefix.
    pub hits: u64,
    /// Requests that did not.
    pub misses: u64,
    /// Entries removed to make room.
    pub evictions: u64,
    /// Tokens not decoded because a saved prefix was restored.
    pub tokens_reused: u64,
    /// Tokens the engine decoded.
    pub tokens_decoded: u64,
}

/// The store.
#[derive(Debug)]
pub struct PrefixStore {
    limit_bytes: usize,
    entries: HashMap<PrefixKey, Arc<Entry>>,
    /// Oldest use first.
    order: Vec<PrefixKey>,
    bytes: usize,
    /// Counters (public: the scorer adds to them).
    pub counters: Counters,
    /// Content prefixes seen once and not saved.
    pub seen: Seen,
}

impl PrefixStore {
    /// An empty store that may hold `limit_bytes` bytes. `0` holds nothing.
    pub fn new(limit_bytes: usize) -> Self {
        PrefixStore {
            limit_bytes,
            entries: HashMap::new(),
            order: Vec::new(),
            bytes: 0,
            counters: Counters::default(),
            seen: Seen::new(),
        }
    }

    /// The limit in bytes.
    pub fn limit_bytes(&self) -> usize {
        self.limit_bytes
    }

    /// Bytes held now.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Entries held now.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store holds nothing.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn touch(&mut self, key: &PrefixKey) {
        if let Some(i) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(i);
            self.order.push(k);
        }
    }

    /// The entry for `key`, if it holds exactly these `ids` and was made by this `fingerprint`;
    /// it becomes the most recently used. (Hits and misses are counted by the caller, once per
    /// request.)
    pub fn get(
        &mut self,
        key: &PrefixKey,
        ids: &[u32],
        fingerprint: &Fingerprint,
    ) -> Option<Arc<Entry>> {
        let entry = self.entries.get(key)?;
        if entry.fingerprint != *fingerprint || *entry.ids != *ids {
            return None;
        }
        let entry = Arc::clone(entry);
        self.touch(key);
        Some(entry)
    }

    /// Whether `key` is there (does not change the order).
    pub fn contains(&self, key: &PrefixKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Save an entry, evicting the least recently used until it fits. An entry larger than the
    /// whole limit is not saved (that is not an error). Returns whether it was saved.
    pub fn insert(&mut self, key: PrefixKey, entry: Entry) -> bool {
        let size = entry.bytes();
        if size > self.limit_bytes {
            return false;
        }
        self.remove(&key);
        while self.bytes + size > self.limit_bytes {
            let Some(oldest) = self.order.first().copied() else {
                break;
            };
            if self.remove(&oldest) {
                self.counters.evictions += 1;
            }
        }
        self.bytes += size;
        self.order.push(key);
        self.entries.insert(key, Arc::new(entry));
        true
    }

    /// Drop an entry (for example one whose restore failed). Returns whether it was there.
    pub fn remove(&mut self, key: &PrefixKey) -> bool {
        let Some(entry) = self.entries.remove(key) else {
            return false;
        };
        self.bytes -= entry.bytes();
        self.order.retain(|k| k != key);
        true
    }

    /// Drop every entry and the remembered keys. The counters stay.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
        self.seen.clear();
    }

    /// Set the counters back to zero.
    pub fn reset_counters(&mut self) {
        self.counters = Counters::default();
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::prefix::key::{FingerprintParts, KeyHasher};

    fn fp(model: &str) -> Fingerprint {
        Fingerprint::new(&FingerprintParts {
            model,
            gguf_sha256: "g",
            family: "f",
            dtype: "d",
            template_id: "t",
            template_version: 1,
            tokenizer_sha256: "k",
            engine: "e",
        })
    }

    fn key_of(fp: &Fingerprint, ids: &[u32]) -> PrefixKey {
        let mut h = KeyHasher::new(fp);
        h.update(ids);
        h.key()
    }

    /// An entry whose `bytes()` is exactly `size` (needs `size >= 128 + 4`).
    fn entry(fp: &Fingerprint, ids: &[u32], size: usize) -> Entry {
        let snapshot_bytes = size - 128 - 4 * ids.len();
        Entry {
            ids: ids.into(),
            snapshot: Snapshot::new(ids.len(), snapshot_bytes, ()),
            fingerprint: *fp,
        }
    }

    #[test]
    fn insert_get_and_the_ids_and_fingerprint_must_match() {
        let f = fp("a");
        let mut s = PrefixStore::new(10_000);
        let ids = [1u32, 2, 3];
        let k = key_of(&f, &ids);
        assert!(s.insert(k, entry(&f, &ids, 500)));
        assert_eq!((s.len(), s.bytes()), (1, 500));
        assert!(s.get(&k, &ids, &f).is_some());
        // Same digest asked with other ids (a forced collision): a miss, never the wrong prefix.
        assert!(s.get(&k, &[1, 2, 4], &f).is_none());
        // Same digest and ids but another fingerprint: a miss.
        assert!(s.get(&k, &ids, &fp("b")).is_none());
    }

    #[test]
    fn equal_digest_different_ids_is_a_miss() {
        // The entry stored under `k` holds other ids than the ones the lookup brings.
        let f = fp("a");
        let mut s = PrefixStore::new(10_000);
        let k = key_of(&f, &[1, 2, 3]);
        assert!(s.insert(k, entry(&f, &[9, 9, 9], 500)));
        assert!(s.get(&k, &[1, 2, 3], &f).is_none());
    }

    #[test]
    fn the_least_recently_used_goes_first() {
        let f = fp("a");
        let mut s = PrefixStore::new(1_000);
        let keys: Vec<_> = (0..3u32).map(|i| key_of(&f, &[i])).collect();
        for (i, k) in keys.iter().enumerate() {
            assert!(s.insert(*k, entry(&f, &[i as u32], 400)));
        }
        // 3 x 400 > 1000: the first one was evicted to fit the third.
        assert_eq!((s.len(), s.counters.evictions), (2, 1));
        assert!(!s.contains(&keys[0]));
        // Use the second; the next insertion must evict the third, not the second.
        assert!(s.get(&keys[1], &[1], &f).is_some());
        let k4 = key_of(&f, &[4]);
        assert!(s.insert(k4, entry(&f, &[4], 400)));
        assert!(s.contains(&keys[1]) && s.contains(&k4) && !s.contains(&keys[2]));
        assert_eq!(s.counters.evictions, 2);
    }

    #[test]
    fn an_entry_bigger_than_the_limit_is_not_saved_and_is_not_an_error() {
        let f = fp("a");
        let mut s = PrefixStore::new(1_000);
        let k = key_of(&f, &[1]);
        assert!(s.insert(k, entry(&f, &[1], 600)));
        assert!(!s.insert(key_of(&f, &[2]), entry(&f, &[2], 1_001)));
        // Nothing was evicted for it.
        assert_eq!((s.len(), s.bytes(), s.counters.evictions), (1, 600, 0));
    }

    #[test]
    fn limit_zero_stores_nothing() {
        let f = fp("a");
        let mut s = PrefixStore::new(0);
        assert!(!s.insert(key_of(&f, &[1]), entry(&f, &[1], 200)));
        assert!(s.is_empty());
    }

    #[test]
    fn inserting_the_same_key_again_replaces_it() {
        let f = fp("a");
        let mut s = PrefixStore::new(2_000);
        let k = key_of(&f, &[1]);
        assert!(s.insert(k, entry(&f, &[1], 500)));
        assert!(s.insert(k, entry(&f, &[1], 700)));
        assert_eq!((s.len(), s.bytes()), (1, 700));
    }

    #[test]
    fn clear_drops_entries_and_remembered_keys_but_not_the_counters() {
        let f = fp("a");
        let mut s = PrefixStore::new(1_000);
        s.insert(key_of(&f, &[1]), entry(&f, &[1], 400));
        s.insert(key_of(&f, &[2]), entry(&f, &[2], 400));
        s.insert(key_of(&f, &[3]), entry(&f, &[3], 400));
        s.seen.remember(&key_of(&f, &[9]));
        s.clear();
        assert_eq!((s.len(), s.bytes()), (0, 0));
        assert!(s.seen.is_empty());
        assert_eq!(s.counters.evictions, 1);
        s.reset_counters();
        assert_eq!(s.counters, Counters::default());
    }

    #[test]
    fn remove_frees_the_bytes() {
        let f = fp("a");
        let mut s = PrefixStore::new(1_000);
        let k = key_of(&f, &[1]);
        s.insert(k, entry(&f, &[1], 400));
        assert!(s.remove(&k));
        assert!(!s.remove(&k));
        assert_eq!((s.len(), s.bytes()), (0, 0));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            rng_seed: proptest::test_runner::RngSeed::Fixed(11),
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn prop_never_over_limit(
            limit in 0usize..5_000,
            ops in proptest::collection::vec((0u32..12, 140usize..2_500, any::<bool>()), 0..60),
        ) {
            let f = fp("a");
            let mut s = PrefixStore::new(limit);
            for (id, size, read) in ops {
                let ids = [id];
                let k = key_of(&f, &ids);
                if read {
                    let _ = s.get(&k, &ids, &f);
                } else {
                    let saved = s.insert(k, entry(&f, &ids, size));
                    prop_assert_eq!(saved, size <= limit);
                    if saved {
                        prop_assert!(s.contains(&k));
                    }
                }
                prop_assert!(s.bytes() <= limit);
                // The bytes are exactly the sum of the entries.
                let total: usize = s.entries.values().map(|e| e.bytes()).sum();
                prop_assert_eq!(total, s.bytes());
                prop_assert_eq!(s.order.len(), s.len());
            }
        }
    }
}
