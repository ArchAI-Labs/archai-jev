//! Which prefixes are worth saving.
//!
//! A snapshot costs memory (about 29 KB per token on the default model) and a save costs time.
//! The fixed part of a task is worth keeping from the first time: every request of that task
//! starts with it. The part that depends on the user's text (the state) is only worth keeping if
//! it is going to be asked again: when several questions of the request share it, or when the same
//! state was already seen (a stream of questions about one source).

use std::collections::{HashSet, VecDeque};

use super::key::PrefixKey;
use crate::prompt::CutKind;

/// A prefix shorter than this is cheaper to decode than to restore.
pub const MIN_CACHED_TOKENS: usize = 16;

/// How many keys of unsaved `Content` prefixes are remembered.
pub const SEEN_CAPACITY: usize = 4096;

/// The keys of `Content` prefixes that were offered once and not saved (first in, first out).
#[derive(Debug, Default)]
pub struct Seen {
    order: VecDeque<u64>,
    set: HashSet<u64>,
    capacity: usize,
}

impl Seen {
    /// A set that remembers the last [`SEEN_CAPACITY`] keys.
    pub fn new() -> Self {
        Seen::with_capacity(SEEN_CAPACITY)
    }

    /// A set that remembers the last `capacity` keys.
    pub fn with_capacity(capacity: usize) -> Self {
        Seen {
            order: VecDeque::new(),
            set: HashSet::new(),
            capacity,
        }
    }

    /// Whether the key was remembered.
    pub fn contains(&self, key: &PrefixKey) -> bool {
        self.set.contains(&key.short())
    }

    /// Remember a key, forgetting the oldest if the set is full.
    pub fn remember(&mut self, key: &PrefixKey) {
        if self.capacity == 0 || !self.set.insert(key.short()) {
            return;
        }
        self.order.push_back(key.short());
        while self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
    }

    /// How many keys are remembered.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether nothing is remembered.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.order.clear();
        self.set.clear();
    }
}

/// Whether to save the prefix of `tokens` tokens that ends at a cut of this `kind`, shared by
/// `rows_sharing` rows of the request.
pub fn admit(
    kind: CutKind,
    tokens: usize,
    rows_sharing: usize,
    key: &PrefixKey,
    seen: &mut Seen,
) -> bool {
    if tokens < MIN_CACHED_TOKENS {
        return false;
    }
    match kind {
        CutKind::Fixed => true,
        CutKind::Content => {
            if rows_sharing >= 2 || seen.contains(key) {
                true
            } else {
                seen.remember(key);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::key::{Fingerprint, FingerprintParts, KeyHasher};

    fn key(n: u32) -> PrefixKey {
        let fp = Fingerprint::new(&FingerprintParts {
            model: "m",
            gguf_sha256: "g",
            family: "f",
            dtype: "d",
            template_id: "t",
            template_version: 1,
            tokenizer_sha256: "k",
            engine: "e",
        });
        let mut h = KeyHasher::new(&fp);
        h.update(&[n]);
        h.key()
    }

    #[test]
    fn a_fixed_prefix_is_always_saved_if_long_enough() {
        let mut seen = Seen::new();
        assert!(admit(CutKind::Fixed, 43, 1, &key(1), &mut seen));
        assert!(admit(
            CutKind::Fixed,
            MIN_CACHED_TOKENS,
            1,
            &key(1),
            &mut seen
        ));
        assert!(!admit(
            CutKind::Fixed,
            MIN_CACHED_TOKENS - 1,
            1,
            &key(1),
            &mut seen
        ));
        assert!(seen.is_empty(), "fixed prefixes are not remembered");
    }

    #[test]
    fn a_content_prefix_needs_two_rows_or_a_second_sighting() {
        let mut seen = Seen::new();
        // One row, first time: not saved, but remembered.
        assert!(!admit(CutKind::Content, 80, 1, &key(7), &mut seen));
        assert!(seen.contains(&key(7)));
        // The same prefix again: saved.
        assert!(admit(CutKind::Content, 80, 1, &key(7), &mut seen));
        // Two rows share it: saved at once, and nothing needs to be remembered.
        assert!(admit(CutKind::Content, 80, 2, &key(8), &mut seen));
        assert!(!seen.contains(&key(8)));
        // Too short: never.
        assert!(!admit(CutKind::Content, 10, 5, &key(9), &mut seen));
    }

    #[test]
    fn the_seen_set_has_a_fixed_capacity_and_forgets_the_oldest() {
        let mut seen = Seen::with_capacity(3);
        for n in 0..5 {
            seen.remember(&key(n));
        }
        assert_eq!(seen.len(), 3);
        assert!(!seen.contains(&key(0)) && !seen.contains(&key(1)));
        assert!(seen.contains(&key(2)) && seen.contains(&key(4)));
        // Remembering twice does not take two places.
        seen.remember(&key(4));
        assert_eq!(seen.len(), 3);
        seen.clear();
        assert!(seen.is_empty());
        // A set of capacity 0 remembers nothing.
        let mut none = Seen::with_capacity(0);
        none.remember(&key(1));
        assert!(none.is_empty());
    }
}
