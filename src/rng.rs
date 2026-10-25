//! The generator behind every random choice of the crate (fold assignment, dataset splits,
//! bootstrap): SplitMix64, written out in full so the same seed gives the same numbers on every
//! operating system, architecture and version (spec 017 §3.1, §3.3).

/// SplitMix64 (Steele, Lea & Flood, 2014): a 64-bit counter run through a bit mixer.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator that starts from `seed`.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n` with no modulo bias (draws that fall in the incomplete last block of
    /// `n` values are thrown away). `below(0)` is 0.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let block = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < block {
                return x % n;
            }
        }
    }

    /// Fisher-Yates: walks from the end, swapping each position with a random earlier one.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = usize::try_from(self.below(i as u64 + 1)).unwrap_or(0);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_literals() {
        // The reference sequence of SplitMix64 for seed 0 (also in the original C code).
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next_u64(), 0x06C4_5D18_8009_454F);
        let mut r = SplitMix64::new(1234567);
        assert_eq!(r.next_u64(), 6457827717110365317);
        assert_eq!(r.next_u64(), 3203168211198807973);
    }

    #[test]
    fn below_is_unbiased_on_small_n() {
        let mut r = SplitMix64::new(7);
        let mut counts = [0usize; 3];
        for _ in 0..30_000 {
            counts[r.below(3) as usize] += 1;
        }
        for c in counts {
            assert!((9_500..=10_500).contains(&c), "{counts:?}");
        }
        assert_eq!(r.below(0), 0);
        assert_eq!(r.below(1), 0);
    }

    #[test]
    fn below_never_leaves_its_range() {
        let mut r = SplitMix64::new(99);
        for n in [2u64, 3, 10, 1 << 40, u64::MAX - 1, u64::MAX] {
            for _ in 0..50 {
                assert!(r.below(n) < n);
            }
        }
    }

    #[test]
    fn shuffle_is_a_permutation_and_deterministic() {
        let mut a: Vec<usize> = (0..20).collect();
        SplitMix64::new(3).shuffle(&mut a);
        let mut b: Vec<usize> = (0..20).collect();
        SplitMix64::new(3).shuffle(&mut b);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_ne!(a, (0..20).collect::<Vec<_>>());
        let mut c: Vec<usize> = (0..20).collect();
        SplitMix64::new(4).shuffle(&mut c);
        assert_ne!(a, c);
        let mut empty: Vec<u8> = Vec::new();
        SplitMix64::new(0).shuffle(&mut empty);
    }
}
