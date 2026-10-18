//! Which option keys continue together in one decode (spec 010, P2).
//!
//! A key of several tokens is scored by its first token at the end of the prompt and by the
//! following tokens one after the other, fed to the model after a copy of the prompt. A decode of
//! a few tokens costs about as much as a decode of many (the weights are read once either way), so
//! the keys of a request go on **together**: a wave is a group of keys fed in one decode, each on
//! its own copy of the prompt. This module only decides the groups.

/// The indices of the keys of more than one token, in order, in waves of at most `wave` keys
/// whose fed tokens (all but the last token of each key) fit one batch of `max_batch` tokens.
///
/// Keys of one token are scored by the first token alone and are in no wave. A key that alone
/// needs more than `max_batch` tokens still gets its own wave (the decode then reports it); a
/// `wave` of `0` counts as `1`.
pub(super) fn plan_waves(keys: &[Vec<u32>], wave: usize, max_batch: usize) -> Vec<Vec<usize>> {
    let wave = wave.max(1);
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut tokens = 0usize;
    for (index, key) in keys.iter().enumerate() {
        if key.len() < 2 {
            continue;
        }
        let fed = key.len() - 1;
        match out.last_mut() {
            Some(last) if last.len() < wave && tokens + fed <= max_batch => {
                last.push(index);
                tokens += fed;
            }
            _ => {
                out.push(vec![index]);
                tokens = fed;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(len: usize) -> Vec<u32> {
        (0..len as u32).collect()
    }

    #[test]
    fn keys_of_one_token_are_in_no_wave() {
        let keys = vec![key(1), key(1), key(1)];
        assert!(plan_waves(&keys, 8, 512).is_empty());
        assert!(plan_waves(&[], 8, 512).is_empty());
    }

    #[test]
    fn a_wave_of_one_is_the_reference_path_one_key_at_a_time() {
        let keys = vec![key(3), key(1), key(2), key(4)];
        assert_eq!(plan_waves(&keys, 1, 512), vec![vec![0], vec![2], vec![3]]);
        // zero counts as one
        assert_eq!(plan_waves(&keys, 0, 512), plan_waves(&keys, 1, 512));
    }

    #[test]
    fn keys_fill_waves_in_order_up_to_the_wave_size() {
        let keys: Vec<Vec<u32>> = (0..10).map(|_| key(3)).collect();
        let waves = plan_waves(&keys, 4, 512);
        assert_eq!(
            waves,
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]],
            "{waves:?}"
        );
    }

    #[test]
    fn single_token_keys_between_longer_ones_do_not_split_a_wave() {
        let keys = vec![key(2), key(1), key(2), key(1), key(3)];
        assert_eq!(plan_waves(&keys, 8, 512), vec![vec![0, 2, 4]]);
    }

    #[test]
    fn a_wave_never_exceeds_the_batch() {
        // each key feeds 3 tokens; a batch of 7 takes two keys (6), not three (9)
        let keys: Vec<Vec<u32>> = (0..5).map(|_| key(4)).collect();
        let waves = plan_waves(&keys, 8, 7);
        assert_eq!(waves, vec![vec![0, 1], vec![2, 3], vec![4]]);
        for wave in &waves {
            let fed: usize = wave.iter().map(|&i| keys[i].len() - 1).sum();
            assert!(fed <= 7);
        }
    }

    #[test]
    fn a_key_larger_than_the_batch_goes_alone_and_every_key_is_in_exactly_one_wave() {
        let keys = vec![key(2), key(9), key(2), key(2)];
        let waves = plan_waves(&keys, 8, 4);
        assert_eq!(waves, vec![vec![0], vec![1], vec![2, 3]]);
        let mut seen: Vec<usize> = waves.into_iter().flatten().collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }
}
