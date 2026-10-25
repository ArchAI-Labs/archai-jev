//! Groups of records, folds and train/test splits (spec 017 §3.1, §3.3).
//!
//! Records of one group never end up on different sides: a user's several tickets, or the
//! variants of one document, must not leak from the part used to calibrate into the part used
//! to measure.

use crate::error::{Error, Result};
use crate::rng::SplitMix64;

use super::problem::ArgumentProblem;

/// The most folds of `calibrate`.
pub const MAX_FOLDS: usize = 50;

/// The fold of each group: the groups (in order of first appearance) are shuffled with
/// `SplitMix64(seed)` and the one at position `p` goes to fold `p mod folds`, so the folds differ
/// by at most one group.
///
/// # Errors
/// `folds` outside `2..=50`, or more folds than groups.
pub fn assign_folds(groups: usize, folds: usize, seed: u64) -> Result<Vec<usize>> {
    if !(2..=MAX_FOLDS).contains(&folds) {
        return Err(Error::EvalArgument(ArgumentProblem::Folds {
            got: folds.to_string(),
        }));
    }
    if groups < folds {
        return Err(Error::EvalArgument(ArgumentProblem::MoreFoldsThanGroups {
            groups,
            folds,
        }));
    }
    let mut order: Vec<usize> = (0..groups).collect();
    SplitMix64::new(seed).shuffle(&mut order);
    let mut fold_of = vec![0; groups];
    for (position, group) in order.into_iter().enumerate() {
        if let Some(slot) = fold_of.get_mut(group) {
            *slot = position % folds;
        }
    }
    Ok(fold_of)
}

/// Which groups go to the test side. `sizes[g]` is the number of records of group `g`.
///
/// The groups are shuffled with `SplitMix64(seed)` and added whole to the test side until it
/// holds at least `test_fraction` of the records.
///
/// # Errors
/// `test_fraction` outside `(0, 1)`, or one side left empty.
pub fn split_groups(sizes: &[usize], test_fraction: f64, seed: u64) -> Result<Vec<bool>> {
    if !(test_fraction.is_finite() && test_fraction > 0.0 && test_fraction < 1.0) {
        return Err(Error::EvalArgument(ArgumentProblem::TestFraction {
            got: test_fraction,
        }));
    }
    let total: usize = sizes.iter().sum();
    let target = test_fraction * total as f64;
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    SplitMix64::new(seed).shuffle(&mut order);
    let mut is_test = vec![false; sizes.len()];
    let mut taken = 0usize;
    for group in order {
        if taken as f64 >= target && taken > 0 {
            break;
        }
        if let (Some(flag), Some(size)) = (is_test.get_mut(group), sizes.get(group)) {
            *flag = true;
            taken += size;
        }
    }
    let test_groups = is_test.iter().filter(|t| **t).count();
    if test_groups == 0 || test_groups == sizes.len() {
        let side = if test_groups == 0 { "test" } else { "train" };
        return Err(Error::EvalArgument(ArgumentProblem::EmptySide {
            fraction: test_fraction,
            side,
            groups: sizes.len(),
        }));
    }
    Ok(is_test)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn message(e: Error) -> String {
        e.to_string()
    }

    #[test]
    fn folds_literal_small_example() {
        // Cross-checked with an independent Python SplitMix64 + Fisher-Yates.
        assert_eq!(assign_folds(7, 3, 0).unwrap(), PIN_FOLDS_7_3_0);
        assert_eq!(assign_folds(7, 3, 1).unwrap(), PIN_FOLDS_7_3_1);
        assert_ne!(
            assign_folds(7, 3, 0).unwrap(),
            assign_folds(7, 3, 1).unwrap()
        );
    }

    const PIN_FOLDS_7_3_0: [usize; 7] = [2, 2, 0, 1, 1, 0, 0];
    const PIN_FOLDS_7_3_1: [usize; 7] = [1, 2, 0, 0, 2, 0, 1];

    #[test]
    fn same_group_same_fold() {
        // The fold belongs to the group, so every record of the group has it by construction;
        // here: the function returns exactly one fold per group.
        let folds = assign_folds(10, 4, 5).unwrap();
        assert_eq!(folds.len(), 10);
        assert!(folds.iter().all(|f| *f < 4));
    }

    proptest! {
        #[test]
        fn folds_balanced_within_one_group(groups in 2usize..200, folds in 2usize..=50, seed in any::<u64>()) {
            prop_assume!(groups >= folds);
            let fold_of = assign_folds(groups, folds, seed).unwrap();
            let mut counts = vec![0usize; folds];
            for f in &fold_of { counts[*f] += 1; }
            let (min, max) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
            prop_assert!(max - min <= 1, "{counts:?}");
            prop_assert_eq!(counts.iter().sum::<usize>(), groups);
        }

        #[test]
        fn split_fraction_proptest(
            sizes in prop::collection::vec(1usize..8, 2..60),
            fraction in 0.05f64..0.95,
            seed in any::<u64>(),
        ) {
            if let Ok(is_test) = split_groups(&sizes, fraction, seed) {
                let total: usize = sizes.iter().sum();
                let test: usize = sizes.iter().zip(&is_test).filter(|(_, t)| **t).map(|(s, _)| *s).sum();
                prop_assert!(test as f64 >= fraction * total as f64);
                let biggest = *sizes.iter().max().unwrap();
                prop_assert!((test as f64) < fraction * total as f64 + biggest as f64);
            }
        }
    }

    #[test]
    fn split_literal_small_example() {
        assert_eq!(split_groups(&[2, 1, 3, 1, 1], 0.4, 0).unwrap(), PIN_SPLIT);
    }

    const PIN_SPLIT: [bool; 5] = [false, false, true, true, false];

    #[test]
    fn split_never_cuts_a_group() {
        // Groups are the unit: a flag per group, so a group is on one side by construction;
        // with sizes the test side is made of whole groups.
        let sizes = [3, 3, 3, 3];
        let is_test = split_groups(&sizes, 0.5, 9).unwrap();
        let test_records: usize = sizes
            .iter()
            .zip(&is_test)
            .filter(|(_, t)| **t)
            .map(|(s, _)| *s)
            .sum();
        assert_eq!(test_records % 3, 0);
    }

    #[test]
    fn split_errors() {
        for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            let e = split_groups(&[1, 1, 1], bad, 0).unwrap_err();
            assert!(
                message(e).starts_with("test_fraction must be a number strictly between 0 and 1")
            );
        }
        let e = split_groups(&[5], 0.5, 0).unwrap_err();
        assert!(message(e).contains("leaves the train set empty"));
        let e = split_groups(&[], 0.5, 0).unwrap_err();
        assert!(message(e).contains("empty"));
        assert_eq!(
            message(assign_folds(3, 5, 0).unwrap_err()),
            "dataset has 3 groups but folds=5; use more records or fewer folds"
        );
        assert_eq!(
            message(assign_folds(10, 1, 0).unwrap_err()),
            "folds must be an integer between 2 and 50, got 1"
        );
        assert_eq!(
            message(assign_folds(100, 51, 0).unwrap_err()),
            "folds must be an integer between 2 and 50, got 51"
        );
    }
}
