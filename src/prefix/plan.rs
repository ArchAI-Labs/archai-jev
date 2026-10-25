//! The plan of a request: which part of the rows is the same for all of them and is decoded once.
//!
//! A template declares **cut points** in every row (where it may be split and the prefix kept).
//! A row is always decoded in the segments its own cuts define, whatever the other rows are, so a
//! row's numbers never depend on its neighbours or on what was cached. What the rows share is
//! therefore not "any common prefix" but the longest list of **common cuts**: the first `k` cuts
//! that every row has at the same position, with the same kind and the same ids before them. The
//! segments before the last common cut are decoded once (shared); each row then decodes its own
//! segments after it.

use std::ops::Range;

use crate::error::{Error, Result};
use crate::prompt::{Cut, Prompt};

/// What one row decodes by itself, after the common cuts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowPlan {
    /// The cuts of the row, in positions inside the row (the common ones first).
    pub cuts: Vec<Cut>,
    /// The segments after the last common cut, as ranges of the row: one per remaining cut and
    /// the tail, never empty (the tail holds the positions the head reads).
    pub own: Vec<Range<usize>>,
    /// Length of the row.
    pub len: usize,
}

/// The plan of one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPlan {
    /// The cuts every row has (in the same place, of the same kind, after the same ids).
    pub common: Vec<Cut>,
    /// The segments before the last common cut, one per common cut, as ranges of any row.
    pub shared: Vec<Range<usize>>,
    /// One plan per question, in question order.
    pub rows: Vec<RowPlan>,
}

fn bad(detail: String) -> Error {
    Error::Inference { detail }
}

/// The cuts of row `i`, checked: strictly increasing, inside the row, with every position the
/// head reads at or after the last one (so that all outputs come out of the last segment).
fn checked_cuts(prompt: &Prompt, i: usize) -> Result<Vec<Cut>> {
    let (Some(cuts), Some(len), Some(b)) = (
        prompt.row_cuts(i),
        prompt.row_len(i),
        prompt.branches().get(i),
    ) else {
        return Err(bad(format!("question {i} is not in the prompt")));
    };
    let mut before = 0usize;
    for c in &cuts {
        if c.pos <= before || c.pos >= len {
            return Err(bad(format!(
                "the template put a cut at {} in a row of {len} tokens (after {before}); cuts must be increasing and inside the row",
                c.pos
            )));
        }
        before = c.pos;
    }
    if let Some(last) = cuts.last() {
        let start = prompt.state().len();
        let reads = std::iter::once(b.decide()).chain(b.option_ends().iter().copied());
        if let Some(early) = reads.map(|p| start + p).find(|&p| p < last.pos) {
            return Err(bad(format!(
                "the template reads position {early} before its last cut at {}; outputs must come after the last cut",
                last.pos
            )));
        }
    }
    Ok(cuts)
}

/// `[0..c1) [c1..c2) … [cn..len)` for the cuts at `positions`, starting at `from`.
fn segments(from: usize, positions: impl Iterator<Item = usize>, len: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = from;
    for p in positions.chain(std::iter::once(len)) {
        out.push(start..p);
        start = p;
    }
    out
}

impl RequestPlan {
    /// Plan the request.
    ///
    /// # Errors
    /// [`Error::Inference`] if a template declared cuts that are not increasing and inside the
    /// row, or read a position before its last cut (a fault of the template, never a different
    /// result).
    pub fn new(prompt: &Prompt) -> Result<RequestPlan> {
        let n = prompt.branches().len();
        let mut rows_cuts = Vec::with_capacity(n);
        let mut rows_ids = Vec::with_capacity(n);
        for i in 0..n {
            rows_cuts.push(checked_cuts(prompt, i)?);
            rows_ids.push(prompt.row(i).unwrap_or_default());
        }

        // The common cuts: those of the first row that every other row has too.
        let mut common: Vec<Cut> = Vec::new();
        let mut shared: Vec<Range<usize>> = Vec::new();
        if let (Some(first_cuts), Some(first_ids)) = (rows_cuts.first(), rows_ids.first()) {
            let mut start = 0usize;
            for (j, cut) in first_cuts.iter().enumerate() {
                let same = rows_cuts.iter().zip(&rows_ids).all(|(cuts, ids)| {
                    cuts.get(j) == Some(cut)
                        && ids.get(start..cut.pos) == first_ids.get(start..cut.pos)
                });
                if !same {
                    break;
                }
                common.push(*cut);
                shared.push(start..cut.pos);
                start = cut.pos;
            }
        }

        let k = common.len();
        let shared_end = common.last().map_or(0, |c| c.pos);
        let rows = rows_cuts
            .into_iter()
            .zip(&rows_ids)
            .map(|(cuts, ids)| {
                let own = segments(shared_end, cuts.iter().skip(k).map(|c| c.pos), ids.len());
                RowPlan {
                    len: ids.len(),
                    cuts,
                    own,
                }
            })
            .collect();
        Ok(RequestPlan {
            common,
            shared,
            rows,
        })
    }

    /// Tokens before the last common cut: decoded once for the whole request.
    pub fn shared_len(&self) -> usize {
        self.common.last().map_or(0, |c| c.pos)
    }

    /// Tokens the model decodes for this request with no cache: the shared part once plus the
    /// own segments of every row.
    pub fn decoded_tokens(&self) -> usize {
        self.shared_len()
            + self
                .rows
                .iter()
                .map(|r| r.own.iter().map(|s| s.len()).sum::<usize>())
                .sum::<usize>()
    }

    /// The segments of row `i` in order: the shared ones, then its own.
    pub fn row_partition(&self, i: usize) -> Option<Vec<Range<usize>>> {
        let row = self.rows.get(i)?;
        Some(
            self.shared
                .iter()
                .cloned()
                .chain(row.own.iter().cloned())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::prompt::{Branch, CutKind};

    fn cut(pos: usize, kind: CutKind) -> Cut {
        Cut { pos, kind }
    }

    fn fixed(pos: usize) -> Cut {
        cut(pos, CutKind::Fixed)
    }

    fn branch(ids: Vec<u32>, cuts: Vec<Cut>) -> Branch {
        let decide = ids.len() - 1;
        Branch::with_cuts(ids, decide, Vec::new(), cuts)
    }

    fn lens(p: &[Range<usize>]) -> Vec<usize> {
        p.iter().map(Range::len).collect()
    }

    #[test]
    fn one_question_shares_all_its_cuts() {
        let p = Prompt::new(
            Vec::new(),
            vec![branch(vec![1, 2, 3, 4, 5, 6], vec![fixed(2), fixed(4)])],
        );
        let plan = RequestPlan::new(&p).unwrap();
        assert_eq!(plan.common, vec![fixed(2), fixed(4)]);
        assert_eq!(lens(&plan.shared), vec![2, 2]);
        assert_eq!(lens(&plan.rows[0].own), vec![2]);
        assert_eq!(plan.decoded_tokens(), 6);
    }

    #[test]
    fn rows_share_the_cuts_they_have_in_common_and_no_more() {
        // Same first cut and same ids before it; the second cut is in another place.
        let a = branch(vec![1, 2, 3, 4, 5, 6], vec![fixed(2), fixed(4)]);
        let b = branch(vec![1, 2, 3, 9, 9, 9, 9], vec![fixed(2), fixed(5)]);
        let plan = RequestPlan::new(&Prompt::new(Vec::new(), vec![a, b])).unwrap();
        assert_eq!(plan.common, vec![fixed(2)]);
        assert_eq!(plan.shared_len(), 2);
        // Row 0 still decodes [2..4) and [4..6) apart, row 1 [2..5) and [5..7): its own partition.
        assert_eq!(lens(&plan.rows[0].own), vec![2, 2]);
        assert_eq!(lens(&plan.rows[1].own), vec![3, 2]);
        assert_eq!(plan.decoded_tokens(), 2 + 4 + 5);
    }

    #[test]
    fn a_row_with_one_cut_less_leaves_only_what_all_rows_have() {
        let a = branch(
            vec![1, 2, 3, 4, 5, 6],
            vec![fixed(2), cut(4, CutKind::Content)],
        );
        let b = branch(vec![1, 2, 3, 4, 7, 8], vec![fixed(2)]);
        let c = branch(
            vec![1, 2, 3, 4, 5, 6],
            vec![fixed(2), cut(4, CutKind::Content)],
        );
        let plan = RequestPlan::new(&Prompt::new(Vec::new(), vec![a, b, c])).unwrap();
        assert_eq!(plan.common, vec![fixed(2)]);
        // Rows 0 and 2 keep their own cut at 4 (they decode [2..4) and [4..6) apart).
        assert_eq!(lens(&plan.rows[0].own), vec![2, 2]);
        assert_eq!(lens(&plan.rows[1].own), vec![4]);
    }

    #[test]
    fn same_position_but_different_ids_or_kind_is_not_common() {
        let a = branch(vec![1, 2, 3, 4], vec![fixed(2)]);
        let other_ids = branch(vec![1, 9, 3, 4], vec![fixed(2)]);
        let other_kind = branch(vec![1, 2, 3, 4], vec![cut(2, CutKind::Content)]);
        for b in [other_ids, other_kind] {
            let plan = RequestPlan::new(&Prompt::new(Vec::new(), vec![a.clone(), b])).unwrap();
            assert!(plan.common.is_empty());
            assert_eq!(plan.shared_len(), 0);
            // Each row decodes in the segments its own cut defines.
            assert_eq!(lens(&plan.rows[0].own), vec![2, 2]);
        }
    }

    #[test]
    fn the_state_end_of_kev_is_a_common_content_cut() {
        let p = Prompt::new(
            vec![7, 7, 7],
            vec![
                Branch::new(vec![1, 2], 1, Vec::new()),
                Branch::new(vec![3, 4, 5], 2, Vec::new()),
            ],
        );
        let plan = RequestPlan::new(&p).unwrap();
        assert_eq!(plan.common, vec![cut(3, CutKind::Content)]);
        assert_eq!(plan.decoded_tokens(), 3 + 2 + 3);
    }

    #[test]
    fn rows_without_cuts_are_decoded_whole() {
        let p = Prompt::new(
            Vec::new(),
            vec![
                Branch::new(vec![1, 2, 3], 2, Vec::new()),
                Branch::new(vec![1, 2, 4], 2, Vec::new()),
            ],
        );
        let plan = RequestPlan::new(&p).unwrap();
        assert!(plan.common.is_empty() && plan.shared.is_empty());
        assert_eq!(plan.decoded_tokens(), 6);
    }

    #[test]
    fn no_questions_is_an_empty_plan() {
        let plan = RequestPlan::new(&Prompt::new(vec![1, 2], Vec::new())).unwrap();
        assert!(plan.rows.is_empty());
        assert_eq!(plan.decoded_tokens(), 0);
    }

    #[test]
    fn outputs_before_the_last_cut_are_an_error() {
        let ids = vec![1, 2, 3, 4, 5];
        // The head reads position 1, the cut is at 3.
        let b = Branch::with_cuts(ids, 1, Vec::new(), vec![fixed(3)]);
        let err = RequestPlan::new(&Prompt::new(Vec::new(), vec![b])).unwrap_err();
        assert!(matches!(err, Error::Inference { .. }), "{err}");
        assert!(err.to_string().contains("after the last cut"), "{err}");
        // An option end counts too.
        let b = Branch::with_cuts(vec![1, 2, 3, 4, 5], 4, vec![1], vec![fixed(3)]);
        assert!(RequestPlan::new(&Prompt::new(Vec::new(), vec![b])).is_err());
    }

    #[test]
    fn cuts_that_are_not_increasing_or_inside_the_row_are_an_error() {
        for cuts in [
            vec![fixed(3), fixed(2)],
            vec![fixed(2), fixed(2)],
            vec![fixed(0)],
            vec![fixed(5)],
        ] {
            let b = Branch::with_cuts(vec![1, 2, 3, 4, 5], 4, Vec::new(), cuts.clone());
            let r = RequestPlan::new(&Prompt::new(Vec::new(), vec![b]));
            assert!(matches!(r, Err(Error::Inference { .. })), "{cuts:?}");
        }
    }

    /// A random prompt over a tiny alphabet, so rows often share ids and cuts.
    fn prompts() -> impl Strategy<Value = Prompt> {
        let state = proptest::collection::vec(0u32..3, 0..4);
        let branch = (3usize..10).prop_flat_map(|len| {
            (
                proptest::collection::vec(0u32..3, len),
                proptest::collection::btree_set(1..len - 1, 0..4),
                proptest::collection::vec(any::<bool>(), 4),
            )
                .prop_map(|(ids, positions, kinds)| {
                    let cuts = positions
                        .into_iter()
                        .zip(kinds)
                        .map(|(pos, c)| cut(pos, if c { CutKind::Fixed } else { CutKind::Content }))
                        .collect();
                    branch(ids, cuts)
                })
        });
        (state, proptest::collection::vec(branch, 0..7))
            .prop_map(|(state, branches)| Prompt::new(state, branches))
    }

    fn config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_seed: proptest::test_runner::RngSeed::Fixed(7),
            failure_persistence: None,
            ..ProptestConfig::default()
        }
    }

    /// The boundaries a row would have decoded alone: its own cuts and its length.
    fn alone(p: &Prompt, i: usize) -> Vec<Range<usize>> {
        let single = Prompt::new(p.state().to_vec(), vec![p.branches()[i].clone()]);
        let plan = RequestPlan::new(&single).unwrap();
        plan.row_partition(0).unwrap()
    }

    proptest! {
        #![proptest_config(config())]

        #[test]
        fn prop_common_cuts_are_the_shared_partition(p in prompts()) {
            let plan = RequestPlan::new(&p).unwrap();
            for i in 0..p.branches().len() {
                let part = plan.row_partition(i).unwrap();
                // Contiguous, covers the row exactly.
                let mut at = 0;
                for s in &part {
                    prop_assert_eq!(s.start, at);
                    prop_assert!(s.end > s.start);
                    at = s.end;
                }
                prop_assert_eq!(at, p.row_len(i).unwrap());
                // And it is the partition the row has alone: it never depends on the others.
                prop_assert_eq!(part, alone(&p, i));
            }
        }

        #[test]
        fn prop_decoded_tokens_are_shared_once_plus_own(p in prompts()) {
            let plan = RequestPlan::new(&p).unwrap();
            let own: usize = plan.rows.iter().map(|r| r.own.iter().map(Range::len).sum::<usize>()).sum();
            prop_assert_eq!(plan.decoded_tokens(), plan.shared_len() + own);
            // Never more than the rows decoded one by one, never less than the longest row.
            let total: usize = (0..p.branches().len()).map(|i| p.row_len(i).unwrap()).sum();
            prop_assert!(plan.decoded_tokens() <= total);
            // Repeating a row does not change what is shared (the shared part does not depend on k).
            if let Some(first) = p.branches().first() {
                let mut more = p.branches().to_vec();
                more.push(first.clone());
                more.push(first.clone());
                let again = RequestPlan::new(&Prompt::new(p.state().to_vec(), more)).unwrap();
                prop_assert_eq!(again.common, plan.common);
            }
        }

        #[test]
        fn prop_order_independent(p in prompts()) {
            let n = p.branches().len();
            let plan = RequestPlan::new(&p).unwrap();
            let reversed = Prompt::new(
                p.state().to_vec(),
                p.branches().iter().rev().cloned().collect(),
            );
            let rev = RequestPlan::new(&reversed).unwrap();
            prop_assert_eq!(&plan.common, &rev.common);
            for i in 0..n {
                prop_assert_eq!(plan.row_partition(i), rev.row_partition(n - 1 - i));
            }
        }
    }
}
