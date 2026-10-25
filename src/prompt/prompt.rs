//! The prompt as ids: the state and one branch per question (spec 006a section 5).
//!
//! Nothing here holds text. The **packed** prompt is the state followed by every branch (what
//! Kev's golden calls `ids` and `usage.input_tokens` counts); the **row** of a question is the
//! state followed by that question's branch (what the model actually runs, and what
//! `max_context` limits).

/// What a cut point depends on (it decides whether a cached prefix is worth keeping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutKind {
    /// Depends only on the task: the system block and the head of the user text.
    Fixed,
    /// Depends on the text the user sent: the end of the state.
    Content,
}

/// A place where a template says "the row may be split here and the prefix kept".
///
/// A row is always decoded in the segments its own cuts define, so the numbers a row gets do not
/// depend on which other rows travel with it or on what was cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    /// Position: the number of tokens before the cut.
    pub pos: usize,
    /// What the prefix before the cut depends on.
    pub kind: CutKind,
}

/// One question's tokens and the positions the head reads from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    ids: Vec<u32>,
    decide: usize,
    option_ends: Vec<usize>,
    cuts: Vec<Cut>,
    keys: Vec<Vec<u32>>,
}

impl Branch {
    pub(crate) fn new(ids: Vec<u32>, decide: usize, option_ends: Vec<usize>) -> Self {
        Branch::with_cuts(ids, decide, option_ends, Vec::new())
    }

    /// A branch with cut points, as positions **inside the branch** (strictly increasing, each
    /// `0 < pos < len`; the request plan checks this and that the outputs lie after the last cut).
    pub(crate) fn with_cuts(
        ids: Vec<u32>,
        decide: usize,
        option_ends: Vec<usize>,
        cuts: Vec<Cut>,
    ) -> Self {
        Branch {
            ids,
            decide,
            option_ends,
            cuts,
            keys: Vec::new(),
        }
    }

    /// The token ids of each option key, in option order (spec 010): what a key-scoring head
    /// reads as the continuation of the branch. Empty for the templates that do not use keys.
    #[must_use]
    pub(crate) fn with_keys(mut self, keys: Vec<Vec<u32>>) -> Self {
        self.keys = keys;
        self
    }

    /// The token ids of each option key, in option order (empty if the template has none).
    pub fn keys(&self) -> &[Vec<u32>] {
        &self.keys
    }

    /// The cut points of the branch, as positions inside the branch.
    pub fn cuts(&self) -> &[Cut] {
        &self.cuts
    }

    /// The tokens of the branch.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True for a branch without tokens.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Index (inside the branch) of the token the query is read from.
    pub fn decide(&self) -> usize {
        self.decide
    }

    /// Index (inside the branch) of the closing token of each option, in option order.
    pub fn option_ends(&self) -> &[usize] {
        &self.option_ends
    }
}

/// A prompt: shared state tokens and one branch per question, in question order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    state: Vec<u32>,
    branches: Vec<Branch>,
}

impl Prompt {
    pub(crate) fn new(state: Vec<u32>, branches: Vec<Branch>) -> Self {
        Prompt { state, branches }
    }

    /// The state tokens (opening delimiter included).
    pub fn state(&self) -> &[u32] {
        &self.state
    }

    /// The branches, one per question.
    pub fn branches(&self) -> &[Branch] {
        &self.branches
    }

    /// State followed by every branch.
    pub fn packed_ids(&self) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.packed_len());
        ids.extend_from_slice(&self.state);
        for b in &self.branches {
            ids.extend_from_slice(&b.ids);
        }
        ids
    }

    /// Length of the packed prompt: `usage.input_tokens`.
    pub fn packed_len(&self) -> usize {
        self.state.len() + self.branches.iter().map(Branch::len).sum::<usize>()
    }

    /// Where branch `i` starts inside the packed prompt.
    pub fn branch_start(&self, i: usize) -> Option<usize> {
        if i >= self.branches.len() {
            return None;
        }
        Some(self.state.len() + self.branches.iter().take(i).map(Branch::len).sum::<usize>())
    }

    /// Length of the row of question `i`: state plus its branch.
    pub fn row_len(&self, i: usize) -> Option<usize> {
        self.branches.get(i).map(|b| self.state.len() + b.len())
    }

    /// The cut points of the row of question `i`, as positions inside the row: the end of the
    /// state (when there is one, `Content`) followed by the cuts of the branch shifted by it.
    pub fn row_cuts(&self, i: usize) -> Option<Vec<Cut>> {
        let b = self.branches.get(i)?;
        let shift = self.state.len();
        let mut cuts = Vec::with_capacity(b.cuts.len() + 1);
        if shift > 0 {
            cuts.push(Cut {
                pos: shift,
                kind: CutKind::Content,
            });
        }
        cuts.extend(b.cuts.iter().map(|c| Cut {
            pos: c.pos + shift,
            kind: c.kind,
        }));
        Some(cuts)
    }

    /// The row of question `i`: state followed by its branch.
    pub fn row(&self, i: usize) -> Option<Vec<u32>> {
        let b = self.branches.get(i)?;
        let mut row = Vec::with_capacity(self.state.len() + b.len());
        row.extend_from_slice(&self.state);
        row.extend_from_slice(&b.ids);
        Some(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Prompt {
        Prompt::new(
            vec![1, 2],
            vec![
                Branch::new(vec![10, 11, 12], 2, vec![1]),
                Branch::new(vec![20, 21], 1, vec![0]),
            ],
        )
    }

    #[test]
    fn prompt_holds_only_ids_and_positions() {
        // The only data are `u32` ids and `usize` positions: there is no field that could hold text.
        let p = sample();
        let _: &[u32] = p.state();
        let _: &[u32] = p.branches()[0].ids();
        let _: usize = p.branches()[0].decide();
        let _: &[usize] = p.branches()[0].option_ends();
        assert_eq!(p.packed_ids(), vec![1, 2, 10, 11, 12, 20, 21]);
        assert_eq!(p.packed_len(), 7);
    }

    #[test]
    fn rows_and_offsets() {
        let p = sample();
        assert_eq!(p.branch_start(0), Some(2));
        assert_eq!(p.branch_start(1), Some(5));
        assert_eq!(p.branch_start(2), None);
        assert_eq!(p.row_len(1), Some(4));
        assert_eq!(p.row(1), Some(vec![1, 2, 20, 21]));
        assert_eq!(p.row(2), None);
    }

    #[test]
    fn row_cuts_are_the_state_end_then_the_branch_cuts_shifted() {
        let fixed = |pos| Cut {
            pos,
            kind: CutKind::Fixed,
        };
        let p = Prompt::new(
            vec![1, 2, 3],
            vec![
                Branch::with_cuts(vec![10, 11, 12, 13], 3, Vec::new(), vec![fixed(2)]),
                Branch::new(vec![20, 21], 1, Vec::new()),
            ],
        );
        let content = |pos| Cut {
            pos,
            kind: CutKind::Content,
        };
        assert_eq!(p.row_cuts(0), Some(vec![content(3), fixed(5)]));
        assert_eq!(p.row_cuts(1), Some(vec![content(3)]));
        assert_eq!(p.row_cuts(2), None);
        // Without a state (chatml) only the cuts of the branch are there, unshifted.
        let q = Prompt::new(
            Vec::new(),
            vec![Branch::with_cuts(
                vec![1, 2, 3],
                2,
                Vec::new(),
                vec![fixed(1)],
            )],
        );
        assert_eq!(q.row_cuts(0), Some(vec![fixed(1)]));
    }
}
