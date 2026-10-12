//! The segmented path of [`ModelScorer`]: what the rows of a request share is decoded once (or
//! restored from a saved prefix), then every question runs in the segments its own cuts define.
//!
//! For a row the engine always sees the same sequence of decodes, `[0..c1) [c1..c2) â€¦ [cn..len)`,
//! whether a prefix was decoded here, restored from memory or shared with the other rows of the
//! request: that is what keeps the numbers identical with the memory cold, warm or empty.

use std::ops::Range;

use super::{ModelScorer, name_the_question};
use crate::control::Control;
use crate::engine::OutputSpec;
use crate::error::{Error, Result};
use crate::prefix::{Entry, KeyHasher, PrefixKey, RequestPlan, admit};
use crate::prompt::Prompt;
use crate::schema::Questions;

fn outside(what: &str) -> Error {
    Error::Inference {
        detail: format!("{what} is outside the row"),
    }
}

/// A head names positions inside the branch; the engine counts from the first token after the
/// shared part.
fn rebase(spec: &OutputSpec, state_len: usize, shared_end: usize) -> Result<OutputSpec> {
    let shift = |p: usize| {
        (state_len + p)
            .checked_sub(shared_end)
            .ok_or_else(|| Error::Inference {
                detail: format!("output position {p} lies inside the shared part of the request"),
            })
    };
    Ok(match spec {
        OutputSpec::Hidden { position } => OutputSpec::Hidden {
            position: shift(*position)?,
        },
        OutputSpec::Logits { position, ids } => OutputSpec::Logits {
            position: shift(*position)?,
            ids: ids.clone(),
        },
    })
}

impl ModelScorer {
    /// Raw logits of every question, with the shared part decoded once or restored.
    pub(super) fn score_segmented(
        &self,
        questions: &Questions,
        prompt: &Prompt,
        control: &dyn Control,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let plan = RequestPlan::new(prompt)?;
        let shared_end = plan.shared_len();
        let state_len = prompt.state().len();
        let rows = prompt.branches().len();
        let mut session = self.forward.begin_empty_with(control)?;
        let mut decoded = 0usize;
        // Where the rows start: the end of the shared part (a test mutant moves it back).
        let mut base_end = shared_end;

        if let Some(first) = prompt.row(0) {
            // The key of the prefix that ends at each common cut, in one pass over the ids.
            let mut hasher = KeyHasher::new(&self.fingerprint);
            let mut keys: Vec<PrefixKey> = Vec::with_capacity(plan.shared.len());
            for segment in &plan.shared {
                hasher.update(
                    first
                        .get(segment.clone())
                        .ok_or_else(|| outside("a segment"))?,
                );
                keys.push(hasher.key());
            }

            // The longest saved prefix, if any: one lookup for the whole request.
            let mut restored = 0usize;
            if !keys.is_empty() {
                let mut found = None;
                {
                    let mut store = self.store();
                    for (j, key) in keys.iter().enumerate().rev() {
                        let end = plan.shared.get(j).map_or(0, |r| r.end);
                        let ids = first.get(..end).ok_or_else(|| outside("a prefix"))?;
                        if let Some(entry) = store.get(key, ids, &self.fingerprint) {
                            found = Some((j, entry));
                            break;
                        }
                    }
                    if found.is_some() {
                        store.counters.hits += 1;
                    } else {
                        store.counters.misses += 1;
                    }
                }
                if let Some((j, entry)) = found {
                    if let Err(e) = session.restore(&entry.snapshot) {
                        // A prefix that cannot be restored is not served again.
                        if let Some(key) = keys.get(j) {
                            self.store().remove(key);
                        }
                        return Err(e);
                    }
                    restored = j + 1;
                }
            }

            let rest = plan.shared.get(restored..).unwrap_or_default();
            if self.merges_shared() && plan.shared.len() > 1 {
                // Test mutant: the shared prefix is one segment.
                let from = rest.first().map_or(0, |r| r.start);
                let to = rest.last().map_or(0, |r| r.end);
                let merged = first.get(from..to).ok_or_else(|| outside("a segment"))?;
                session.extend(merged)?;
                decoded += merged.len();
            } else if self.merges_after_restore() && restored > 0 && !rest.is_empty() {
                // Test mutant: after a hit, what is left of the shared part is not decoded as its
                // own segments but together with the first segment of every row.
                base_end = plan.shared.get(restored - 1).map_or(0, |r| r.end);
            } else {
                for (offset, segment) in rest.iter().enumerate() {
                    let j = restored + offset;
                    let ids = first
                        .get(segment.clone())
                        .ok_or_else(|| outside("a segment"))?;
                    control.checkpoint()?;
                    session.extend(ids)?;
                    decoded += ids.len();
                    self.save_if_worth_it(
                        &mut *session,
                        &plan,
                        &keys,
                        j,
                        first.get(..segment.end),
                        rows,
                    )?;
                }
            }
        }

        let mut logits = Vec::with_capacity(questions.len());
        let mut own_tokens = 0usize;
        for (i, ((name, question), branch)) in questions.iter().zip(prompt.branches()).enumerate() {
            control.checkpoint()?;
            let (Some(row), Some(own)) = (prompt.row(i), plan.rows.get(i)) else {
                return Err(Error::Inference {
                    detail: format!("question {i} has no row"),
                });
            };
            let ranges: Vec<Range<usize>> = match own.own.split_first() {
                Some((head, tail)) if base_end != shared_end => std::iter::once(base_end..head.end)
                    .chain(tail.iter().cloned())
                    .collect(),
                _ => own.own.clone(),
            };
            let segments = ranges
                .iter()
                .map(|r| row.get(r.clone()).ok_or_else(|| outside("a segment")))
                .collect::<Result<Vec<&[u32]>>>()?;
            own_tokens += segments.iter().map(|s| s.len()).sum::<usize>();
            let specs = self
                .head
                .outputs(question, branch)
                .iter()
                .map(|s| rebase(s, state_len, base_end))
                .collect::<Result<Vec<_>>>()?;
            let outputs = session
                .row_segments(&segments, &specs)
                .map_err(|e| name_the_question(e, name))?;
            logits.push((name.to_string(), self.head.score(question, &outputs)?));
        }

        let total: usize = (0..rows).filter_map(|i| prompt.row_len(i)).sum();
        let decoded = decoded + own_tokens;
        let mut store = self.store();
        store.counters.tokens_decoded += decoded as u64;
        store.counters.tokens_reused += total.saturating_sub(decoded) as u64;
        Ok(logits)
    }

    /// Save the state at common cut `j` if the admission rule says it is worth it.
    fn save_if_worth_it(
        &self,
        session: &mut dyn crate::engine::Session,
        plan: &RequestPlan,
        keys: &[PrefixKey],
        j: usize,
        ids: Option<&[u32]>,
        rows: usize,
    ) -> Result<()> {
        let (Some(cut), Some(key), Some(ids)) = (plan.common.get(j), keys.get(j), ids) else {
            return Ok(());
        };
        let worth = {
            let mut store = self.store();
            !store.contains(key)
                && store.limit_bytes() > 0
                && admit(cut.kind, cut.pos, rows, key, &mut store.seen)
        };
        if !worth {
            return Ok(());
        }
        let snapshot = session.snapshot()?;
        self.store().insert(
            *key,
            Entry {
                ids: ids.into(),
                snapshot,
                fingerprint: self.fingerprint,
            },
        );
        Ok(())
    }
}
