//! Test helpers: a scorer that answers from a table, so the numbers of an evaluation can be
//! worked out by hand, and a few builders of records. Only compiled for tests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, Score, State, StateValue, TextEntry, YesNo,
};
use crate::scorer::{Calibration, Scorer};

use super::dataset::{Dataset, LabelledRecord, RawLabel};

/// Logits chosen by the text of the state: `table[state] = one list per question, in order`.
/// A state that is not in the table gets an error (a model that refuses it).
pub struct TableScorer {
    pub calibration: Calibration,
    pub table: HashMap<String, Vec<Vec<f64>>>,
    pub calls: AtomicUsize,
    /// States for which `score` fails with this error.
    pub fail_on: HashMap<String, Error>,
}

impl TableScorer {
    pub fn new(calibration: Calibration, rows: Vec<(&str, Vec<Vec<f64>>)>) -> Self {
        TableScorer {
            calibration,
            table: rows.into_iter().map(|(s, l)| (s.to_string(), l)).collect(),
            calls: AtomicUsize::new(0),
            fail_on: HashMap::new(),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Scorer for TableScorer {
    fn calibration(&self) -> Calibration {
        self.calibration
    }

    fn score(&self, state: &State, _questions: &Questions) -> Result<Vec<Vec<f64>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let key = state.canonical_text();
        if let Some(e) = self.fail_on.get(key) {
            return Err(e.clone());
        }
        self.table.get(key).cloned().ok_or(Error::Inference {
            detail: format!("no logits for the state {key:?}"),
        })
    }
}

pub fn instr(s: &str) -> TextEntry {
    TextEntry::text(s).unwrap()
}

pub fn state(text: &str) -> State {
    State::new(StateValue::string(text)).unwrap()
}

pub fn choice(keys: &[&str]) -> Question {
    Question::from(
        Choice::new(
            instr("Which one?"),
            keys.iter().map(|k| ChoiceOption::key(*k)).collect(),
        )
        .unwrap(),
    )
}

pub fn yes_no() -> Question {
    Question::from(YesNo::new(instr("Is it?"), None, None).unwrap())
}

pub fn score(levels: usize) -> Question {
    Question::from(
        Score::new(
            instr("How much?"),
            (0..levels).map(|i| instr(&format!("level {i}"))).collect(),
        )
        .unwrap(),
    )
}

/// A record with the given questions `(name, question, label)`.
pub fn record(
    place: &str,
    state_text: &str,
    group: Option<&str>,
    items: Vec<(&str, Question, RawLabel)>,
) -> LabelledRecord {
    let labels: Vec<(String, RawLabel)> = items
        .iter()
        .map(|(n, _, l)| (n.to_string(), l.clone()))
        .collect();
    let questions = Questions::new(items.into_iter().map(|(n, q, _)| (n, q))).unwrap();
    LabelledRecord::build(
        place,
        state(state_text),
        questions,
        &labels,
        None,
        group.map(str::to_string),
    )
    .unwrap()
}

pub fn dataset(records: Vec<LabelledRecord>) -> Dataset {
    Dataset::new(records).unwrap()
}

/// A seeded source of logits and labels for the larger tests.
pub struct Gen(pub crate::rng::SplitMix64);

impl Gen {
    pub fn new(seed: u64) -> Self {
        Gen(crate::rng::SplitMix64::new(seed))
    }

    pub fn uniform(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn gaussian(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }

    pub fn categorical(&mut self, p: &[f64]) -> usize {
        let mut u = self.uniform();
        for (i, q) in p.iter().enumerate() {
            if u < *q {
                return i;
            }
            u -= q;
        }
        p.len() - 1
    }
}

/// `n` records with one choice question of `k` options each. The logits are Gaussian with
/// spread `sigma` and the label is drawn from `softmax(logits / true_t)`. Record `i` belongs to
/// group `i / group_size`. Returns the scorer (current temperature 1, not calibrated), the
/// dataset and the logits of each record.
pub fn generated(
    seed: u64,
    n: usize,
    k: usize,
    sigma: f64,
    true_t: f64,
    group_size: usize,
) -> (TableScorer, Dataset, Vec<Vec<f64>>) {
    let mut source = Gen::new(seed);
    let keys: Vec<String> = (0..k).map(|i| format!("k{i}")).collect();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let mut scorer = TableScorer::new(Calibration::uncalibrated(), vec![]);
    let mut records = Vec::with_capacity(n);
    let mut all = Vec::with_capacity(n);
    for i in 0..n {
        let z: Vec<f64> = (0..k).map(|_| sigma * source.gaussian()).collect();
        let p = crate::calibration::calibrated_softmax(&z, true_t).unwrap();
        let label = source.categorical(&p);
        let state_text = format!("s{i}");
        scorer.table.insert(state_text.clone(), vec![z.clone()]);
        records.push(record(
            &format!("record {i}"),
            &state_text,
            Some(&format!("g{}", i / group_size.max(1))),
            vec![("q", choice(&key_refs), RawLabel::Str(keys[label].clone()))],
        ));
        all.push(z);
    }
    (scorer, dataset(records), all)
}
