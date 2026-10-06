//! A deterministic fake scorer. **It is not a model**: it exists to develop and test everything
//! above the model (API, calibration, batching) and to give users something runnable.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::error::Result;
use crate::schema::{Questions, State};
use crate::scorer::{Calibration, Scorer};

/// A deliberate failure injected into the mock output, to exercise the fail-closed checks.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum Fault {
    /// Replace one logit with NaN.
    Nan { question: String, option: usize },
    /// Replace one logit with +infinity.
    PosInf { question: String, option: usize },
    /// Replace one logit with -infinity.
    NegInf { question: String, option: usize },
    /// Return the wrong number of logits for one question.
    WrongOptionCount { question: String, count: usize },
    /// Return logits for the wrong number of questions.
    WrongQuestionCount { count: usize },
}

impl Fault {
    /// NaN at (`question`, `option`).
    pub fn nan(question: &str, option: usize) -> Self {
        Fault::Nan {
            question: question.to_string(),
            option,
        }
    }
    /// +infinity at (`question`, `option`).
    pub fn pos_inf(question: &str, option: usize) -> Self {
        Fault::PosInf {
            question: question.to_string(),
            option,
        }
    }
    /// -infinity at (`question`, `option`).
    pub fn neg_inf(question: &str, option: usize) -> Self {
        Fault::NegInf {
            question: question.to_string(),
            option,
        }
    }
    /// `count` logits for `question` instead of one per option.
    pub fn wrong_option_count(question: &str, count: usize) -> Self {
        Fault::WrongOptionCount {
            question: question.to_string(),
            count,
        }
    }
    /// Logits for `count` questions instead of one per question.
    pub fn wrong_question_count(count: usize) -> Self {
        Fault::WrongQuestionCount { count }
    }
}

#[derive(Debug, Clone)]
enum Mode {
    Seeded(u64),
    Scripted(Vec<Vec<f64>>),
}

/// The deterministic fake scorer.
#[derive(Debug)]
pub struct MockScorer {
    mode: Mode,
    calibration: Calibration,
    latency: Duration,
    faults: Vec<Fault>,
    calls: AtomicU64,
}

impl Clone for MockScorer {
    /// A clone has its own call counter, starting at zero.
    fn clone(&self) -> Self {
        MockScorer {
            mode: self.mode.clone(),
            calibration: self.calibration,
            latency: self.latency,
            faults: self.faults.clone(),
            calls: AtomicU64::new(0),
        }
    }
}

/// FNV-1a over length-prefixed fields: stable on every OS, architecture and Rust version
/// (unlike `std`'s hasher).
struct Mix(u64);

impl Mix {
    fn new(seed: u64) -> Self {
        let mut m = Mix(0xcbf2_9ce4_8422_2325);
        m.bytes(&seed.to_le_bytes());
        m
    }
    fn raw(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.raw(&(bytes.len() as u64).to_le_bytes());
        self.raw(bytes);
    }
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
}

/// SplitMix64 step.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl MockScorer {
    /// Logits that are a deterministic function of (seed, canonical state text, question kind,
    /// instructions, option texts) in `[-4, 4]`. They do **not** depend on the question's name,
    /// its position, or the other questions.
    pub fn new(seed: u64) -> Self {
        MockScorer {
            mode: Mode::Seeded(seed),
            calibration: Calibration::uncalibrated(),
            latency: Duration::ZERO,
            faults: Vec::new(),
            calls: AtomicU64::new(0),
        }
    }

    /// Returns exactly these logits, whatever they are (NaN and infinities included).
    pub fn scripted(logits: Vec<Vec<f64>>) -> Self {
        MockScorer {
            mode: Mode::Scripted(logits),
            ..MockScorer::new(0)
        }
    }

    /// The same scorer with another calibration. The original is untouched.
    #[must_use]
    pub fn with_calibration(&self, calibration: Calibration) -> Self {
        MockScorer {
            calibration,
            ..self.clone()
        }
    }

    /// The same scorer sleeping `latency` inside each call (no lock held by the caller).
    #[must_use]
    pub fn with_latency(&self, latency: Duration) -> Self {
        MockScorer {
            latency,
            ..self.clone()
        }
    }

    /// The same scorer with one more injected fault.
    #[must_use]
    pub fn with_fault(&self, fault: Fault) -> Self {
        let mut s = self.clone();
        s.faults.push(fault);
        s
    }

    /// How many times `score` has been called on this instance.
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Scorer for MockScorer {
    fn calibration(&self) -> Calibration {
        self.calibration
    }

    fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        let mut out: Vec<Vec<f64>> = match &self.mode {
            Mode::Scripted(rows) => rows.clone(),
            Mode::Seeded(seed) => questions
                .iter()
                .map(|(_, q)| {
                    let mut mix = Mix::new(*seed);
                    mix.str(state.canonical_text());
                    mix.str(&format!("{:?}", q.kind()));
                    mix.str(q.instructions().rendered());
                    for t in q.option_texts() {
                        mix.str(t);
                    }
                    let mut rng = mix.0;
                    (0..q.option_count())
                        .map(|_| {
                            (splitmix(&mut rng) >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0
                        })
                        .collect()
                })
                .collect(),
        };
        let position = |name: &str| questions.iter().position(|(n, _)| n == name);
        for fault in &self.faults {
            match fault {
                Fault::Nan { question, option }
                | Fault::PosInf { question, option }
                | Fault::NegInf { question, option } => {
                    let replacement = match fault {
                        Fault::Nan { .. } => f64::NAN,
                        Fault::PosInf { .. } => f64::INFINITY,
                        _ => f64::NEG_INFINITY,
                    };
                    if let Some(slot) = position(question)
                        .and_then(|i| out.get_mut(i))
                        .and_then(|row| row.get_mut(*option))
                    {
                        *slot = replacement;
                    }
                }
                Fault::WrongOptionCount { question, count } => {
                    if let Some(row) = position(question).and_then(|i| out.get_mut(i)) {
                        row.resize(*count, 0.0);
                    }
                }
                Fault::WrongQuestionCount { count } => out.resize(*count, vec![0.0]),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Choice, ChoiceOption, Question, StateValue, TextEntry};

    fn state(text: &str) -> State {
        State::new(StateValue::string(text)).unwrap()
    }

    fn qs(names: &[&str], instructions: &str) -> Questions {
        Questions::new(names.iter().map(|n| {
            (
                *n,
                Question::from(
                    Choice::new(
                        TextEntry::text(instructions).unwrap(),
                        vec![
                            ChoiceOption::key("a"),
                            ChoiceOption::key("b"),
                            ChoiceOption::key("c"),
                        ],
                    )
                    .unwrap(),
                ),
            )
        }))
        .unwrap()
    }

    #[test]
    fn same_seed_same_logits_other_seed_other_logits() {
        let a = MockScorer::new(1)
            .score(&state("s"), &qs(&["q"], "i"))
            .unwrap();
        let b = MockScorer::new(1)
            .score(&state("s"), &qs(&["q"], "i"))
            .unwrap();
        let c = MockScorer::new(2)
            .score(&state("s"), &qs(&["q"], "i"))
            .unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(
            a.iter()
                .flatten()
                .all(|x| x.is_finite() && (-4.0..=4.0).contains(x))
        );
    }

    #[test]
    fn logits_are_pinned_across_platforms() {
        // Literals pin the hash and the PRNG: they must not change between OS, architectures or
        // Rust versions.
        let row = MockScorer::new(0)
            .score(&state("hello"), &qs(&["q"], "pick one"))
            .unwrap();
        assert_eq!(row.len(), 1);
        let got: Vec<String> = row[0].iter().map(|x| format!("{x:.12}")).collect();
        assert_eq!(got, PINNED_0, "{got:?}");
    }

    const PINNED_0: [&str; 3] = ["1.192414304466", "-1.599415484648", "3.348417950250"];

    #[test]
    fn isolation_from_name_position_and_other_questions() {
        let alone = MockScorer::new(5)
            .score(&state("s"), &qs(&["x"], "i"))
            .unwrap();
        let renamed = MockScorer::new(5)
            .score(&state("s"), &qs(&["other"], "i"))
            .unwrap();
        let with_others = MockScorer::new(5)
            .score(&state("s"), &qs(&["y", "x"], "i"))
            .unwrap();
        assert_eq!(alone, renamed);
        assert_eq!(alone[0], with_others[0]);
        assert_eq!(alone[0], with_others[1]);
    }

    #[test]
    fn with_methods_do_not_modify_the_original_and_calls_count() {
        let base = MockScorer::new(3);
        let faulty = base.with_fault(Fault::nan("q", 0));
        let s = state("s");
        let q = qs(&["q"], "i");
        assert!(base.score(&s, &q).unwrap()[0][0].is_finite());
        assert!(faulty.score(&s, &q).unwrap()[0][0].is_nan());
        assert_eq!((base.calls(), faulty.calls()), (1, 1));
        assert!(!base.calibration().calibrated());
    }
}
