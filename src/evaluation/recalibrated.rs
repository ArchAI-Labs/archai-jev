//! `Recalibrated`: the same model with another temperature (spec 017 §3.6).
//!
//! `Jev.with_temperature` builds one of these over the scorer of the model. Everything but the
//! calibration is the model's own — the weights and the memory of reused prefixes are shared,
//! not copied — and the original scorer is never touched, so both can be used from several
//! threads at once.

use std::sync::Arc;

use crate::control::Control;
use crate::error::Result;
use crate::schema::{Questions, State};
use crate::scorer::{CacheInfo, Calibration, Scorer};

/// A scorer with its calibration replaced.
pub struct Recalibrated {
    inner: Arc<dyn Scorer>,
    calibration: Calibration,
}

impl Recalibrated {
    /// `inner` with `calibration` (a calibration given by the user, so declared).
    pub fn new(inner: Arc<dyn Scorer>, calibration: Calibration) -> Self {
        Recalibrated { inner, calibration }
    }
}

impl Scorer for Recalibrated {
    fn calibration(&self) -> Calibration {
        self.calibration
    }

    fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>> {
        self.inner.score(state, questions)
    }

    fn score_with(
        &self,
        state: &State,
        questions: &Questions,
        control: &dyn Control,
    ) -> Result<Vec<Vec<f64>>> {
        self.inner.score_with(state, questions, control)
    }

    fn cache_info(&self) -> Option<CacheInfo> {
        self.inner.cache_info()
    }

    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    fn reset_cache_counters(&self) {
        self.inner.reset_cache_counters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answers::Answer;
    use crate::ask::ask;
    use crate::evaluation::testing::{choice, instr, state};
    use crate::schema::{Question, YesNo};

    fn questions() -> Questions {
        Questions::new(vec![
            ("team", choice(&["a", "b", "c"])),
            (
                "ok",
                Question::from(YesNo::new(instr("Ok?"), None, None).unwrap()),
            ),
        ])
        .unwrap()
    }

    fn base() -> Arc<dyn Scorer> {
        Arc::new(crate::evaluation::testing::TableScorer::new(
            Calibration::new(1.0, false).unwrap(),
            vec![("s", vec![vec![2.0, 1.0, 0.0], vec![0.0, 1.5]])],
        ))
    }

    #[test]
    fn only_the_calibration_changes() {
        let inner = base();
        let tuned = Recalibrated::new(Arc::clone(&inner), Calibration::new(2.5, true).unwrap());
        assert_eq!(tuned.calibration().temperature(), 2.5);
        assert!(tuned.calibration().calibrated());
        let q = questions();
        assert_eq!(
            tuned.score(&state("s"), &q).unwrap(),
            inner.score(&state("s"), &q).unwrap()
        );
        // the probabilities are softmax(raw logits / 2.5)
        let answers = ask(&tuned, &state("s"), &q).unwrap();
        let Some(Answer::Choice(c)) = answers.get("team") else {
            panic!("not a choice");
        };
        let expected = crate::calibration::calibrated_softmax(&[2.0, 1.0, 0.0], 2.5).unwrap();
        let got: Vec<f64> = c.probabilities().iter().map(|(_, p)| *p).collect();
        assert_eq!(got, expected);
        assert!(c.calibrated());
    }

    #[test]
    fn original_unchanged() {
        let inner = base();
        let q = questions();
        let before = ask(inner.as_ref(), &state("s"), &q).unwrap();
        let _tuned = Recalibrated::new(Arc::clone(&inner), Calibration::new(9.0, true).unwrap());
        let after = ask(inner.as_ref(), &state("s"), &q).unwrap();
        assert_eq!(before, after);
        assert!(!inner.calibration().calibrated());
        assert_eq!(inner.calibration().temperature(), 1.0);
    }

    #[test]
    fn last_one_wins() {
        let inner = base();
        let first = Recalibrated::new(Arc::clone(&inner), Calibration::new(2.0, true).unwrap());
        // `with_temperature` always builds over the base scorer, never over the wrapper
        let second = Recalibrated::new(Arc::clone(&inner), Calibration::new(3.0, true).unwrap());
        assert_eq!(first.calibration().temperature(), 2.0);
        assert_eq!(second.calibration().temperature(), 3.0);
        let q = questions();
        assert_eq!(
            ask(&second, &state("s"), &q).unwrap(),
            ask(
                &Recalibrated::new(inner, Calibration::new(3.0, true).unwrap()),
                &state("s"),
                &q
            )
            .unwrap()
        );
    }
}
