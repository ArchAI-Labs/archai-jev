//! Typed answers. None of them can be built from raw probabilities outside this crate, so
//! an answer with invalid probabilities does not exist (D20, level 4).

use crate::error::DistributionProblem;

/// A validated probability distribution: non-empty, finite, each in `[0, 1]`, sum 1 within 1e-9.
#[derive(Debug, Clone, PartialEq)]
pub struct Probabilities(Vec<f64>);

impl Probabilities {
    /// Validate `values`.
    ///
    /// # Errors
    /// The first [`DistributionProblem`] found.
    pub fn new(values: Vec<f64>) -> Result<Self, DistributionProblem> {
        if values.is_empty() {
            return Err(DistributionProblem::Empty);
        }
        if values.iter().any(|p| !p.is_finite()) {
            return Err(DistributionProblem::NotFinite);
        }
        if let Some(p) = values.iter().find(|p| **p < 0.0 || **p > 1.0) {
            return Err(DistributionProblem::OutOfRange(*p));
        }
        let sum: f64 = values.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(DistributionProblem::BadSum(sum));
        }
        Ok(Probabilities(values))
    }

    /// The values, one per option.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

/// Answer to a Choice question.
///
/// ```compile_fail
/// // Answers cannot be built from raw probabilities outside the crate.
/// let a = _core::ChoiceAnswer { value: String::new(), index: 0, probabilities: vec![], confidence: 1.0, calibrated: true };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceAnswer {
    pub(crate) value: String,
    pub(crate) index: usize,
    pub(crate) probabilities: Vec<(String, f64)>,
    pub(crate) confidence: f64,
    pub(crate) calibrated: bool,
}

impl ChoiceAnswer {
    /// The chosen key (first maximum on ties).
    pub fn value(&self) -> &str {
        &self.value
    }
    /// Index of the chosen option.
    pub fn index(&self) -> usize {
        self.index
    }
    /// Probability of each key, in the order of the options.
    pub fn probabilities(&self) -> &[(String, f64)] {
        &self.probabilities
    }
    /// `(max p - 1/K) / (1 - 1/K)` in `[0, 1]`; 1 for a single option.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }
    /// Whether the scorer declared a calibration (D20).
    pub fn calibrated(&self) -> bool {
        self.calibrated
    }
}

/// Answer to a Score question.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreAnswer {
    pub(crate) value: f64,
    pub(crate) probabilities: Vec<f64>,
    pub(crate) confidence: f64,
    pub(crate) calibrated: bool,
}

impl ScoreAnswer {
    /// Expected level `sum(i * p_i)`, 0-based, not normalised.
    pub fn value(&self) -> f64 {
        self.value
    }
    /// Probability of each level.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    /// Confidence in `[0, 1]`; 1 for a single level.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }
    /// Whether the scorer declared a calibration (D20).
    pub fn calibrated(&self) -> bool {
        self.calibrated
    }
}

/// Answer to a YesNo question: only the probability of "yes".
#[derive(Debug, Clone, PartialEq)]
pub struct YesNoAnswer {
    pub(crate) probability: f64,
    pub(crate) calibrated: bool,
}

impl YesNoAnswer {
    /// Probability of "yes" (true).
    pub fn probability(&self) -> f64 {
        self.probability
    }
    /// Whether the scorer declared a calibration (D20).
    pub fn calibrated(&self) -> bool {
        self.calibrated
    }
}

/// One answer, of the kind of its question.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// See [`ChoiceAnswer`].
    Choice(ChoiceAnswer),
    /// See [`ScoreAnswer`].
    Score(ScoreAnswer),
    /// See [`YesNoAnswer`].
    YesNo(YesNoAnswer),
}

/// The answers to all the questions of a request, in question order.
#[derive(Debug, Clone, PartialEq)]
pub struct Answers {
    pub(crate) items: Vec<(String, Answer)>,
}

impl Answers {
    /// The answers in question order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Answer)> {
        self.items.iter().map(|(n, a)| (n.as_str(), a))
    }

    /// The answer to the question called `name`.
    pub fn get(&self, name: &str) -> Option<&Answer> {
        self.items.iter().find(|(n, _)| n == name).map(|(_, a)| a)
    }

    /// Number of answers.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if there are no answers (never the case for a successful `ask`).
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_validation() {
        assert_eq!(Probabilities::new(vec![]), Err(DistributionProblem::Empty));
        assert_eq!(
            Probabilities::new(vec![0.5, f64::NAN]),
            Err(DistributionProblem::NotFinite)
        );
        assert_eq!(
            Probabilities::new(vec![f64::INFINITY]),
            Err(DistributionProblem::NotFinite)
        );
        assert_eq!(
            Probabilities::new(vec![-0.1, 1.1]),
            Err(DistributionProblem::OutOfRange(-0.1))
        );
        assert_eq!(
            Probabilities::new(vec![1.5, -0.5]),
            Err(DistributionProblem::OutOfRange(1.5))
        );
        assert!(matches!(
            Probabilities::new(vec![0.5, 0.4]),
            Err(DistributionProblem::BadSum(_))
        ));
        assert!(matches!(
            Probabilities::new(vec![0.6, 0.5]),
            Err(DistributionProblem::BadSum(_))
        ));
        assert!(Probabilities::new(vec![0.25, 0.75]).is_ok());
        assert!(Probabilities::new(vec![0.5, 0.5 + 5e-10]).is_ok());
        assert!(Probabilities::new(vec![0.5, 0.5 + 2e-9]).is_err());
    }
}
