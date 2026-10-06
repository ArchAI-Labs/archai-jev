//! Self-check (D20, level 2): run the manifest's vectors through the engine and compare.
//!
//! Prompt token ids must be identical; logits and probabilities must be within the tolerance of
//! the (family, dtype). A failure is never cached and the model is not returned.

use super::failures::VerifyFailure;
use super::manifest::Variant;
use super::tolerance::Tolerance;
use super::vectors::Vector;
use crate::calibration::calibrated_softmax;
use crate::error::{Error, Result};
use crate::json_strict::Json;

/// What the engine produced for one request: the prompt it built and the raw logits.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutput {
    /// Token ids of the prompt.
    pub input_ids: Vec<u32>,
    /// Raw logits (after the head, before temperature) per question name.
    pub questions: Vec<(String, Vec<f64>)>,
}

/// Runs a self-check request through the same path as a real one (the engine implements it).
pub trait VectorRunner: Send + Sync {
    /// Build the prompt for `request` (TypeSafe JSON shape) and score it.
    ///
    /// # Errors
    /// Whatever prevents the engine from answering.
    fn run(&self, request: &Json) -> Result<RunOutput>;
}

fn fail(f: VerifyFailure) -> Error {
    Error::ModelVerification(f)
}

/// Run every vector of `variant` and compare with the expected numbers.
///
/// `manifest_temperature` is the temperature the expected probabilities were computed with
/// (the user's `temperature=` never enters the self-check).
///
/// # Errors
/// [`Error::ModelVerification`] on the first discrepancy.
pub fn run_selfcheck(
    model: &str,
    variant: &Variant,
    runner: &dyn VectorRunner,
    tolerance: Tolerance,
    manifest_temperature: f64,
) -> Result<()> {
    for vector in &variant.vectors {
        check_vector(model, vector, runner, tolerance, manifest_temperature)?;
    }
    Ok(())
}

fn check_vector(
    model: &str,
    vector: &Vector,
    runner: &dyn VectorRunner,
    tol: Tolerance,
    temperature: f64,
) -> Result<()> {
    let shape = |detail: String| {
        fail(VerifyFailure::Shape {
            model: model.to_string(),
            vector: vector.id.clone(),
            detail,
        })
    };
    let out = runner
        .run(&vector.request)
        .map_err(|e| shape(format!("the engine could not answer the vector: {e}")))?;

    if out.input_ids != vector.input_ids {
        let position = out
            .input_ids
            .iter()
            .zip(&vector.input_ids)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| out.input_ids.len().min(vector.input_ids.len()));
        return Err(fail(VerifyFailure::Ids {
            model: model.to_string(),
            vector: vector.id.clone(),
            position,
            expected: vector.input_ids.get(position).copied(),
            got: out.input_ids.get(position).copied(),
            expected_len: vector.input_ids.len(),
            got_len: out.input_ids.len(),
        }));
    }

    for (name, expected) in &vector.questions {
        let Some((_, logits)) = out.questions.iter().find(|(n, _)| n == name) else {
            return Err(shape(format!(
                "the engine returned no answer for question '{name}'"
            )));
        };
        if logits.len() != expected.logits.len() {
            return Err(shape(format!(
                "question '{name}' got {} logits, expected {}",
                logits.len(),
                expected.logits.len()
            )));
        }
        if let Some((option, value)) = logits
            .iter()
            .copied()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(fail(VerifyFailure::NonFinite {
                model: model.to_string(),
                vector: vector.id.clone(),
                question: name.clone(),
                option,
                value,
            }));
        }
        if let Some(limit) = tol.logit {
            for (option, (got, want)) in logits.iter().zip(&expected.logits).enumerate() {
                if (got - want).abs() > limit {
                    return Err(fail(VerifyFailure::Logit {
                        model: model.to_string(),
                        vector: vector.id.clone(),
                        question: name.clone(),
                        option,
                        expected: *want,
                        got: *got,
                        tolerance: limit,
                    }));
                }
            }
        }
        let probs = calibrated_softmax(logits, temperature)?;
        for (option, (got, want)) in probs.iter().zip(&expected.probabilities).enumerate() {
            if (got - want).abs() > tol.p {
                return Err(fail(VerifyFailure::Probability {
                    model: model.to_string(),
                    vector: vector.id.clone(),
                    question: name.clone(),
                    option,
                    expected: *want,
                    got: *got,
                    tolerance: tol.p,
                }));
            }
        }
    }
    Ok(())
}
