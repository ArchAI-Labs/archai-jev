//! The factual notes of an evaluation (spec 017 §2.2). They are deterministic and never change a
//! number; they say what a reader of the table should keep in mind.

use crate::schema::QuestionKind;

use super::fingerprint::kind_name;

/// Fewer rows than this make the ECE and the reliability bins unreliable.
pub const SMALL_SAMPLE: usize = 100;

/// The notes, in the order of the specification.
///
/// `accuracy` and `mean_confidence` are those of all the rows; `rows_by_kind` the number of rows
/// of each kind present (in the fixed order choice, score, yes/no); `overlap` is
/// `(records also used to calibrate, records)` when the calibration has a provenance.
pub fn notes(
    accuracy: f64,
    mean_confidence: f64,
    rows_by_kind: &[(QuestionKind, usize)],
    calibrated: bool,
    overlap: Option<(usize, usize)>,
) -> Vec<String> {
    let mut out = Vec::new();
    if mean_confidence - accuracy > 0.05 {
        out.push(format!(
            "overconfident: mean confidence {mean_confidence:.3} exceeds accuracy {accuracy:.3}; consider Jev.calibrate() on held-out data"
        ));
    }
    if accuracy - mean_confidence > 0.05 {
        out.push(format!(
            "underconfident: accuracy {accuracy:.3} exceeds mean confidence {mean_confidence:.3}; consider Jev.calibrate() on held-out data"
        ));
    }
    for (kind, n) in rows_by_kind {
        if *n < SMALL_SAMPLE {
            out.push(format!(
                "small sample: only {n} {} questions; ECE and the reliability bins are unreliable (try bootstrap_samples)",
                kind_name(*kind)
            ));
        }
    }
    if !calibrated {
        out.push(
            "the model has no declared calibration (calibrated=False); its probabilities are not guaranteed to be calibrated"
                .to_string(),
        );
    }
    if let Some((k, n)) = overlap
        && k > 0
    {
        out.push(overlap_note(k, n));
    }
    out
}

/// The note (and the text of the `UserWarning`) for records used to calibrate.
pub fn overlap_note(k: usize, n: usize) -> String {
    format!(
        "{k} of {n} records were also used to calibrate this model; these metrics are optimistic, evaluate on records not used for calibration"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use QuestionKind::{Choice, Score, YesNo};

    #[test]
    fn notes_each_row_exactly() {
        let kinds = [(Choice, 150), (Score, 40), (YesNo, 99)];
        let n = notes(0.9, 0.96, &kinds, false, Some((3, 7)));
        assert_eq!(
            n,
            vec![
                "overconfident: mean confidence 0.960 exceeds accuracy 0.900; consider Jev.calibrate() on held-out data",
                "small sample: only 40 score questions; ECE and the reliability bins are unreliable (try bootstrap_samples)",
                "small sample: only 99 yes_no questions; ECE and the reliability bins are unreliable (try bootstrap_samples)",
                "the model has no declared calibration (calibrated=False); its probabilities are not guaranteed to be calibrated",
                "3 of 7 records were also used to calibrate this model; these metrics are optimistic, evaluate on records not used for calibration",
            ]
        );
        let under = notes(0.95, 0.88, &[(Choice, 100)], true, None);
        assert_eq!(
            under,
            vec![
                "underconfident: accuracy 0.950 exceeds mean confidence 0.880; consider Jev.calibrate() on held-out data"
            ]
        );
    }

    #[test]
    fn no_notes_when_none_applies() {
        assert!(
            notes(
                0.9,
                0.93,
                &[(Choice, 100), (Score, 500)],
                true,
                Some((0, 600))
            )
            .is_empty()
        );
        // a difference below 0.05 says nothing
        assert!(notes(0.90, 0.94, &[(Choice, 100)], true, None).is_empty());
    }
}
