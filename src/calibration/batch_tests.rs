//! Tests of the public formulas (`calibration::batch`).

use proptest::prelude::*;

use super::batch::*;
use super::calibrated_softmax;
use super::formulas::{first_argmax, score_confidence};
use super::rows::{Rows, SumTolerance};
use crate::answers::Answer;
use crate::ask::ask;
use crate::error::Error;
use crate::mock::MockScorer;
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, Score, State, StateValue, TextEntry, YesNo,
};
use crate::scorer::Calibration;

fn vectors() -> Vec<serde_json::Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/formulas/formulas.jsonl"
    );
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn floats(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn tol() -> SumTolerance {
    SumTolerance::default()
}

#[test]
fn softmax_batch_equals_rows() {
    let data = [
        vec![2.0, 1.0, 0.1],
        vec![-3.0, 4.0],
        vec![1e300, 1e300, 5.0],
    ];
    let out = softmax_batch(Rows::from_slices(&data), 2.3).unwrap();
    for (row, got) in data.iter().zip(&out) {
        assert_eq!(&calibrated_softmax(row, 2.3).unwrap(), got);
    }
    assert!(matches!(
        softmax_batch(Rows::from_slices(&data), 0.0),
        Err(Error::InvalidTemperature(_))
    ));
}

#[test]
fn naive_softmax_would_overflow_and_ours_does_not() {
    let naive: Vec<f64> = [1000.0_f64, 1000.0].iter().map(|x| x.exp()).collect();
    let sum: f64 = naive.iter().sum();
    assert!((naive[0] / sum).is_nan());
    let ours = softmax_batch(Rows::from_slices(&[[1000.0, 1000.0]]), 1.0).unwrap();
    assert_eq!(ours[0], vec![0.5, 0.5]);
}

#[test]
fn yes_no_vectors_8() {
    let mut seen = 0;
    for v in vectors().iter().filter(|v| v["kind"] == "noul") {
        let raw = floats(&v["logits_raw"]);
        let t = v["temperature"].as_f64().unwrap();
        let expected = v["p_true"].as_f64().unwrap();
        let p = yes_no_probability(raw[0], raw[1], t).unwrap();
        assert!((p - expected).abs() <= 1e-12, "{raw:?}: {p} vs {expected}");
        seen += 1;
    }
    assert_eq!(seen, 8);
}

fn sums_to_one(p: &[f64]) -> bool {
    (p.iter().sum::<f64>() - 1.0).abs() <= 1e-6
}

#[test]
fn choice_score_vectors_26() {
    let mut valid = 0;
    for v in vectors().iter().filter(|v| v["kind"] != "noul") {
        let p = floats(&v["p"]);
        if !sums_to_one(&p) {
            continue;
        }
        valid += 1;
        let rows = Rows::from_slices(std::slice::from_ref(&p));
        let expected = v["confidence"].as_f64().unwrap();
        if v["kind"] == "choice" {
            let got = choice_confidence_batch(rows, tol()).unwrap();
            assert!((got[0] - expected).abs() <= 1e-12, "{p:?}");
            assert_eq!(first_argmax(&p) as u64, v["argmax"].as_u64().unwrap());
        } else {
            let got = score_confidence_batch(rows.clone(), tol()).unwrap();
            assert!((got[0] - expected).abs() <= 1e-12, "{p:?}");
            let value = score_value_batch(rows, tol()).unwrap();
            assert!(
                (value[0] - v["value"].as_f64().unwrap()).abs() <= 1e-12,
                "{p:?}"
            );
        }
    }
    assert_eq!(valid, 26);
}

#[test]
fn rejected_vectors_4_with_message() {
    let mut rejected = 0;
    for v in vectors().iter().filter(|v| v["kind"] != "noul") {
        let p = floats(&v["p"]);
        if sums_to_one(&p) {
            continue;
        }
        rejected += 1;
        for result in [
            choice_confidence_batch(Rows::from_slices(std::slice::from_ref(&p)), tol()),
            score_confidence_batch(Rows::from_slices(std::slice::from_ref(&p)), tol()),
            score_value_batch(Rows::from_slices(std::slice::from_ref(&p)), tol()),
        ] {
            let message = result.unwrap_err().to_string();
            let expected = if p.iter().any(|x| *x > 1.0) {
                "probabilities row 0: value 2.0 at option 0 is outside [0, 1]"
            } else {
                "probabilities row 0: sums to 0.0, expected 1 within 1e-6"
            };
            assert_eq!(message, expected);
        }
    }
    assert_eq!(rejected, 4);
}

fn instr(s: &str) -> TextEntry {
    TextEntry::text(s).unwrap()
}

#[test]
fn public_equals_ask_bit_for_bit() {
    let state = State::new(StateValue::object([("t", StateValue::string("x"))])).unwrap();
    let questions = Questions::new([
        (
            "team",
            Question::from(
                Choice::new(
                    instr("Which?"),
                    vec![
                        ChoiceOption::described("a", "A").unwrap(),
                        ChoiceOption::described("b", "B").unwrap(),
                        ChoiceOption::described("c", "C").unwrap(),
                    ],
                )
                .unwrap(),
            ),
        ),
        (
            "urgency",
            Question::from(
                Score::new(instr("How?"), vec![instr("lo"), instr("mid"), instr("hi")]).unwrap(),
            ),
        ),
        (
            "angry",
            Question::from(YesNo::new(instr("Angry?"), None, None).unwrap()),
        ),
    ])
    .unwrap();
    for (a, b, c) in [(2.0, 1.0, 0.1), (0.3, 0.31, 0.29), (-4.0, 9.0, 2.5)] {
        let scorer = MockScorer::scripted(vec![vec![a, b, c], vec![c, a, b], vec![b, c]])
            .with_calibration(Calibration::new(1.7, true).unwrap());
        let answers = ask(&scorer, &state, &questions).unwrap();
        let Some(Answer::Choice(choice)) = answers.get("team") else {
            panic!("not a choice");
        };
        let p: Vec<f64> = choice.probabilities().iter().map(|(_, p)| *p).collect();
        let rows = Rows::from_slices(&[p]);
        let got = choice_confidence_batch(rows, tol()).unwrap();
        assert_eq!(got[0].to_bits(), choice.confidence().to_bits());
        let Some(Answer::Score(score)) = answers.get("urgency") else {
            panic!("not a score");
        };
        let rows = Rows::from_slices(&[score.probabilities().to_vec()]);
        assert_eq!(
            score_confidence_batch(rows.clone(), tol()).unwrap()[0].to_bits(),
            score.confidence().to_bits()
        );
        assert_eq!(
            score_value_batch(rows, tol()).unwrap()[0].to_bits(),
            score.value().to_bits()
        );
        let Some(Answer::YesNo(yes_no)) = answers.get("angry") else {
            panic!("not a yes/no");
        };
        let p = yes_no_probability(b, c, 1.7).unwrap();
        assert!((p - yes_no.probability()).abs() <= 1e-12);
    }
}

#[test]
fn yes_no_swap_gives_complement() {
    let p = yes_no_probability(-1.0, 2.0, 1.5).unwrap();
    let q = yes_no_probability(2.0, -1.0, 1.5).unwrap();
    assert!((p + q - 1.0).abs() <= 1e-15);
}

#[test]
fn yes_no_batch_rows_must_be_pairs() {
    let ok = yes_no_batch(Rows::from_slices(&[[0.0, 1.0], [1.0, 0.0]]), 1.0).unwrap();
    assert_eq!(ok.len(), 2);
    let e = yes_no_batch(Rows::from_slices(&[vec![0.0, 1.0], vec![1.0]]), 1.0).unwrap_err();
    assert_eq!(
        e.to_string(),
        "logits row 1 must have exactly 2 options ([no, yes]), got 1"
    );
}

#[test]
fn single_and_batch_agree_on_scores() {
    let p = [0.1, 0.2, 0.7];
    let rows = Rows::from_slices(&[p]);
    assert_eq!(
        score_confidence_batch(rows, tol()).unwrap()[0],
        score_confidence(&p)
    );
}

proptest! {
    #[test]
    fn yes_no_matches_sigmoid_and_never_nan(
        no in -1e300f64..1e300,
        yes in -1e300f64..1e300,
        t in 1e-3f64..1e3,
    ) {
        let p = yes_no_probability(no, yes, t).unwrap();
        prop_assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        let reference = 1.0 / (1.0 + (-(yes - no) / t).exp());
        if reference.is_finite() {
            prop_assert!((p - reference).abs() <= 1e-12);
        }
        let swapped = yes_no_probability(yes, no, t).unwrap();
        prop_assert!((p + swapped - 1.0).abs() <= 1e-12);
    }
}
