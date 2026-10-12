//! The pipeline: scorer logits â†’ temperature softmax â†’ numeric checks â†’ typed answers.

use crate::answers::{Answer, Answers, ChoiceAnswer, Probabilities, ScoreAnswer, YesNoAnswer};
use crate::calibration::calibrated_softmax;
use crate::calibration::formulas::{
    choice_confidence, first_argmax, score_confidence, score_value,
};
use crate::control::{Control, NoControl};
use crate::error::{Error, Result, ShapeProblem};
use crate::schema::{Choice, Question, Questions, State};
use crate::scorer::{Calibration, Scorer};

/// Ask every question of `questions` about `state`, all or nothing.
///
/// The numeric checks of D20 (level 4) always run: shape of the scorer output, finite
/// logits, and a valid distribution. A failure on any question fails the whole call.
///
/// # Errors
/// [`Error::ScorerOutputShape`], [`Error::NonFiniteLogit`], [`Error::InvalidDistribution`],
/// or whatever the scorer itself reports.
pub fn ask(scorer: &dyn Scorer, state: &State, questions: &Questions) -> Result<Answers> {
    ask_with(scorer, state, questions, &NoControl)
}

/// Like [`ask`], but `control` is asked between the rows of the request: if it says stop, the
/// call ends with [`Error::Cancelled`] and no answer at all (spec 008).
///
/// # Errors
/// [`Error::Cancelled`] when `control` says stop; otherwise as [`ask`].
pub fn ask_with(
    scorer: &dyn Scorer,
    state: &State,
    questions: &Questions,
    control: &dyn Control,
) -> Result<Answers> {
    let calibration = scorer.calibration();
    let logits = scorer.score_with(state, questions, control)?;
    check_logits(questions, &logits)?;
    answers_from_logits(questions, &logits, calibration)
}

/// The numeric checks of D20 level 4 on what a scorer returned: one list per question, one value
/// per option, all finite. `ask` and the evaluation functions use this one function.
///
/// # Errors
/// [`Error::ScorerOutputShape`] or [`Error::NonFiniteLogit`].
pub fn check_logits(questions: &Questions, logits: &[Vec<f64>]) -> Result<()> {
    if logits.len() != questions.len() {
        return Err(Error::ScorerOutputShape(ShapeProblem::Questions {
            got: logits.len(),
            expected: questions.len(),
        }));
    }
    for ((name, question), z) in questions.iter().zip(logits) {
        if z.len() != question.option_count() {
            return Err(Error::ScorerOutputShape(ShapeProblem::Options {
                question: name.to_string(),
                got: z.len(),
                expected: question.option_count(),
            }));
        }
    }
    for ((name, _), z) in questions.iter().zip(logits) {
        if let Some((index, value)) = z.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(Error::NonFiniteLogit {
                question: name.to_string(),
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Temperature, softmax and the typed answers, from logits that passed [`check_logits`].
///
/// # Errors
/// [`Error::InvalidDistribution`] if a softmax is not a valid distribution.
pub fn answers_from_logits(
    questions: &Questions,
    logits: &[Vec<f64>],
    calibration: Calibration,
) -> Result<Answers> {
    let mut items = Vec::with_capacity(logits.len());
    for ((name, question), z) in questions.iter().zip(logits) {
        let raw = calibrated_softmax(z, calibration.temperature())?;
        let probs = Probabilities::new(raw).map_err(|reason| Error::InvalidDistribution {
            question: name.to_string(),
            reason,
        })?;
        let p = probs.as_slice();
        let calibrated = calibration.calibrated();
        let answer = match question {
            Question::Choice(choice) => Answer::Choice(choice_answer(choice, p, calibrated)),
            Question::Score(_) => Answer::Score(ScoreAnswer {
                value: score_value(p),
                probabilities: p.to_vec(),
                confidence: score_confidence(p),
                calibrated,
            }),
            Question::YesNo(_) => Answer::YesNo(YesNoAnswer {
                probability: p.get(1).copied().unwrap_or(0.0),
                calibrated,
            }),
        };
        items.push((name.to_string(), answer));
    }
    Ok(Answers { items })
}

fn choice_answer(choice: &Choice, p: &[f64], calibrated: bool) -> ChoiceAnswer {
    let index = first_argmax(p);
    let options = choice.options();
    ChoiceAnswer {
        value: options
            .get(index)
            .map(|o| o.key.clone())
            .unwrap_or_default(),
        index,
        probabilities: options
            .iter()
            .zip(p)
            .map(|(o, p)| (o.key.clone(), *p))
            .collect(),
        confidence: choice_confidence(p),
        calibrated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::formulas::score_deviation;
    use crate::mock::{Fault, MockScorer};
    use crate::schema::{ChoiceOption, Score, StateValue, TextEntry, YesNo};
    use crate::scorer::Calibration;

    fn instr(s: &str) -> TextEntry {
        TextEntry::text(s).unwrap()
    }

    fn state() -> State {
        State::new(StateValue::object([(
            "ticket",
            StateValue::string("Shoes arrived late and in the wrong size. Charged twice."),
        )]))
        .unwrap()
    }

    fn questions() -> Questions {
        Questions::new([
            (
                "team",
                Question::from(
                    Choice::new(
                        instr("Which team should handle this?"),
                        vec![
                            ChoiceOption::described("returns", "Exchanges, refunds").unwrap(),
                            ChoiceOption::described("shipping", "Delays").unwrap(),
                            ChoiceOption::described("billing", "Charges").unwrap(),
                        ],
                    )
                    .unwrap(),
                ),
            ),
            (
                "urgency",
                Question::from(
                    Score::new(
                        instr("How urgent is it?"),
                        vec![instr("low"), instr("medium"), instr("high")],
                    )
                    .unwrap(),
                ),
            ),
            (
                "angry",
                Question::from(YesNo::new(instr("Is the customer angry?"), None, None).unwrap()),
            ),
        ])
        .unwrap()
    }

    fn scripted() -> MockScorer {
        MockScorer::scripted(vec![
            vec![2.0, 1.0, 0.1],
            vec![0.1_f64.ln(), 0.2_f64.ln(), 0.7_f64.ln()],
            vec![1.0, -1.0],
        ])
        .with_calibration(Calibration::new(1.0, true).unwrap())
    }

    /// Stops the work the `limit`-th time it is asked.
    struct StopAt {
        asked: std::sync::atomic::AtomicUsize,
        limit: usize,
    }

    impl Control for StopAt {
        fn checkpoint(&self) -> Result<()> {
            let n = self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            if n >= self.limit {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    /// A scorer that asks the control before each question, like the model scorer does between rows.
    struct Obeying(MockScorer);

    impl Scorer for Obeying {
        fn calibration(&self) -> Calibration {
            self.0.calibration()
        }

        fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>> {
            self.score_with(state, questions, &NoControl)
        }

        fn score_with(
            &self,
            state: &State,
            questions: &Questions,
            control: &dyn Control,
        ) -> Result<Vec<Vec<f64>>> {
            for _ in questions.iter() {
                control.checkpoint()?;
            }
            self.0.score(state, questions)
        }
    }

    /// A scorer that only implements `score`: it never heard of a control.
    struct Plain(MockScorer);

    impl Scorer for Plain {
        fn calibration(&self) -> Calibration {
            self.0.calibration()
        }

        fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>> {
            self.0.score(state, questions)
        }
    }

    #[test]
    fn check_logits_reports_shape_then_finiteness() {
        let q = questions();
        assert!(matches!(
            check_logits(&q, &[vec![0.0; 3]]),
            Err(Error::ScorerOutputShape(ShapeProblem::Questions {
                got: 1,
                expected: 3
            }))
        ));
        let wrong = vec![vec![0.0; 3], vec![0.0; 2], vec![0.0; 2]];
        assert!(matches!(
            check_logits(&q, &wrong),
            Err(Error::ScorerOutputShape(ShapeProblem::Options {
                got: 2,
                expected: 3,
                ..
            }))
        ));
        let nan = vec![vec![0.0, f64::NAN, 0.0], vec![0.0; 3], vec![0.0; 2]];
        assert!(matches!(
            check_logits(&q, &nan),
            Err(Error::NonFiniteLogit { index: 1, .. })
        ));
        let ok = vec![vec![0.0; 3], vec![0.0; 3], vec![0.0; 2]];
        assert!(check_logits(&q, &ok).is_ok());
        assert!(answers_from_logits(&q, &ok, Calibration::uncalibrated()).is_ok());
    }

    #[test]
    fn a_scorer_that_ignores_the_control_gives_the_same_answers() {
        let stop_at_once = StopAt {
            asked: std::sync::atomic::AtomicUsize::new(0),
            limit: 1,
        };
        let plain = ask(&Plain(scripted()), &state(), &questions()).unwrap();
        let with = ask_with(&Plain(scripted()), &state(), &questions(), &stop_at_once).unwrap();
        assert_eq!(plain, with);
        // the default `score_with` never even looked at it
        assert_eq!(
            stop_at_once.asked.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[test]
    fn a_control_that_stops_leaves_no_partial_answer() {
        let scorer = Obeying(scripted());
        let stop_at_second = StopAt {
            asked: std::sync::atomic::AtomicUsize::new(0),
            limit: 2,
        };
        let verdict = ask_with(&scorer, &state(), &questions(), &stop_at_second);
        assert!(matches!(verdict, Err(Error::Cancelled)), "{verdict:?}");
        // the scorer looked once per question until it was stopped, and `ask` (no control) still works
        assert_eq!(
            stop_at_second
                .asked
                .load(std::sync::atomic::Ordering::SeqCst),
            2
        );
        assert!(ask(&scorer, &state(), &questions()).is_ok());
    }

    #[test]
    fn the_worked_example_of_the_spec() {
        let a = ask(&scripted(), &state(), &questions()).unwrap();
        let Some(Answer::Choice(team)) = a.get("team") else {
            panic!("team")
        };
        assert_eq!((team.value(), team.index()), ("returns", 0));
        assert!((team.probabilities()[0].1 - 0.659).abs() < 1e-3);
        assert!((team.confidence() - 0.4885).abs() < 1e-4);
        assert!(team.calibrated());
        let Some(Answer::Score(u)) = a.get("urgency") else {
            panic!("urgency")
        };
        assert!((u.value() - 1.6).abs() < 1e-12);
        assert!((u.confidence() - 0.4).abs() < 1e-12);
        let Some(Answer::YesNo(angry)) = a.get("angry") else {
            panic!("angry")
        };
        assert!((angry.probability() - 0.1192).abs() < 1e-4);
        assert_eq!(
            a.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            ["team", "urgency", "angry"]
        );
    }

    #[test]
    fn closed_form_deviation_matches_the_sum() {
        for l in 1..=255usize {
            let mid = (l as f64 - 1.0) / 2.0;
            let direct: f64 = (0..l).map(|i| (i as f64 - mid).abs()).sum::<f64>() / l as f64;
            assert!((score_deviation(l) - direct).abs() < 1e-12, "L={l}");
        }
        assert_eq!(
            [2, 3, 5, 10].map(score_deviation),
            [0.5, 2.0 / 3.0, 1.2, 2.5]
        );
    }

    #[test]
    fn ties_pick_the_first() {
        assert_eq!(first_argmax(&[0.5, 0.5]), 0);
        assert_eq!(first_argmax(&[0.4, 0.4, 0.2]), 0);
        assert_eq!(first_argmax(&[0.5, 0.0, 0.5]), 0);
        // Score with a tie: m is the first level, spread = 0.5*2 = 1, D = 2/3, so 1 - 1.5 < 0 -> 0
        assert_eq!(score_confidence(&[0.5, 0.0, 0.5]), 0.0);
        // and [0.4, 0.4, 0.2]: m = 0, spread = 0.4 + 0.4 = 0.8, D = 2/3 -> clamped at 0
        assert_eq!(score_confidence(&[0.4, 0.4, 0.2]), 0.0);
    }

    #[test]
    fn confidence_edge_cases() {
        assert_eq!(choice_confidence(&[1.0]), 1.0);
        assert_eq!(score_confidence(&[1.0]), 1.0);
        assert!(choice_confidence(&[0.25; 4]).abs() < 1e-12);
        assert!(score_confidence(&[0.5, 0.5]).abs() < 1e-12);
        // sum 0 normalises to uniform: confidence 0
        assert!(choice_confidence(&[0.0, 0.0, 0.0]).abs() < 1e-12);
        assert!(choice_confidence(&[1.0, 0.0]) <= 1.0);
    }

    #[test]
    fn no_rounding_anywhere() {
        let q = Questions::new([(
            "y",
            Question::from(YesNo::new(instr("?"), None, None).unwrap()),
        )])
        .unwrap();
        let s = MockScorer::scripted(vec![vec![0.123, -0.456]])
            .with_calibration(Calibration::new(2.3510958125672174, true).unwrap());
        let a = ask(&s, &state(), &q).unwrap();
        let Some(Answer::YesNo(y)) = a.get("y") else {
            panic!()
        };
        assert!((y.probability() - 0.4387422484049817).abs() < 1e-15);
        assert_ne!(y.probability(), 0.4387);
    }

    #[test]
    fn calibrated_flag_is_copied_from_the_scorer() {
        for flag in [true, false] {
            let s = scripted().with_calibration(Calibration::new(1.0, flag).unwrap());
            let a = ask(&s, &state(), &questions()).unwrap();
            for (_, ans) in a.iter() {
                let c = match ans {
                    Answer::Choice(x) => x.calibrated(),
                    Answer::Score(x) => x.calibrated(),
                    Answer::YesNo(x) => x.calibrated(),
                };
                assert_eq!(c, flag);
            }
        }
    }

    #[test]
    fn faults_fail_closed_with_the_right_question_and_index() {
        let base = MockScorer::new(7);
        for (fault, value_name) in [
            (Fault::nan("urgency", 1), "NaN"),
            (Fault::pos_inf("urgency", 1), "inf"),
            (Fault::neg_inf("urgency", 1), "-inf"),
        ] {
            let err = ask(&base.with_fault(fault), &state(), &questions()).unwrap_err();
            let Error::NonFiniteLogit {
                question,
                index,
                value,
            } = &err
            else {
                panic!("{err}")
            };
            assert_eq!((question.as_str(), *index), ("urgency", 1));
            assert_eq!(value.to_string(), value_name);
            assert!(err.to_string().contains("option 1 of question \"urgency\""));
        }
        // first failing question in order is reported
        let two = base
            .with_fault(Fault::nan("angry", 0))
            .with_fault(Fault::nan("team", 2));
        let err = ask(&two, &state(), &questions()).unwrap_err();
        assert!(matches!(&err, Error::NonFiniteLogit { question, .. } if question == "team"));
    }

    #[test]
    fn wrong_shapes_are_rejected() {
        let base = MockScorer::new(7);
        for count in [0, 2, 4] {
            let err = ask(
                &base.with_fault(Fault::wrong_option_count("team", count)),
                &state(),
                &questions(),
            )
            .unwrap_err();
            assert_eq!(
                err,
                Error::ScorerOutputShape(ShapeProblem::Options {
                    question: "team".into(),
                    got: count,
                    expected: 3
                })
            );
        }
        for count in [0, 2, 4] {
            let err = ask(
                &base.with_fault(Fault::wrong_question_count(count)),
                &state(),
                &questions(),
            )
            .unwrap_err();
            assert_eq!(
                err,
                Error::ScorerOutputShape(ShapeProblem::Questions {
                    got: count,
                    expected: 3
                })
            );
        }
    }

    #[test]
    fn extreme_logits_are_valid() {
        let q = Questions::new([(
            "c",
            Question::from(
                Choice::new(
                    instr("?"),
                    vec![
                        ChoiceOption::key("a"),
                        ChoiceOption::key("b"),
                        ChoiceOption::key("c"),
                    ],
                )
                .unwrap(),
            ),
        )])
        .unwrap();
        for row in [
            vec![1e300, -1e300, 0.0],
            vec![f64::MAX, -f64::MAX, f64::MIN_POSITIVE],
            vec![-f64::MAX, -f64::MAX, -f64::MAX],
        ] {
            for t in [1e-3, 1.0, 1e3] {
                let s = MockScorer::scripted(vec![row.clone()])
                    .with_calibration(Calibration::new(t, true).unwrap());
                let a = ask(&s, &state(), &q).unwrap();
                let Some(Answer::Choice(c)) = a.get("c") else {
                    panic!()
                };
                let sum: f64 = c.probabilities().iter().map(|(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-9 && sum.is_finite());
            }
        }
    }

    #[test]
    fn all_or_nothing() {
        let s = MockScorer::scripted(vec![
            vec![1.0, 2.0, 3.0],
            vec![f64::NAN, 0.0, 0.0],
            vec![0.0, 0.0],
        ]);
        assert!(ask(&s, &state(), &questions()).is_err());
    }
}

#[cfg(test)]
mod props {
    use super::*;
    use crate::mock::MockScorer;
    use crate::schema::{Choice, ChoiceOption, StateValue, TextEntry};
    use crate::scorer::Calibration;
    use proptest::prelude::*;

    fn choice_with(k: usize) -> Questions {
        let options: Vec<ChoiceOption> =
            (0..k).map(|i| ChoiceOption::key(format!("k{i}"))).collect();
        Questions::new([(
            "q",
            Question::from(Choice::new(TextEntry::text("?").unwrap(), options).unwrap()),
        )])
        .unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, rng_seed: proptest::test_runner::RngSeed::Fixed(7), ..ProptestConfig::default() })]

        #[test]
        fn distribution_invariants(
            logits in prop::collection::vec(-50.0f64..50.0, 1..=255),
            t in 1e-3f64..1e3,
        ) {
            let q = choice_with(logits.len());
            let s = MockScorer::scripted(vec![logits.clone()]).with_calibration(Calibration::new(t, true).unwrap());
            let state = State::new(StateValue::string("s")).unwrap();
            let a = ask(&s, &state, &q).unwrap();
            let Some(Answer::Choice(c)) = a.get("q") else { panic!() };
            let p: Vec<f64> = c.probabilities().iter().map(|(_, p)| *p).collect();
            prop_assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
            prop_assert!((p.iter().sum::<f64>() - 1.0).abs() <= 1e-9);
            prop_assert!((0.0..=1.0).contains(&c.confidence()));
            prop_assert_eq!(c.index(), first_argmax(&p));
            // adding a constant to every logit does not change p
            let shifted: Vec<f64> = logits.iter().map(|x| x + 17.5).collect();
            let s2 = MockScorer::scripted(vec![shifted]).with_calibration(Calibration::new(t, true).unwrap());
            let a2 = ask(&s2, &state, &q).unwrap();
            let Some(Answer::Choice(c2)) = a2.get("q") else { panic!() };
            for ((_, x), (_, y)) in c.probabilities().iter().zip(c2.probabilities()) {
                prop_assert!((x - y).abs() <= 1e-9);
            }
        }

        #[test]
        fn nan_or_inf_anywhere_is_always_an_error(
            k in 1usize..=40,
            pos in 0usize..40,
            kind in 0u8..3,
        ) {
            let pos = pos % k;
            let mut row = vec![0.5; k];
            if let Some(slot) = row.get_mut(pos) {
                *slot = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][usize::from(kind)];
            }
            let q = choice_with(k);
            let state = State::new(StateValue::string("s")).unwrap();
            let r = ask(&MockScorer::scripted(vec![row]), &state, &q);
            let is_expected = matches!(r, Err(Error::NonFiniteLogit { index, .. }) if index == pos);
            prop_assert!(is_expected);
        }

        #[test]
        fn confidence_and_value_ranges(l in 1usize..=255, seed in 0u64..1000) {
            let mut x = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let logits: Vec<f64> = (0..l).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (x >> 40) as f64 / 1e6 }).collect();
            let p = calibrated_softmax(&logits, 1.0).unwrap();
            let v = score_value(&p);
            prop_assert!(v >= -1e-12 && v <= (l as f64 - 1.0) + 1e-12);
            let c = score_confidence(&p);
            prop_assert!((0.0..=1.0).contains(&c));
            if l == 1 { prop_assert_eq!(c, 1.0); }
        }
    }

    #[test]
    fn equal_logits_give_zero_confidence() {
        for k in 2..=20 {
            let p = vec![1.0 / k as f64; k];
            assert!(choice_confidence(&p).abs() < 1e-12);
            assert!(score_confidence(&p).abs() < 1e-9);
        }
    }
}
