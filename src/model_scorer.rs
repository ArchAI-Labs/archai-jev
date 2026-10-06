//! The scorer of a real model: prompt, forward, head (spec 006b section 7).
//!
//! One function, [`ModelScorer::run`], builds the prompt, runs the engine and applies the head.
//! `ask` and the self-check of model loading both go through it, so what is verified at load time
//! is what answers requests.

use std::sync::Arc;

use crate::engine::Forward;
use crate::error::{Error, Result};
use crate::heads::Head;
use crate::json_strict::Json;
use crate::models::verify::{RunOutput, VectorRunner};
use crate::prompt::{PromptTokenizer, RequestLimits, Template};
use crate::request::request_to_domain;
use crate::schema::{Questions, State};
use crate::scorer::{Calibration, Scorer};

/// Everything a loaded model is made of.
pub struct ModelScorer {
    template: Box<dyn Template>,
    tokenizer: PromptTokenizer,
    forward: Arc<dyn Forward>,
    head: Box<dyn Head>,
    calibration: Calibration,
    model: String,
    max_context: u64,
    question_types: Vec<String>,
}

/// What one request produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    /// The packed prompt ids (state, then every branch).
    pub input_ids: Vec<u32>,
    /// Raw logits per question, in question order.
    pub questions: Vec<(String, Vec<f64>)>,
}

impl Run {
    /// `usage.input_tokens`: the length of the packed prompt (as Kev counts it).
    pub fn input_tokens(&self) -> usize {
        self.input_ids.len()
    }
}

impl ModelScorer {
    /// Put the pieces together.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: impl Into<String>,
        template: Box<dyn Template>,
        tokenizer: PromptTokenizer,
        forward: Arc<dyn Forward>,
        head: Box<dyn Head>,
        calibration: Calibration,
        max_context: u64,
        question_types: Vec<String>,
    ) -> Self {
        ModelScorer {
            template,
            tokenizer,
            forward,
            head,
            calibration,
            model: model.into(),
            max_context,
            question_types,
        }
    }

    /// Answer a request: raw logits for every question, or the reason it cannot be answered.
    ///
    /// # Errors
    /// [`Error::Unsupported`] before the engine is touched; [`Error::Inference`] or
    /// [`Error::NonFiniteHidden`] from the engine; nothing is returned partially.
    pub fn run(&self, state: &State, questions: &Questions) -> Result<Run> {
        let types: Vec<&str> = self.question_types.iter().map(String::as_str).collect();
        let limits = RequestLimits {
            model: &self.model,
            max_context: self.max_context,
            question_types: &types,
            support: self.head.as_ref(),
        };
        let prompt = self
            .template
            .build(state, questions, &self.tokenizer, &limits)?;

        let mut session = self.forward.begin(prompt.state())?;
        let mut logits = Vec::with_capacity(questions.len());
        for ((name, question), branch) in questions.iter().zip(prompt.branches()) {
            let specs = self.head.outputs(question, branch);
            let outputs = session
                .row(branch.ids(), &specs)
                .map_err(|e| name_the_question(e, name))?;
            logits.push((name.to_string(), self.head.score(question, &outputs)?));
        }
        Ok(Run {
            input_ids: prompt.packed_ids(),
            questions: logits,
        })
    }
}

/// The engine does not know question names: put the name on the errors that need one.
fn name_the_question(error: Error, name: &str) -> Error {
    match error {
        Error::NonFiniteHidden {
            position, value, ..
        } => Error::NonFiniteHidden {
            question: name.to_string(),
            position,
            value,
        },
        other => other,
    }
}

impl Scorer for ModelScorer {
    fn calibration(&self) -> Calibration {
        self.calibration
    }

    fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>> {
        Ok(self
            .run(state, questions)?
            .questions
            .into_iter()
            .map(|(_, z)| z)
            .collect())
    }
}

impl VectorRunner for ModelScorer {
    fn run(&self, request: &Json) -> Result<RunOutput> {
        let (state, questions) =
            request_to_domain(request).map_err(|detail| Error::Inference { detail })?;
        let run = ModelScorer::run(self, &state, &questions)?;
        Ok(RunOutput {
            input_ids: run.input_ids,
            questions: run.questions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ask::ask;
    use crate::engine::fake::{FakeForward, Faults};
    use crate::heads::PointerHead;
    use crate::heads::test_head::LastTokenHead;
    use crate::prompt::kev::KEV_V1;
    use crate::prompt::tokenize::testkit;
    use crate::schema::{Choice, ChoiceOption, Question, Score, StateValue, TextEntry, YesNo};

    const DIM: usize = 8;
    const TYPES: [&str; 3] = ["choice", "noul", "score"];

    fn pointer() -> Box<dyn Head> {
        let w = |seed: usize| -> Vec<f32> {
            (0..4 * DIM)
                .map(|i| (((i * 7 + seed * 13) % 17) as f32 - 8.0) / 8.0)
                .collect()
        };
        Box::new(
            PointerHead::new(
                DIM,
                4,
                w(1),
                vec![0.1, -0.2, 0.3, 0.0],
                w(2),
                vec![0.0, 0.2, -0.1, 0.4],
            )
            .unwrap(),
        )
    }

    fn scorer_with(head: Box<dyn Head>, max: u64) -> (ModelScorer, Arc<FakeForward>) {
        let fake = Arc::new(FakeForward::new(DIM));
        let s = ModelScorer::new(
            "test-kev",
            Box::new(crate::prompt::kev::KevV1),
            testkit::kev(),
            fake.clone(),
            head,
            Calibration::new(1.0, true).unwrap(),
            max,
            TYPES.iter().map(|t| (*t).to_string()).collect(),
        );
        (s, fake)
    }

    fn scorer() -> (ModelScorer, Arc<FakeForward>) {
        scorer_with(pointer(), 10_000)
    }

    fn state(s: &str) -> State {
        State::new(StateValue::string(s)).unwrap()
    }

    fn choice(instr: &str, keys: &[&str]) -> Question {
        Choice::new(
            TextEntry::text(instr).unwrap(),
            keys.iter().map(|k| ChoiceOption::key(*k)).collect(),
        )
        .unwrap()
        .into()
    }

    fn two_questions() -> Questions {
        Questions::new([
            ("first", choice("Which one?", &["a", "b", "c"])),
            (
                "second",
                YesNo::new(TextEntry::text("Sure?").unwrap(), None, None)
                    .unwrap()
                    .into(),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn each_row_is_the_state_plus_its_own_branch() {
        let (s, fake) = scorer();
        let st = state("some state");
        let qs = two_questions();
        let run = s.run(&st, &qs).unwrap();
        let stats = fake.stats();
        assert_eq!((stats.sessions, stats.rows), (1, 2));
        let prompt = KEV_V1
            .build(
                &st,
                &qs,
                &testkit::kev(),
                &RequestLimits {
                    model: "m",
                    max_context: u64::MAX,
                    question_types: &TYPES,
                    support: &crate::prompt::NoLimits,
                },
            )
            .unwrap();
        // The fake records state + branch for every row: that is what the model would see.
        assert_eq!(stats.seen_rows[0], prompt.row(0).unwrap());
        assert_eq!(stats.seen_rows[1], prompt.row(1).unwrap());
        assert_eq!(run.input_ids, prompt.packed_ids());
        assert_eq!(run.input_tokens(), prompt.packed_len());
    }

    #[test]
    fn the_state_is_decoded_once_whatever_the_number_of_questions() {
        for n in [1usize, 40] {
            let (s, fake) = scorer();
            let st = state("a state that is not tiny at all, to make the difference visible");
            let items: Vec<(String, Question)> = (0..n)
                .map(|i| {
                    (
                        format!("q{i}"),
                        choice(&format!("Question {i}?"), &["x", "y"]),
                    )
                })
                .collect();
            let qs = Questions::new(items).unwrap();
            let run = s.run(&st, &qs).unwrap();
            let stats = fake.stats();
            // state once + every branch, never `n` times the state
            assert_eq!(stats.decoded_tokens, run.input_ids.len(), "n = {n}");
            assert_eq!(stats.rows, n);
        }
    }

    #[test]
    fn order_and_repeats_do_not_change_a_question() {
        let (s, _) = scorer();
        let st = state("shared state");
        let a = choice("A?", &["x", "y"]);
        let b = choice("B?", &["p", "q", "r"]);
        let forward = s
            .run(
                &st,
                &Questions::new([("a", a.clone()), ("b", b.clone())]).unwrap(),
            )
            .unwrap();
        let backward = s
            .run(
                &st,
                &Questions::new([("b", b.clone()), ("a", a.clone())]).unwrap(),
            )
            .unwrap();
        let repeated = s
            .run(
                &st,
                &Questions::new([("a", a.clone()), ("a2", a), ("b", b)]).unwrap(),
            )
            .unwrap();
        let find = |r: &Run, n: &str| r.questions.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(find(&forward, "a"), find(&backward, "a"));
        assert_eq!(find(&forward, "b"), find(&backward, "b"));
        assert_eq!(find(&forward, "a"), find(&repeated, "a"));
        assert_eq!(find(&repeated, "a"), find(&repeated, "a2"));
        assert_eq!(find(&forward, "b"), find(&repeated, "b"));
    }

    #[test]
    fn outputs_are_decide_and_the_box_ends_only() {
        let (s, fake) = scorer();
        let qs = two_questions();
        s.run(&state("s"), &qs).unwrap();
        let stats = fake.stats();
        let expected: usize = qs.iter().map(|(_, q)| 1 + q.option_count()).sum();
        assert_eq!(stats.outputs, expected);
        for (specs, (_, q)) in stats.seen_outputs.iter().zip(qs.iter()) {
            assert_eq!(specs.len(), 1 + q.option_count());
        }
    }

    #[test]
    fn a_fault_is_all_or_nothing_and_the_next_request_is_unharmed() {
        let (s, fake) = scorer();
        let st = state("s");
        let qs = two_questions();
        let reference = s.run(&st, &qs).unwrap();
        fake.set_faults(Faults {
            fail_row: Some(3),
            ..Faults::default()
        }); // second row of the 2nd request
        let err = s.run(&st, &qs).unwrap_err();
        assert!(matches!(err, Error::Inference { .. }), "{err}");
        fake.set_faults(Faults::default());
        assert_eq!(s.run(&st, &qs).unwrap(), reference);
    }

    #[test]
    fn a_non_finite_hidden_state_names_the_question() {
        let (s, fake) = scorer();
        fake.set_faults(Faults {
            nan_row: Some(1),
            ..Faults::default()
        });
        let err = s.run(&state("s"), &two_questions()).unwrap_err();
        match err {
            Error::NonFiniteHidden { question, .. } => assert_eq!(question, "second"),
            other => panic!("{other}"),
        }
    }

    #[test]
    fn concurrent_requests_give_the_sequential_answers() {
        let (s, _) = scorer();
        let st = state("shared");
        let qs = two_questions();
        let reference = s.run(&st, &qs).unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| s.run(&st, &qs).unwrap()))
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), reference);
            }
        });
    }

    #[test]
    fn logits_are_raw_and_the_temperature_is_applied_by_ask() {
        let (s, fake) = scorer();
        let st = state("s");
        let qs = Questions::new([("q", choice("Q?", &["a", "b", "c"]))]).unwrap();
        let raw = s.run(&st, &qs).unwrap().questions[0].1.clone();
        let other = ModelScorer::new(
            "m",
            Box::new(crate::prompt::kev::KevV1),
            testkit::kev(),
            fake,
            pointer(),
            Calibration::new(4.0, true).unwrap(),
            10_000,
            TYPES.iter().map(|t| (*t).to_string()).collect(),
        );
        assert_eq!(other.run(&st, &qs).unwrap().questions[0].1, raw);
        let p1 = ask(&s, &st, &qs).unwrap();
        let p4 = ask(&other, &st, &qs).unwrap();
        let (crate::Answer::Choice(a), crate::Answer::Choice(b)) =
            (p1.get("q").unwrap(), p4.get("q").unwrap())
        else {
            panic!("choice answers")
        };
        assert_ne!(a.probabilities()[0].1, b.probabilities()[0].1);
        assert!((a.probabilities().iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_head_that_reads_only_the_last_token_plugs_in_without_touching_the_engine() {
        let (s, fake) = scorer_with(Box::new(LastTokenHead { max_options: 4 }), 10_000);
        let st = state("s");
        let qs = Questions::new([("q", choice("Q?", &["a", "b"]))]).unwrap();
        let run = s.run(&st, &qs).unwrap();
        // One output, at the last token of the branch; the logits are the first values of the
        // fake hidden state at the end of the row.
        let stats = fake.stats();
        assert_eq!(stats.outputs, 1);
        let row = stats.seen_rows[0].clone();
        let want = fake.hidden_of(&row);
        assert_eq!(
            run.questions[0].1,
            vec![f64::from(want[0]), f64::from(want[1])]
        );
        // The head can refuse a question: five options against a maximum of four.
        let five = Questions::new([("q", choice("Q?", &["a", "b", "c", "d", "e"]))]).unwrap();
        assert!(matches!(s.run(&st, &five), Err(Error::Unsupported(_))));
    }

    #[test]
    fn refusals_happen_before_the_engine_is_touched() {
        let (s, fake) = scorer_with(pointer(), 5);
        let err = s
            .run(&state("a long enough state"), &two_questions())
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
        assert_eq!(fake.stats().sessions, 0, "the engine must not be started");
        let score = Questions::new([(
            "s",
            Score::new(
                TextEntry::text("How?").unwrap(),
                vec![TextEntry::text("a").unwrap(), TextEntry::text("b").unwrap()],
            )
            .unwrap()
            .into(),
        )])
        .unwrap();
        let (only_choice, fake2) = {
            let fake = Arc::new(FakeForward::new(DIM));
            let s = ModelScorer::new(
                "m",
                Box::new(crate::prompt::kev::KevV1),
                testkit::kev(),
                fake.clone(),
                pointer(),
                Calibration::uncalibrated(),
                10_000,
                vec!["choice".to_string()],
            );
            (s, fake)
        };
        assert!(matches!(
            only_choice.run(&state("s"), &score),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(fake2.stats().sessions, 0);
    }
}
