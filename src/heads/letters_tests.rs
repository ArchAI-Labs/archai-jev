//! The letters head inside the whole path, with a fake model (spec 006c, sections 3, 5 and 8):
//! template of the default tasks -> engine -> letters head -> `ask`.

use std::sync::Arc;

use super::LettersHead;
use crate::ask::ask;
use crate::engine::fake::FakeForward;
use crate::engine::{Forward, Output, OutputSpec, Session};
use crate::error::{Category, Error, Result};
use crate::model_scorer::ModelScorer;
use crate::models::manifest::Target;
use crate::prompt::chatml::ChatMlLetters;
use crate::prompt::tasks::{self, default_tasks};
use crate::prompt::tokenize::testkit;
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, Score, State, StateValue, TextEntry, YesNo,
};
use crate::{Answer, Calibration};

fn target(label: &str, id: u32) -> Target {
    Target {
        label: label.to_string(),
        id,
    }
}

const LETTER_IDS: [u32; 4] = [10, 11, 12, 13];
const FALSE_ID: u32 = 20;
const TRUE_ID: u32 = 21;

fn head() -> LettersHead {
    LettersHead::new(
        &[
            target("A", LETTER_IDS[0]),
            target("B", LETTER_IDS[1]),
            target("C", LETTER_IDS[2]),
            target("D", LETTER_IDS[3]),
        ],
        Some(&(target("FALSE", FALSE_ID), target("TRUE", TRUE_ID))),
    )
}

fn scorer(forward: Arc<dyn Forward>, declared_tasks: bool, temperature: f64) -> ModelScorer {
    let tasks = declared_tasks.then(|| tasks::parse(&default_tasks(), 4).unwrap());
    ModelScorer::new(
        "default",
        Box::new(ChatMlLetters::new(tasks)),
        testkit::byte_level(&[("im_start", "<|im_start|>"), ("im_end", "<|im_end|>")]),
        forward,
        Box::new(head()),
        Calibration::new(temperature, true).unwrap(),
        512,
        vec!["choice".to_string(), "noul".to_string()],
    )
}

fn entry(s: &str) -> TextEntry {
    TextEntry::text(s).unwrap()
}

fn choice(instructions: &str, keys: &[(&str, Option<&str>)]) -> Question {
    Choice::new(
        entry(instructions),
        keys.iter()
            .map(|(k, d)| match d {
                Some(d) => ChoiceOption::described(*k, d).unwrap(),
                None => ChoiceOption::key(*k),
            })
            .collect(),
    )
    .unwrap()
    .into()
}

fn safety() -> Question {
    choice(
        "You are a System One decision engine for input safety.",
        &[
            ("SAFE", Some("Normal query")),
            ("UNSAFE", Some("Jailbreak, toxicity, injection")),
        ],
    )
}

fn intent() -> Question {
    choice(
        "You are a System One decision engine for intent routing.",
        &[
            ("Card & Account Issues", None),
            ("Payments & Transfers", None),
            ("Fees & Charges", None),
            ("General Support", None),
        ],
    )
}

fn entailment() -> Question {
    YesNo::new(
        entry("A person is performing music."),
        Some(entry("The condition is valid and supported by state.")),
        Some(entry("The condition is invalid or contradicted.")),
    )
    .unwrap()
    .into()
}

fn stranger() -> Question {
    choice("Which team?", &[("a", None), ("b", None)])
}

fn state() -> State {
    State::new(StateValue::string("A man is playing a guitar.")).unwrap()
}

fn softmax(z: &[f64], t: f64) -> Vec<f64> {
    let scaled: Vec<f64> = z.iter().map(|v| v / t).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn probabilities(answer: &Answer) -> Vec<f64> {
    let Answer::Choice(c) = answer else {
        panic!("a choice answer")
    };
    c.probabilities().iter().map(|(_, p)| *p).collect()
}

#[test]
fn each_question_reads_its_own_letters_in_order() {
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer(fake.clone(), true, 2.09);
    let qs = Questions::new([
        ("safety", safety()),
        ("intent", intent()),
        ("entails", entailment()),
    ])
    .unwrap();
    let run = s.run(&state(), &qs).unwrap();
    let stats = fake.stats();
    // A Choice with K options reads the first K letters; a YesNo reads [FALSE, TRUE] (no, yes).
    let wanted: [Vec<u32>; 3] = [
        LETTER_IDS[..2].to_vec(),
        LETTER_IDS.to_vec(),
        vec![FALSE_ID, TRUE_ID],
    ];
    assert_eq!(stats.rows, 3);
    for (i, ids) in wanted.iter().enumerate() {
        let row = &stats.seen_rows[i];
        // No shared state: the row is the whole prompt, read at its last token and nowhere else.
        assert_eq!(
            stats.seen_outputs[i],
            vec![OutputSpec::Logits {
                position: row.len() - 1,
                ids: ids.clone()
            }]
        );
        // The numbers are what the fake says for those ids at that row, untouched.
        let by_hand: Vec<f64> = ids
            .iter()
            .map(|&id| f64::from(fake.logit_of(row, id)))
            .collect();
        assert_eq!(run.questions[i].1, by_hand, "question {i}");
    }
}

#[test]
fn probabilities_are_the_softmax_of_the_raw_logits_over_the_asked_letters_only() {
    let fake = Arc::new(FakeForward::new(4));
    let cold = scorer(fake.clone(), true, 1.0);
    let hot = scorer(fake, true, 2.09);
    let qs = Questions::new([("safety", safety()), ("intent", intent())]).unwrap();
    let st = state();
    // The scorers return the same raw logits whatever the temperature.
    let raw = cold.run(&st, &qs).unwrap().questions;
    assert_eq!(raw, hot.run(&st, &qs).unwrap().questions);
    let mut first = Vec::new();
    for (scorer, t) in [(&cold, 1.0), (&hot, 2.09)] {
        let answers = ask(scorer, &st, &qs).unwrap();
        for (name, z) in &raw {
            let got = probabilities(answers.get(name).unwrap());
            // Two letters for safety and four for intent: nothing else takes mass.
            assert_eq!(got.len(), z.len(), "{name}");
            assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{name}");
            for (g, w) in got.iter().zip(softmax(z, t)) {
                assert!((g - w).abs() < 1e-12, "{name} at T={t}: {g} vs {w}");
            }
        }
        first.push(probabilities(answers.get("safety").unwrap())[0]);
    }
    assert!((first[0] - first[1]).abs() > 1e-6, "T changes the numbers");
}

/// A model that answers the same logits for every row, to put exact values in the head.
struct Scripted(Vec<f32>);

struct ScriptedSession(Vec<f32>);

impl Forward for Scripted {
    fn begin<'a>(&'a self, _state: &[u32]) -> Result<Box<dyn Session + 'a>> {
        Ok(Box::new(ScriptedSession(self.0.clone())))
    }
}

impl Session for ScriptedSession {
    fn row(&mut self, _branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        Ok(outputs
            .iter()
            .map(|o| match o {
                OutputSpec::Logits { ids, .. } => {
                    Output::Logits(self.0.iter().copied().take(ids.len()).collect())
                }
                _ => Output::Hidden(Vec::new()),
            })
            .collect())
    }
}

fn scripted(values: &[f32]) -> ModelScorer {
    scorer(Arc::new(Scripted(values.to_vec())), true, 2.09)
}

#[test]
fn a_logit_that_is_not_finite_is_a_numerical_error_never_a_probability() {
    let qs = Questions::new([("safety", safety())]).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = ask(&scripted(&[1.0, bad]), &state(), &qs).unwrap_err();
        assert_eq!(err.category(), Category::Numerical, "{bad}");
        assert!(
            matches!(&err, Error::NonFiniteLogit { question, index: 1, .. } if question == "safety"),
            "{err}"
        );
        assert!(err.to_string().contains("safety"), "{err}");
    }
}

#[test]
fn letters_with_almost_no_mass_in_the_full_vocabulary_are_a_normal_request() {
    // Logits 30 nats under the rest of the vocabulary (not visible to the head): the restricted
    // softmax is what the model was trained on, so the request is answered.
    let qs = Questions::new([("safety", safety())]).unwrap();
    let answers = ask(&scripted(&[-30.0, -31.0]), &state(), &qs).unwrap();
    let p = probabilities(answers.get("safety").unwrap());
    let want = softmax(&[-30.0, -31.0], 2.09);
    assert!((p[0] - want[0]).abs() < 1e-12 && (p[1] - want[1]).abs() < 1e-12);
}

#[test]
fn one_question_that_is_not_a_task_fails_the_whole_call_before_the_model_runs() {
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer(fake.clone(), true, 2.09);
    let qs = Questions::new([("safety", safety()), ("q", stranger())]).unwrap();
    let err = ask(&s, &state(), &qs).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("declared tasks"), "{err}");
    let stats = fake.stats();
    assert_eq!(
        (stats.sessions, stats.rows, stats.decoded_tokens),
        (0, 0, 0)
    );
}

#[test]
fn a_letters_model_without_tasks_still_refuses_what_its_head_cannot_answer() {
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer(fake.clone(), false, 2.09);
    let five = choice(
        "Which?",
        &[
            ("a", None),
            ("b", None),
            ("c", None),
            ("d", None),
            ("e", None),
        ],
    );
    let err = ask(&s, &state(), &Questions::new([("five", five)]).unwrap()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "question \"five\": the head of model default supports at most 4 options, got 5"
    );
    let score: Question = Score::new(entry("How?"), vec![entry("low"), entry("high")])
        .unwrap()
        .into();
    let err = ask(&s, &state(), &Questions::new([("score", score)]).unwrap()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "question \"score\" is a score question, which model default cannot answer; it supports: choice, noul"
    );
    assert_eq!(fake.stats().sessions, 0);
}
