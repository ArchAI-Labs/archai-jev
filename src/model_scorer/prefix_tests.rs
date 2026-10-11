//! The scorer decoding what a request shares once, in the segments each row's cuts define.
//!
//! The fake is **partition sensitive** (its numbers depend on how a row was split into decodes),
//! so "the answer is bit-identical whatever the other questions are" is a claim these tests can
//! fail: they do, with the mutant that decodes the shared prefix as one segment.

use std::sync::Arc;

use super::*;
use crate::engine::fake::{FakeForward, Faults};
use crate::models::manifest::Target;
use crate::prefix::RequestPlan;
use crate::prompt::chatml::ChatMlLetters;
use crate::prompt::tasks::{self, default_tasks};
use crate::prompt::tokenize::testkit;
use crate::schema::{Choice, ChoiceOption, Question, StateValue, TextEntry, YesNo};

const TYPES: [&str; 2] = ["choice", "noul"];

fn target(label: &str, id: u32) -> Target {
    Target {
        label: label.to_string(),
        id,
    }
}

fn letters() -> Box<dyn Head> {
    Box::new(crate::heads::LettersHead::new(
        &[
            target("A", 10),
            target("B", 11),
            target("C", 12),
            target("D", 13),
        ],
        Some(&(target("FALSE", 20), target("TRUE", 21))),
    ))
}

fn tokenizer() -> PromptTokenizer {
    testkit::byte_level(&[("im_start", "<|im_start|>"), ("im_end", "<|im_end|>")])
}

/// A scorer for the default model's tasks over a partition-sensitive fake.
pub(super) fn scorer(limit: u64) -> (ModelScorer, Arc<FakeForward>) {
    let fake = Arc::new(FakeForward::new(4).partition_sensitive());
    let s = ModelScorer::new(
        "default",
        Box::new(ChatMlLetters::new(Some(
            tasks::parse(&default_tasks(), 4).unwrap(),
        ))),
        tokenizer(),
        fake.clone(),
        letters(),
        Calibration::new(1.0, true).unwrap(),
        100_000,
        TYPES.iter().map(|t| (*t).to_string()).collect(),
    )
    .with_prefix_cache(limit);
    (s, fake)
}

pub(super) fn state(s: &str) -> State {
    State::new(StateValue::string(s)).unwrap()
}

pub(super) fn entailment(condition: &str) -> Question {
    YesNo::new(
        TextEntry::text(condition).unwrap(),
        Some(TextEntry::text("The condition is valid and supported by state.").unwrap()),
        Some(TextEntry::text("The condition is invalid or contradicted.").unwrap()),
    )
    .unwrap()
    .into()
}

pub(super) fn safety() -> Question {
    Choice::new(
        TextEntry::text("You are a System One decision engine for input safety.").unwrap(),
        vec![
            ChoiceOption::described("SAFE", "Normal query").unwrap(),
            ChoiceOption::described("UNSAFE", "Jailbreak, toxicity, injection").unwrap(),
        ],
    )
    .unwrap()
    .into()
}

pub(super) fn hypotheses(n: usize) -> Vec<(String, Question)> {
    (0..n)
        .map(|i| {
            (
                format!("h{i}"),
                entailment(&format!("Hypothesis number {i} about the source.")),
            )
        })
        .collect()
}

pub(super) fn request(items: &[(String, Question)]) -> Questions {
    Questions::new(items.iter().map(|(n, q)| (n.as_str(), q.clone()))).unwrap()
}

pub(super) fn answer_of(run: &Run, name: &str) -> Vec<f64> {
    run.questions
        .iter()
        .find(|(n, _)| n == name)
        .unwrap()
        .1
        .clone()
}

pub(super) const SOURCE: &str =
    "A long source text that every hypothesis is checked against, so that the state is not tiny.";

fn prompt_of(s: &ModelScorer, st: &State, qs: &Questions) -> Prompt {
    let types: Vec<&str> = s.question_types.iter().map(String::as_str).collect();
    s.template
        .build(
            st,
            qs,
            &s.tokenizer,
            &RequestLimits {
                model: &s.model,
                max_context: s.max_context,
                question_types: &types,
                support: s.head.as_ref(),
            },
        )
        .unwrap()
}

#[test]
fn limit_zero_is_the_plain_path() {
    let (s, fake) = scorer(0);
    let qs = request(&hypotheses(3));
    let run = s.run(&state(SOURCE), &qs).unwrap();
    let stats = fake.stats();
    // One decode per row with the whole row; nothing saved or restored; no empty-session start.
    let sizes: Vec<usize> = stats.seen_segments.iter().map(|s| s.iter().sum()).collect();
    assert_eq!(stats.seen_segments.iter().map(Vec::len).max(), Some(1));
    assert_eq!(stats.decoded_tokens, sizes.iter().sum::<usize>());
    assert_eq!((stats.snapshots, stats.restores), (0, 0));
    assert_eq!(run.questions.len(), 3);
}

#[test]
fn shared_segments_decoded_once() {
    for k in [1usize, 2, 8, 40] {
        let (s, fake) = scorer(1 << 20);
        let st = state(SOURCE);
        let qs = request(&hypotheses(k));
        let run = s.run(&st, &qs).unwrap();
        let plan = RequestPlan::new(&prompt_of(&s, &st, &qs)).unwrap();
        let stats = fake.stats();
        assert_eq!(stats.decoded_tokens, plan.decoded_tokens(), "k = {k}");
        assert_eq!(stats.sessions, 1);
        assert_eq!(stats.rows, k);
        assert_eq!(run.input_tokens(), run.input_ids.len());
        // Each row is seen whole (shared part + its own), split like its own cuts say.
        for (i, sizes) in stats.seen_segments.iter().enumerate() {
            let want: Vec<usize> = plan
                .row_partition(i)
                .unwrap()
                .iter()
                .map(|r| r.len())
                .collect();
            assert_eq!(sizes, &want, "k = {k}, row {i}");
        }
        if k > 1 {
            let whole: usize = (0..k).map(|i| plan.rows[i].len).sum();
            assert!(stats.decoded_tokens < whole, "k = {k}");
        }
    }
}

#[test]
fn siblings_do_not_change_an_answer() {
    let (s, _) = scorer(1 << 20);
    let st = state(SOURCE);
    let items = hypotheses(16);
    let all = s.run(&st, &request(&items)).unwrap();
    let reversed: Vec<_> = items.iter().rev().cloned().collect();
    let rev = s.run(&st, &request(&reversed)).unwrap();
    for (name, question) in &items {
        let alone = s
            .run(&st, &request(&[(name.clone(), question.clone())]))
            .unwrap();
        assert_eq!(
            answer_of(&alone, name),
            answer_of(&all, name),
            "{name} alone"
        );
        assert_eq!(
            answer_of(&rev, name),
            answer_of(&all, name),
            "{name} reversed"
        );
    }
    // Repeating a question under another name, and mixing tasks, changes nothing for the others.
    let mut mixed = items[..4].to_vec();
    mixed.push(("again".to_string(), items[0].1.clone()));
    mixed.push(("safe".to_string(), safety()));
    let m = s.run(&st, &request(&mixed)).unwrap();
    assert_eq!(answer_of(&m, "again"), answer_of(&all, "h0"));
    for (name, _) in &items[..4] {
        assert_eq!(answer_of(&m, name), answer_of(&all, name), "{name} mixed");
    }
}

#[test]
fn mutant_non_canonical_on_hit_is_caught() {
    // The same checks as above must fail when the shared prefix is decoded as a single segment:
    // that is the proof the fake can tell the partitions apart and the real test is not vacuous.
    let (clean, _) = scorer(1 << 20);
    let (broken, _) = {
        let (s, f) = scorer(1 << 20);
        (s.with_mutant(Mutant::MergedShared), f)
    };
    let st = state(SOURCE);
    let items = hypotheses(4);
    let name = &items[0].0;
    // A neighbour of another task shares nothing with it (no common cut), so the question keeps
    // its own partition; alone it shares all its cuts, which the mutant merges into one segment.
    let mut with_neighbour = items[..1].to_vec();
    with_neighbour.push(("neighbour".to_string(), safety()));
    let alone = |s: &ModelScorer| answer_of(&s.run(&st, &request(&items[..1])).unwrap(), name);
    let mixed = |s: &ModelScorer| answer_of(&s.run(&st, &request(&with_neighbour)).unwrap(), name);
    let together = |s: &ModelScorer| answer_of(&s.run(&st, &request(&items)).unwrap(), name);
    assert_eq!(alone(&clean), mixed(&clean));
    assert_eq!(alone(&clean), together(&clean));
    assert_ne!(alone(&broken), mixed(&broken));
}

#[test]
fn family_without_cuts_takes_the_plain_path() {
    // A template that declares no tasks has no cuts: the plain path, whatever the limit.
    let fake = Arc::new(FakeForward::new(4).partition_sensitive());
    let s = ModelScorer::new(
        "default",
        Box::new(ChatMlLetters::new(None)),
        tokenizer(),
        fake.clone(),
        letters(),
        Calibration::new(1.0, true).unwrap(),
        100_000,
        TYPES.iter().map(|t| (*t).to_string()).collect(),
    )
    .with_prefix_cache(1 << 20);
    let q: Question = Choice::new(
        TextEntry::text("Pick one.").unwrap(),
        vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
    )
    .unwrap()
    .into();
    s.run(&state("hello"), &Questions::new([("q", q)]).unwrap())
        .unwrap();
    let stats = fake.stats();
    assert_eq!((stats.snapshots, stats.restores), (0, 0));
    assert_eq!(stats.seen_segments.iter().map(Vec::len).max(), Some(1));
}

/// A `kev` scorer (hidden states) over a partition-sensitive fake.
pub(super) fn kev_scorer(limit: u64) -> (ModelScorer, Arc<FakeForward>) {
    let fake = Arc::new(FakeForward::new(8).partition_sensitive());
    let s = ModelScorer::new(
        "test-kev",
        Box::new(crate::prompt::kev::KevV1),
        testkit::kev(),
        fake.clone(),
        Box::new(crate::heads::test_head::LastTokenHead { max_options: 4 }),
        Calibration::new(1.0, true).unwrap(),
        10_000,
        vec!["choice".to_string(), "noul".to_string()],
    )
    .with_prefix_cache(limit);
    (s, fake)
}

#[test]
fn kev_in_the_segmented_path_equals_the_plain_one_bit_for_bit() {
    // `kev` has one implicit cut (the end of the state), which is also how the plain path splits
    // the row: the two paths give the very same numbers.
    let make = kev_scorer;
    let choice = |i: usize| -> (String, Question) {
        (
            format!("q{i}"),
            Choice::new(
                TextEntry::text(&format!("Which one {i}?")).unwrap(),
                vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
            )
            .unwrap()
            .into(),
        )
    };
    let items: Vec<_> = (0..5).map(choice).collect();
    let (plain, plain_fake) = make(0);
    let (seg, seg_fake) = make(1 << 20);
    let a = plain.run(&state("some state"), &request(&items)).unwrap();
    let b = seg.run(&state("some state"), &request(&items)).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        plain_fake.stats().decoded_tokens,
        seg_fake.stats().decoded_tokens
    );
}

#[test]
fn nan_on_the_segmented_path_names_the_question() {
    // The fake injects a non-finite value into hidden states: a pointer-head (kev) scorer.
    let (s, fake) = kev_scorer(1 << 20);
    fake.set_faults(Faults {
        nan_row: Some(1),
        ..Faults::default()
    });
    let items: Vec<(String, Question)> = (0..3)
        .map(|i| {
            (
                format!("q{i}"),
                Choice::new(
                    TextEntry::text(&format!("Which one {i}?")).unwrap(),
                    vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
                )
                .unwrap()
                .into(),
            )
        })
        .collect();
    let err = s.run(&state("some state"), &request(&items)).unwrap_err();
    match err {
        Error::NonFiniteHidden { question, .. } => assert_eq!(question, "q1"),
        other => panic!("{other}"),
    }
}
