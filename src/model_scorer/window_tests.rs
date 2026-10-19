//! A family with a sliding attention window keeps no prefix longer than the window between
//! requests (spec 010b, AC-15 and the limit `snapshot_max_tokens`).
//!
//! The fake here is **lossy over the window**: a snapshot of more tokens than the window restores
//! to numbers other than a cold decode gives, which is what the real model does (measured in the
//! spike). "Cold, warm and no cache agree" is then a claim these tests can fail: they do, with the
//! mutant that saves beyond the limit.

use std::sync::Arc;

use super::*;
use crate::engine::fake::FakeForward;
use crate::heads::AnswerLettersHead;
use crate::models::manifest::Target;
use crate::prompt::spark::SparkDecisionsV3;
use crate::prompt::tokenize::testkit;
use crate::schema::{Choice, ChoiceOption, Question, StateValue, TextEntry};

const WINDOW: u64 = 700;
const BIG: u64 = 64 << 20;

fn letters() -> Box<dyn Head> {
    let targets: Vec<Target> = (0..4u32)
        .map(|i| Target {
            label: char::from(b'A' + u8::try_from(i).unwrap()).to_string(),
            id: 100 + i,
        })
        .collect();
    Box::new(AnswerLettersHead::new(&targets, 4))
}

fn scorer(cache: u64, limit: Option<u64>, lossy: bool) -> (ModelScorer, Arc<FakeForward>) {
    let mut fake = FakeForward::new(4).partition_sensitive();
    if lossy {
        fake = fake.lossy_snapshots_over(usize::try_from(WINDOW).unwrap());
    }
    let fake = Arc::new(fake);
    let s = ModelScorer::new(
        "windowed",
        Box::new(SparkDecisionsV3::new(vec!["<bos>".to_string()])),
        testkit::byte_level(&[
            ("bos", "<bos>"),
            ("system", "<system>"),
            ("user", "<user>"),
            ("bot", "<bot>"),
            ("eos", "<eos>"),
            ("think_end", "<think_end>"),
        ]),
        fake.clone(),
        letters(),
        Calibration::new(1.0, true).unwrap(),
        100_000,
        vec!["choice".to_string(), "noul".to_string()],
    )
    .with_snapshot_limit(limit)
    .with_prefix_cache(cache);
    (s, fake)
}

fn state_of(words: usize) -> State {
    let text = (0..words)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    State::new(StateValue::string(text)).unwrap()
}

fn questions() -> Questions {
    let q = |instructions: &str| -> Question {
        Choice::new(
            TextEntry::text(instructions).unwrap(),
            vec![
                ChoiceOption::described("a", "the first").unwrap(),
                ChoiceOption::described("b", "the second").unwrap(),
            ],
        )
        .unwrap()
        .into()
    };
    Questions::new([("one", q("Which one?")), ("two", q("And which other one?"))]).unwrap()
}

fn info(s: &ModelScorer) -> CacheInfo {
    s.cache_info().unwrap()
}

/// How long the shared part of this request is, in tokens (up to the end of the state).
fn content_cut(s: &ModelScorer, st: &State, qs: &Questions) -> usize {
    let types: Vec<&str> = s.question_types.iter().map(String::as_str).collect();
    let p = s
        .template
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
        .unwrap();
    p.row_cuts(0).unwrap().last().unwrap().pos
}

#[test]
fn a_shared_prefix_longer_than_the_window_is_decoded_once_but_never_kept() {
    let (s, fake) = scorer(BIG, Some(WINDOW), true);
    let (bare, _) = scorer(1, Some(WINDOW), true);
    let st = state_of(400);
    let qs = questions();
    assert!(
        content_cut(&s, &st, &qs) as u64 > WINDOW,
        "the state is longer than the window"
    );

    let reference = bare.run(&st, &qs).unwrap();
    let cold = s.run(&st, &qs).unwrap();
    let after_cold = (fake.stats().snapshots, info(&s).entries);
    let warm = s.run(&st, &qs).unwrap();
    // no snapshot of the long state, so nothing to restore and nothing that differs
    assert_eq!(
        fake.stats().restores,
        fake.stats().restores.min(1),
        "only the head may be restored"
    );
    assert_eq!(cold, reference);
    assert_eq!(warm, reference);
    // what was saved is the fixed head, which is shorter than the window
    assert!(info(&s).entries <= 1, "{} entries", info(&s).entries);
    assert!(after_cold.0 <= 1, "{} snapshots", after_cold.0);
}

#[test]
fn a_prefix_inside_the_window_is_kept_and_restored_with_the_numbers_of_a_cold_decode() {
    let (s, fake) = scorer(BIG, Some(WINDOW), true);
    let (bare, _) = scorer(1, Some(WINDOW), true);
    let st = state_of(8);
    let qs = questions();
    assert!((content_cut(&s, &st, &qs) as u64) <= WINDOW);
    let reference = bare.run(&st, &qs).unwrap();
    let cold = s.run(&st, &qs).unwrap();
    assert!(info(&s).entries >= 1, "the head and the state were saved");
    let warm = s.run(&st, &qs).unwrap();
    assert_eq!(info(&s).hits, 1);
    assert!(fake.stats().restores >= 1);
    assert_eq!(cold, reference);
    assert_eq!(warm, reference, "restored inside the window: the same bits");
}

#[test]
fn the_fixed_head_is_kept_even_when_the_state_is_beyond_the_window() {
    let (s, fake) = scorer(BIG, Some(WINDOW), true);
    let qs = questions();
    s.run(&state_of(400), &qs).unwrap();
    assert!(
        fake.stats().snapshots >= 1,
        "the head is shorter than the window"
    );
    // another long state starts from the saved head
    let hits_before = info(&s).hits;
    s.run(&state_of(500), &qs).unwrap();
    assert_eq!(info(&s).hits, hits_before + 1);
}

#[test]
fn without_the_limit_the_lossy_model_gives_other_numbers_when_warm() {
    // the claim of the tests above can fail: the same request, no limit, a model that cannot
    // restore beyond its window
    let (s, fake) = scorer(BIG, None, true);
    let (bare, _) = scorer(1, None, true);
    let st = state_of(400);
    let qs = questions();
    let reference = bare.run(&st, &qs).unwrap();
    let cold = s.run(&st, &qs).unwrap();
    let warm = s.run(&st, &qs).unwrap();
    assert!(fake.stats().restores >= 1);
    assert_eq!(cold, reference);
    assert_ne!(
        warm, reference,
        "a restore beyond the window changes the numbers"
    );
}

#[test]
fn a_scorer_that_saves_beyond_the_limit_is_noticed() {
    // the mutant: the limit is ignored
    let (s, _) = scorer(BIG, Some(WINDOW), true);
    let (bare, _) = scorer(1, Some(WINDOW), true);
    let s = s.with_mutant(Mutant::KeepsBeyondTheLimit);
    let st = state_of(400);
    let qs = questions();
    let reference = bare.run(&st, &qs).unwrap();
    s.run(&st, &qs).unwrap();
    let warm = s.run(&st, &qs).unwrap();
    assert_ne!(
        warm, reference,
        "the mutant must make the identity test fail"
    );
}

#[test]
fn a_family_with_no_window_keeps_every_length() {
    let (s, fake) = scorer(BIG, None, false);
    let (bare, _) = scorer(1, None, false);
    let st = state_of(400);
    let qs = questions();
    let reference = bare.run(&st, &qs).unwrap();
    s.run(&st, &qs).unwrap();
    let warm = s.run(&st, &qs).unwrap();
    assert!(fake.stats().restores >= 1);
    assert_eq!(warm, reference);
}
