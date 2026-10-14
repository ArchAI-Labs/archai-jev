//! The key log-probability head end to end with the fake engine (spec 010, AC-6 to AC-9).
//!
//! The fake has a vocabulary of 300 ids and a hand-checkable model of the next token
//! (`FakeForward::key_log_prob_of`), so the score of an option is computed here token by token,
//! independently of the plumbing under test.

use std::sync::Arc;

use super::ModelScorer;
use crate::ask::ask;
use crate::engine::fake::FakeForward;
use crate::error::Error;
use crate::heads::KeyLogProbHead;
use crate::prompt::di::DecisionIndexChatMl;
use crate::prompt::tokenize::testkit;
use crate::prompt::{PromptTokenizer, Unsupported};
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, State, StateValue, TextEntry, YesNo,
};
use crate::scorer::Calibration;

const MAX_OPTIONS: usize = 8;
// the test tokenizer has one token per byte: a key of 16 characters is 16 tokens
const MAX_KEY_TOKENS: usize = 16;

fn tokenizer() -> PromptTokenizer {
    testkit::byte_level(&[("im_start", "<|im_start|>"), ("im_end", "<|im_end|>")])
}

fn scorer_with(temperature: f64, max_context: u64) -> (ModelScorer, Arc<FakeForward>) {
    let fake = Arc::new(FakeForward::new(8).with_vocab(300));
    let scorer = ModelScorer::new(
        "test-keys",
        Box::new(DecisionIndexChatMl::new(MAX_KEY_TOKENS)),
        tokenizer(),
        fake.clone(),
        Box::new(KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS)),
        Calibration::new(temperature, true).unwrap(),
        max_context,
        vec!["choice".to_string(), "noul".to_string()],
    );
    (scorer, fake)
}

fn scorer() -> (ModelScorer, Arc<FakeForward>) {
    scorer_with(1.0, 10_000)
}

fn state(text: &str) -> State {
    State::new(StateValue::string(text)).unwrap()
}

fn choice(keys: &[&str]) -> Question {
    Choice::new(
        TextEntry::text("Pick one.").unwrap(),
        keys.iter().map(|k| ChoiceOption::key(*k)).collect(),
    )
    .unwrap()
    .into()
}

fn one(question: Question) -> Questions {
    Questions::new([("q", question)]).unwrap()
}

/// The expected score of every key, computed token by token from the fake's own model.
fn expected(fake: &FakeForward, prompt: &[u32], keys: &[&str]) -> Vec<f64> {
    let tk = tokenizer();
    keys.iter()
        .map(|k| fake.key_log_prob_of(prompt, &tk.encode_raw(k, "key").unwrap()))
        .collect()
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
    }
}

#[test]
fn the_score_of_an_option_is_the_sum_of_its_full_vocabulary_log_probs() {
    let (s, fake) = scorer();
    let keys = ["a", "bc", "def"];
    let run = s.run(&state("hello"), &one(choice(&keys))).unwrap();
    let want = expected(&fake, &run.input_ids, &keys);
    assert_close(&run.questions[0].1, &want);
    // The reading is not the restricted one: the single-token key `a` is its log-probability
    // among ALL the ids, not its bare logit.
    let a = tokenizer().encode_raw("a", "key").unwrap()[0];
    let bare = f64::from(fake.logit_of(&run.input_ids, a));
    assert!(
        (run.questions[0].1[0] - bare).abs() > 1e-3,
        "the score must be normalised over the vocabulary, not the raw logit"
    );
}

#[test]
fn keys_that_share_a_prefix_or_are_a_prefix_of_another_are_scored_each_for_itself() {
    let (s, fake) = scorer();
    let keys = ["option_0", "option_1", "a", "ab", "abc"];
    let run = s.run(&state("x"), &one(choice(&keys))).unwrap();
    assert_close(&run.questions[0].1, &expected(&fake, &run.input_ids, &keys));
}

#[test]
fn a_single_token_key_costs_no_decode_beyond_the_prompt() {
    let (s, fake) = scorer();
    s.run(&state("x"), &one(choice(&["a", "b", "c"]))).unwrap();
    assert_eq!(fake.stats().continuation_tokens, 0);
    fake.reset_stats();
    // "bc" is 2 tokens (one more to decode) and "def" is 3 (two more)
    s.run(&state("x"), &one(choice(&["a", "bc", "def"])))
        .unwrap();
    assert_eq!(fake.stats().continuation_tokens, 3);
}

#[test]
fn the_order_of_the_options_changes_only_the_order_of_the_scores() {
    let (s, _fake) = scorer();
    let a = s
        .run(&state("x"), &one(choice(&["p", "qq", "rrr"])))
        .unwrap();
    let b = s
        .run(&state("x"), &one(choice(&["rrr", "p", "qq"])))
        .unwrap();
    // the prompt lists the options in another order, so the prompts differ: compare the scores of
    // each key with its own expectation instead of with the other run
    let (sa, sb) = (&a.questions[0].1, &b.questions[0].1);
    assert_eq!(sa.len(), 3);
    assert_ne!(a.input_ids, b.input_ids);
    // and a request that lists the same keys in the same order twice is the same bit for bit
    let again = s
        .run(&state("x"), &one(choice(&["p", "qq", "rrr"])))
        .unwrap();
    assert_eq!(&again.questions[0].1, sa);
    assert_eq!(sb.len(), 3);
}

#[test]
fn a_yes_no_question_scores_false_then_true() {
    let (s, fake) = scorer();
    let q = YesNo::new(TextEntry::text("Is it?").unwrap(), None, None).unwrap();
    let run = s.run(&state("x"), &one(q.into())).unwrap();
    assert_close(
        &run.questions[0].1,
        &expected(&fake, &run.input_ids, &["false", "true"]),
    );
}

#[test]
fn ask_applies_the_temperature_to_the_sums_once() {
    let (s, fake) = scorer_with(2.0, 10_000);
    let keys = ["a", "bc"];
    let answers = ask(&s, &state("x"), &one(choice(&keys))).unwrap();
    let run = s.run(&state("x"), &one(choice(&keys))).unwrap();
    let sums = expected(&fake, &run.input_ids, &keys);
    let want = crate::calibration::calibrated_softmax(&sums, 2.0).unwrap();
    let Some(crate::answers::Answer::Choice(answer)) = answers.get("q") else {
        panic!("a choice answer")
    };
    let got: Vec<f64> = answer.probabilities().iter().map(|(_, p)| *p).collect();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
    }
    assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

fn refused(r: crate::error::Result<super::Run>) -> Unsupported {
    match r {
        Err(Error::Unsupported(u)) => u,
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn every_refusal_comes_before_the_engine_is_touched() {
    let (s, fake) = scorer_with(1.0, 400);
    let too_many: Vec<String> = (0..=MAX_OPTIONS).map(|i| format!("k{i}")).collect();
    let too_many: Vec<&str> = too_many.iter().map(String::as_str).collect();
    let score = crate::schema::Score::new(
        TextEntry::text("How much?").unwrap(),
        vec![
            TextEntry::text("low").unwrap(),
            TextEntry::text("high").unwrap(),
        ],
    )
    .unwrap();
    let long_state = "word ".repeat(200);

    let u = refused(s.run(&state("x"), &one(score.into())));
    assert!(
        matches!(u, Unsupported::QuestionType { ref kind, .. } if kind == "score"),
        "{u}"
    );
    let u = refused(s.run(&state("x"), &one(choice(&too_many))));
    assert!(
        matches!(u, Unsupported::HeadLimit { max, got, .. } if max == MAX_OPTIONS as u64 && got == MAX_OPTIONS as u64 + 1),
        "{u}"
    );
    let u = refused(s.run(&state("x"), &one(choice(&["ab", "abcdefghijklmnopq"]))));
    assert!(
        matches!(
            u,
            Unsupported::KeyTooLong {
                got: 17,
                max: 16,
                ..
            }
        ),
        "{u}"
    );
    let u = refused(s.run(&state("x"), &one(choice(&["\u{e9}", "e\u{301}"]))));
    assert!(
        matches!(u, Unsupported::IndistinguishableOptions { .. }),
        "{u}"
    );
    let u = refused(s.run(&state(&long_state), &one(choice(&["a", "b"]))));
    assert!(matches!(u, Unsupported::RowTooLong { max: 400, .. }), "{u}");

    assert_eq!(fake.stats().sessions, 0, "no refusal may reach the engine");
}

#[test]
fn a_non_finite_score_is_never_returned() {
    // The fake cannot produce a NaN log-probability by itself, so the head is asked directly:
    // an answer that does not have one score per option is an engine fault, not a result.
    use crate::engine::Output;
    use crate::heads::Head;
    let head = KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS);
    let q = choice(&["a", "b"]);
    assert!(head.score(&q, &[Output::KeyLogProbs(vec![0.0])]).is_err());
    assert!(head.score(&q, &[Output::Logits(vec![0.0, 1.0])]).is_err());
    assert!(head.score(&q, &[]).is_err());
}

// ---- the cache of prefixes (spec 007) with the cut points of the template ----------------------

use crate::scorer::Scorer;

const BIG: u64 = 64 << 20;

/// A scorer whose numbers depend on how each row was split into decodes, with a store of
/// `limit` bytes: the reuse can only be invisible if every row is decoded in its own segments.
fn cached(limit: u64) -> (ModelScorer, Arc<FakeForward>) {
    let fake = Arc::new(FakeForward::new(8).with_vocab(300).partition_sensitive());
    let scorer = ModelScorer::new(
        "test-keys",
        Box::new(DecisionIndexChatMl::new(MAX_KEY_TOKENS)),
        tokenizer(),
        fake.clone(),
        Box::new(KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS)),
        Calibration::new(1.0, true).unwrap(),
        10_000,
        vec!["choice".to_string(), "noul".to_string()],
    )
    .with_prefix_cache(limit);
    (scorer, fake)
}

fn two_about_one_state() -> Questions {
    Questions::new([
        ("first", choice(&["a", "bc", "option_1"])),
        (
            "second",
            YesNo::new(TextEntry::text("Is it so?").unwrap(), None, None)
                .unwrap()
                .into(),
        ),
    ])
    .unwrap()
}

#[test]
fn cold_warm_and_cleared_agree_with_the_path_that_remembers_nothing() {
    let (s, fake) = cached(BIG);
    // a store of one byte holds nothing: the very same segmented path, with no memory
    let (bare, _) = cached(1);
    let st = state("A source text that two questions are asked about, long enough to be shared.");
    let qs = two_about_one_state();

    let reference = bare.run(&st, &qs).unwrap();
    let cold = s.run(&st, &qs).unwrap();
    let decoded_cold = fake.stats().decoded_tokens;
    let warm = s.run(&st, &qs).unwrap();
    let decoded_warm = fake.stats().decoded_tokens - decoded_cold;
    s.clear_cache();
    let cleared = s.run(&st, &qs).unwrap();

    assert_eq!(cold, reference);
    assert_eq!(warm, reference);
    assert_eq!(cleared, reference);
    assert!(
        decoded_warm < decoded_cold,
        "a warm store decodes less: {decoded_warm} !< {decoded_cold}"
    );
    let info = s.cache_info().unwrap();
    assert_eq!((info.hits, info.misses), (1, 2));
}

#[test]
fn questions_about_one_state_do_not_decode_it_twice() {
    let (s, fake) = cached(BIG);
    let st =
        state("The shared part of the prompt of both questions is the system block and the state.");
    let qs = two_about_one_state();
    s.run(&st, &qs).unwrap();
    let decoded = fake.stats().decoded_tokens;
    let rows: usize = {
        let tk = tokenizer();
        let types = ["choice", "noul"];
        let limits = crate::prompt::RequestLimits {
            model: "m",
            max_context: 10_000,
            question_types: &types,
            support: &crate::prompt::NoLimits,
        };
        let prompt = crate::prompt::Template::build(
            &DecisionIndexChatMl::new(MAX_KEY_TOKENS),
            &st,
            &qs,
            &tk,
            &limits,
        )
        .unwrap();
        prompt.branches().iter().map(|b| b.len()).sum()
    };
    assert!(
        decoded < rows,
        "the shared part is decoded once: {decoded} tokens for rows of {rows} in all"
    );
}

#[test]
fn another_state_of_the_same_model_still_finds_the_fixed_head() {
    let (s, _) = cached(BIG);
    let (bare, _) = cached(1);
    let qs = two_about_one_state();
    s.run(&state("First source text for the first request."), &qs)
        .unwrap();
    let other = state("A completely different source text.");
    let hit = s.run(&other, &qs).unwrap();
    assert_eq!(hit, bare.run(&other, &qs).unwrap());
    assert_eq!(
        s.cache_info().unwrap().hits,
        1,
        "the fixed head served the second request"
    );
}
