//! The answer-letters head inside the whole path, with a fake model (spec 010b, AC-5 and
//! AC-7 to AC-10): template `spark-decisions` -> engine -> head -> `ask`.

use std::sync::{Arc, Mutex};

use super::AnswerLettersHead;
use crate::ask::ask;
use crate::engine::fake::FakeForward;
use crate::engine::{Forward, Output, OutputSpec, Session, pick_logits};
use crate::error::{Category, Error, Result};
use crate::model_scorer::ModelScorer;
use crate::models::manifest::Target;
use crate::prompt::spark::SparkDecisionsV3;
use crate::prompt::tokenize::testkit;
use crate::rng::SplitMix64;
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, Score, State, StateValue, TextEntry, YesNo,
};
use crate::{Answer, Calibration};

/// Ids of `A`...`Z` in the test model: 100, 101, ... (any distinct ids would do).
const FIRST_LETTER: u32 = 100;

fn letters(count: u32) -> Vec<Target> {
    (0..count)
        .map(|i| Target {
            label: char::from(b'A' + u8::try_from(i).unwrap()).to_string(),
            id: FIRST_LETTER + i,
        })
        .collect()
}

fn tokenizer() -> crate::prompt::PromptTokenizer {
    testkit::byte_level(&[
        ("bos", "<bos>"),
        ("system", "<system>"),
        ("user", "<user>"),
        ("bot", "<bot>"),
        ("eos", "<eos>"),
        ("think_end", "<think_end>"),
    ])
}

fn scorer_with(
    forward: Arc<dyn Forward>,
    head: AnswerLettersHead,
    temperature: f64,
) -> ModelScorer {
    ModelScorer::new(
        "spark-test",
        Box::new(SparkDecisionsV3::new(vec![
            "<bos>".into(),
            "<system>".into(),
            "<user>".into(),
            "<bot>".into(),
            "<eos>".into(),
            "<think_end>".into(),
        ])),
        tokenizer(),
        forward,
        Box::new(head),
        Calibration::new(temperature, true).unwrap(),
        4096,
        vec!["choice".to_string(), "noul".to_string()],
    )
}

fn scorer(forward: Arc<dyn Forward>, temperature: f64) -> ModelScorer {
    scorer_with(
        forward,
        AnswerLettersHead::new(&letters(26), 26),
        temperature,
    )
}

fn entry(s: &str) -> TextEntry {
    TextEntry::text(s).unwrap()
}

fn choice(options: usize) -> Question {
    Choice::new(
        entry("Which one?"),
        (0..options)
            .map(|i| {
                ChoiceOption::described(format!("key{i}"), &format!("option number {i}")).unwrap()
            })
            .collect(),
    )
    .unwrap()
    .into()
}

fn yes_no() -> Question {
    YesNo::new(
        entry("Is it so?"),
        Some(entry("it is so")),
        Some(entry("it is not")),
    )
    .unwrap()
    .into()
}

fn state() -> State {
    State::new(StateValue::string("A man is playing a guitar.")).unwrap()
}

fn probabilities(answer: &Answer) -> Vec<f64> {
    let Answer::Choice(c) = answer else {
        panic!("a choice answer")
    };
    c.probabilities().iter().map(|(_, p)| *p).collect()
}

// ---- AC-7: the letters of each question, in order ---------------------------------------------

#[test]
fn each_question_reads_its_own_letters_in_order() {
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer(fake.clone(), 1.0);
    let qs = Questions::new([
        ("three", choice(3)),
        ("two", choice(2)),
        ("yn", yes_no()),
        ("all", choice(26)),
    ])
    .unwrap();
    let run = s.run(&state(), &qs).unwrap();
    let stats = fake.stats();
    assert_eq!(stats.rows, 4);
    let ids = |k: u32| (FIRST_LETTER..FIRST_LETTER + k).collect::<Vec<u32>>();
    // a Choice of K options reads the first K letters; a yes/no question reads A and B
    for (i, k) in [3u32, 2, 2, 26].into_iter().enumerate() {
        let row = &stats.seen_rows[i];
        assert_eq!(
            stats.seen_outputs[i],
            vec![OutputSpec::Logits {
                position: row.len() - 1,
                ids: ids(k)
            }],
            "question {i}"
        );
        // the numbers are what the fake says for those ids at the last position of the row
        let by_hand: Vec<f64> = ids(k)
            .iter()
            .map(|&id| f64::from(fake.logit_of(row, id)))
            .collect();
        assert_eq!(run.questions[i].1, by_hand, "question {i}");
    }
    // the position is the last token of the row, and the row ends where the template says
    let think_end = tokenizer().roles().id("think_end").unwrap();
    assert!(stats.seen_rows.iter().all(|r| r.last() == Some(&think_end)));
}

/// A model that answers, for every row, these logits over a vocabulary: the letters get the
/// values of the test, every other token gets `rest`.
struct Vocabulary {
    letters: Mutex<Vec<f32>>,
    rest: f32,
    size: usize,
}

struct VocabularySession<'a>(&'a Vocabulary);

impl Forward for Vocabulary {
    fn begin<'a>(&'a self, _state: &[u32]) -> Result<Box<dyn Session + 'a>> {
        Ok(Box::new(VocabularySession(self)))
    }
}

impl Session for VocabularySession<'_> {
    fn row(&mut self, _branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        let letters = self.0.letters.lock().unwrap().clone();
        let mut row = vec![self.0.rest; self.0.size];
        for (i, v) in letters.iter().enumerate() {
            row[FIRST_LETTER as usize + i] = *v;
        }
        outputs
            .iter()
            .map(|o| match o {
                OutputSpec::Logits { position, ids } => {
                    Ok(Output::Logits(pick_logits(&row, ids, *position)?))
                }
                _ => Ok(Output::Hidden(Vec::new())),
            })
            .collect()
    }
}

fn restricted_softmax(z: &[f64], t: f64) -> Vec<f64> {
    let scaled: Vec<f64> = z.iter().map(|v| v / t).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// The mutant: the softmax over the whole vocabulary, read at the letters.
fn full_vocabulary_softmax(row: &[f32], ids: &[usize], t: f64) -> Vec<f64> {
    let all = restricted_softmax(&row.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), t);
    ids.iter().map(|&i| all[i]).collect()
}

#[test]
fn a_full_vocabulary_softmax_is_a_different_number() {
    let (size, rest) = (200usize, 9.0f32);
    let values = [3.0f32, 2.0, 1.0];
    let model = Arc::new(Vocabulary {
        letters: Mutex::new(values.to_vec()),
        rest,
        size,
    });
    let s = scorer(model, 1.5);
    let qs = Questions::new([("q", choice(3))]).unwrap();
    let got = probabilities(ask(&s, &state(), &qs).unwrap().get("q").unwrap());
    let z: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
    let want = restricted_softmax(&z, 1.5);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "{g} vs {w}");
    }
    assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // what the mutant would answer is far from that: the test can fail
    let mut row = vec![rest; size];
    for (i, v) in values.iter().enumerate() {
        row[FIRST_LETTER as usize + i] = *v;
    }
    let ids: Vec<usize> = (0..3).map(|i| FIRST_LETTER as usize + i).collect();
    let mutant = full_vocabulary_softmax(&row, &ids, 1.5);
    assert!((mutant.iter().sum::<f64>() - 1.0).abs() > 0.5, "{mutant:?}");
    assert!((mutant[0] - got[0]).abs() > 0.1, "{mutant:?} vs {got:?}");
}

// ---- AC-8: the formula of the reference ---------------------------------------------------------

/// `decisions.softmax` of the reference, written from its Python text: a check that there are at
/// least two finite values, `maximum = max(logits)`, `weights = exp((x - maximum) / T)`, divide by
/// the total. Plain `f64` loops; it shares nothing with `calibrated_softmax`.
fn reference_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    assert!(temperature.is_finite() && temperature > 0.0);
    assert!(logits.len() >= 2 && logits.iter().all(|x| x.is_finite()));
    let mut maximum = logits[0];
    for &x in logits {
        if x > maximum {
            maximum = x;
        }
    }
    let mut weights = Vec::with_capacity(logits.len());
    for &x in logits {
        weights.push(((x - maximum) / temperature).exp());
    }
    let mut total = 0.0;
    for w in &weights {
        total += w;
    }
    weights.iter().map(|w| w / total).collect()
}

#[test]
fn probabilities_equal_the_formula_of_the_reference() {
    let model = Arc::new(Vocabulary {
        letters: Mutex::new(Vec::new()),
        rest: 0.0,
        size: 256,
    });
    let mut rng = SplitMix64::new(0x00C0_FFEE);
    let unit = |rng: &mut SplitMix64| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut worst: f64 = 0.0;
    let mut cases = 0u32;
    for t in [0.5, 1.0, 2.3] {
        let s = scorer(model.clone(), t);
        for case in 0..400u32 {
            let k = 2 + (rng.below(25) as usize);
            // logits as the engine returns them (f32); four shapes: ordinary, huge, almost equal,
            // and a clear winner
            let base = (unit(&mut rng) as f32 - 0.5) * 40.0;
            let z: Vec<f32> = (0..k)
                .map(|i| match case % 4 {
                    0 => (unit(&mut rng) as f32 - 0.5) * 20.0,
                    1 => (unit(&mut rng) as f32 - 0.5) * 2000.0,
                    2 => base + (i as f32) * 1e-6,
                    _ => {
                        if i == 0 {
                            base + 30.0
                        } else {
                            base - unit(&mut rng) as f32 * 5.0
                        }
                    }
                })
                .collect();
            *model.letters.lock().unwrap() = z.clone();
            let qs = Questions::new([("q", choice(k))]).unwrap();
            let got = probabilities(ask(&s, &state(), &qs).unwrap().get("q").unwrap());
            let wide: Vec<f64> = z.iter().map(|v| f64::from(*v)).collect();
            let want = reference_softmax(&wide, t);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max((g - w).abs());
            }
            cases += 1;
        }
    }
    eprintln!("largest gap with the reference formula: {worst:e} over {cases} vectors");
    assert!(cases >= 1_000);
    assert!(
        worst <= 1e-9,
        "the gap {worst:e} needs an explanation before any threshold moves"
    );
}

// ---- AC-5: limits ------------------------------------------------------------------------------

#[test]
fn more_options_than_letters_is_refused_before_the_model_runs() {
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer_with(fake.clone(), AnswerLettersHead::new(&letters(26), 4), 1.0);
    let qs = Questions::new([("ok", choice(4)), ("q", choice(5))]).unwrap();
    let err = ask(&s, &state(), &qs).unwrap_err();
    assert_eq!(err.category(), Category::Unsupported);
    let Error::Unsupported(u) = &err else {
        panic!("{err}")
    };
    assert_eq!(
        *u,
        crate::prompt::Unsupported::HeadLimit {
            question: "q".into(),
            model: "spark-test".into(),
            max: 4,
            got: 5
        }
    );
    assert!(err.to_string().contains("at most 4") && err.to_string().contains("got 5"));
    // nothing ran for the question that fits either: the call fails as a whole
    assert_eq!(fake.stats().rows, 0);
}

#[test]
fn a_letter_without_an_id_is_a_limit_of_its_own_never_a_made_up_id() {
    // a manifest keeps max_options within the letters; if it ever did not, the head still stops
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer_with(fake.clone(), AnswerLettersHead::new(&letters(3), 10), 1.0);
    let qs = Questions::new([("q", choice(4))]).unwrap();
    let err = ask(&s, &state(), &qs).unwrap_err();
    let Error::Unsupported(u) = &err else {
        panic!("{err}")
    };
    assert!(
        matches!(
            u,
            crate::prompt::Unsupported::HeadLimit { max: 3, got: 4, .. }
        ),
        "{u}"
    );
    assert_eq!(fake.stats().rows, 0);
}

#[test]
fn a_score_question_is_refused_by_the_head_too() {
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer(fake.clone(), 1.0);
    let score: Question = Score::new(entry("How much?"), vec![entry("low"), entry("high")])
        .unwrap()
        .into();
    let err = ask(&s, &state(), &Questions::new([("q", score)]).unwrap()).unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(crate::prompt::Unsupported::QuestionType { kind, .. }) if kind == "score"),
        "{err}"
    );
    assert_eq!(fake.stats().rows, 0);
}

// ---- AC-9: nothing that is not a number gets to be a probability -------------------------------

#[test]
fn a_value_that_is_not_finite_anywhere_in_the_row_is_refused() {
    let size = 131_072usize;
    let ids = [FIRST_LETTER, FIRST_LETTER + 1];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // at the first, a middle and the last position of a row of the size of the vocabulary,
        // on a letter and off the letters
        for at in [0usize, 1, 100, 101, 50_000, size - 1] {
            let mut row = vec![0.5f32; size];
            row[at] = bad;
            let err = pick_logits(&row, &ids, 7).unwrap_err();
            assert_eq!(err.category(), Category::Numerical, "{bad} at {at}");
            assert!(
                matches!(err, Error::NonFiniteHidden { position: 7, .. }),
                "{err}"
            );
        }
    }
    // a clean row is read
    let row = vec![0.5f32; size];
    assert_eq!(pick_logits(&row, &ids, 7).unwrap(), vec![0.5, 0.5]);
}

/// The mutant of the check: look only at the values that will be used.
fn pick_only_the_letters(row: &[f32], ids: &[u32]) -> Option<Vec<f32>> {
    let picked: Vec<f32> = ids.iter().map(|&i| row[i as usize]).collect();
    picked.iter().all(|v| v.is_finite()).then_some(picked)
}

#[test]
fn the_finite_check_is_not_limited_to_the_letters() {
    let size = 4096usize;
    let ids = [FIRST_LETTER, FIRST_LETTER + 1];
    let mut row = vec![0.5f32; size];
    row[2000] = f32::NAN; // not a letter
    // the engine refuses the row; a check on the letters only would take it
    assert!(pick_logits(&row, &ids, 0).is_err());
    assert_eq!(pick_only_the_letters(&row, &ids), Some(vec![0.5, 0.5]));
}

#[test]
fn a_letter_that_is_not_finite_never_becomes_a_probability() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let model = Arc::new(Vocabulary {
            letters: Mutex::new(vec![1.0, bad]),
            rest: 0.0,
            size: 200,
        });
        let s = scorer(model, 1.0);
        let qs = Questions::new([("q", choice(2))]).unwrap();
        let err = ask(&s, &state(), &qs).unwrap_err();
        assert_eq!(err.category(), Category::Numerical, "{bad}");
        assert!(err.to_string().contains("\"q\""), "{err}");
    }
}

// ---- AC-10: it plugs in without a change in the engine ----------------------------------------

#[test]
fn the_head_plugs_in_without_touching_the_engine() {
    // the same `Forward`, `Session`, `ModelScorer` and `ask` of the other heads: this file
    // compiles against the engine as it was, and the whole path answers
    let fake = Arc::new(FakeForward::new(4));
    let s = scorer(fake.clone(), 2.0);
    let qs = Questions::new([("a", choice(3)), ("b", yes_no())]).unwrap();
    let answers = ask(&s, &state(), &qs).unwrap();
    let p = probabilities(answers.get("a").unwrap());
    assert_eq!(p.len(), 3);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let Answer::YesNo(yn) = answers.get("b").unwrap() else {
        panic!("a yes/no answer")
    };
    let q = yn.probability();
    assert!((0.0..=1.0).contains(&q));
}

#[test]
fn the_head_names_itself_and_its_limit() {
    use super::Head;
    let h = AnswerLettersHead::new(&letters(26), 20);
    assert_eq!(h.kind(), "answer_letters");
    assert_eq!(h.max_options(), 20);
    // the limit never exceeds the letters it has
    assert_eq!(AnswerLettersHead::new(&letters(3), 20).max_options(), 3);
}
