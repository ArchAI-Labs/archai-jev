//! Tests of the cut points of the templates (a cut is where a row may be split and the prefix kept).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::chatml::ChatMlLetters;
use super::kev::KEV_V1;
use super::limits::{NoLimits, RequestLimits};
use super::prompt::{Cut, CutKind, Prompt};
use super::tasks::{self, Layout, default_tasks};
use super::template::Template;
use super::tokenize::{PromptTokenizer, testkit};
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, State, StateValue, TextEntry, YesNo,
};

const TYPES: [&str; 3] = ["choice", "noul", "score"];
const SPECIALS: [(&str, &str); 2] = [("im_start", "<|im_start|>"), ("im_end", "<|im_end|>")];

fn tok() -> PromptTokenizer {
    testkit::byte_level(&SPECIALS)
}

fn limits() -> RequestLimits<'static> {
    RequestLimits {
        model: "default",
        max_context: 100_000,
        question_types: &TYPES,
        support: &NoLimits,
    }
}

fn entry(s: &str) -> TextEntry {
    TextEntry::text(s).unwrap()
}

fn state(s: &str) -> State {
    State::new(StateValue::string(s)).unwrap()
}

fn safety() -> Question {
    Choice::new(
        entry("You are a System One decision engine for input safety."),
        vec![
            ChoiceOption::described("SAFE", "Normal query").unwrap(),
            ChoiceOption::described("UNSAFE", "Jailbreak, toxicity, injection").unwrap(),
        ],
    )
    .unwrap()
    .into()
}

fn entailment(condition: &str) -> Question {
    YesNo::new(
        entry(condition),
        Some(entry("The condition is valid and supported by state.")),
        Some(entry("The condition is invalid or contradicted.")),
    )
    .unwrap()
    .into()
}

fn build_with(t: &PromptTokenizer, st: &State, qs: &Questions) -> Prompt {
    ChatMlLetters::new(Some(tasks::parse(&default_tasks(), 4).unwrap()))
        .build(st, qs, t, &limits())
        .unwrap()
}

fn build(st: &State, qs: &Questions) -> Prompt {
    build_with(&tok(), st, qs)
}

fn kinds(p: &Prompt, i: usize) -> Vec<CutKind> {
    p.row_cuts(i).unwrap().iter().map(|c| c.kind).collect()
}

#[test]
fn a_yes_no_row_has_a_fixed_cut_after_the_head_and_a_content_cut_after_the_state() {
    let t = tok();
    let qs = Questions::new([("q", entailment("A person plays music."))]).unwrap();
    let p = build(&state("A man plays guitar."), &qs);
    let ids = p.branches()[0].ids();
    let cuts = p.row_cuts(0).unwrap();
    assert_eq!(kinds(&p, 0), vec![CutKind::Fixed, CutKind::Content]);

    let system = "<|im_start|>system\nYou are a System One decision engine. Evaluate if the condition holds for the given state.\nOption TRUE: The condition is valid and supported by state.\nOption FALSE: The condition is invalid or contradicted.<|im_end|>\n<|im_start|>user\n";
    let head = format!("{system}State:");
    assert_eq!(ids[..cuts[0].pos], t.encode_raw(&head, "t").unwrap()[..]);
    // The state of the training layouts: `State: s\nCondition:`, and the cut ends there.
    let upto = format!("{system}State: A man plays guitar.\nCondition:");
    assert_eq!(ids[..cuts[1].pos], t.encode_raw(&upto, "t").unwrap()[..]);
    assert!(cuts[1].pos < ids.len() - 1);
}

#[test]
fn the_state_layout_has_only_the_fixed_cut() {
    let qs = Questions::new([("q", safety())]).unwrap();
    let p = build(&state("How are you?"), &qs);
    assert_eq!(kinds(&p, 0), vec![CutKind::Fixed]);
    // Nothing else is read after the last cut: the answer is at the last token.
    let b = &p.branches()[0];
    assert!(b.cuts()[0].pos <= b.decide());
}

#[test]
fn the_cuts_of_a_row_do_not_change_with_the_other_rows_or_their_order() {
    let st = state("Two friends meet at the station.");
    let conditions: Vec<String> = (0..16).map(|i| format!("Condition number {i}.")).collect();
    let names: Vec<String> = (0..16).map(|i| format!("q{i}")).collect();
    let all = Questions::new(
        names
            .iter()
            .zip(&conditions)
            .map(|(n, c)| (n.as_str(), entailment(c))),
    )
    .unwrap();
    let reversed = Questions::new(
        names
            .iter()
            .zip(&conditions)
            .rev()
            .map(|(n, c)| (n.as_str(), entailment(c))),
    )
    .unwrap();
    let many = build(&st, &all);
    let rev = build(&st, &reversed);
    for (i, (n, c)) in names.iter().zip(&conditions).enumerate() {
        let alone = build(&st, &Questions::new([(n.as_str(), entailment(c))]).unwrap());
        assert_eq!(alone.row(0), many.row(i), "row {i}: ids");
        assert_eq!(alone.row_cuts(0), many.row_cuts(i), "row {i}: cuts");
        assert_eq!(alone.row_cuts(0), rev.row_cuts(15 - i), "row {i}: reversed");
    }
}

#[test]
fn rows_of_the_same_task_share_the_cut_positions_and_the_ids_before_them() {
    let qs = Questions::new([
        ("a", entailment("First condition.")),
        ("b", entailment("A completely different second condition.")),
    ])
    .unwrap();
    let p = build(&state("Same state."), &qs);
    assert_eq!(p.row_cuts(0), p.row_cuts(1));
    let last = p.row_cuts(0).unwrap().last().unwrap().pos;
    assert_eq!(p.row(0).unwrap()[..last], p.row(1).unwrap()[..last]);
}

#[test]
fn a_token_merged_across_the_boundary_drops_the_cut_and_leaves_the_ids_alone() {
    // ":" followed by a space becomes one token, so "State:" alone is not a prefix of the row.
    let merging = testkit::byte_level_with_merges(&SPECIALS, &[(":", "Ġ")]);
    let st = state("A man plays guitar.");
    let qs = Questions::new([("q", entailment("A person plays music."))]).unwrap();
    let p = build_with(&merging, &st, &qs);
    assert!(p.row_cuts(0).unwrap().is_empty(), "{:?}", p.row_cuts(0));
    // The ids are those of the whole string tokenized at once, as without cuts.
    let whole = "<|im_start|>system\nYou are a System One decision engine. Evaluate if the condition holds for the given state.\nOption TRUE: The condition is valid and supported by state.\nOption FALSE: The condition is invalid or contradicted.<|im_end|>\n<|im_start|>user\nState: A man plays guitar.\nCondition: A person plays music.\nDecision:<|im_end|>\n<|im_start|>assistant\nOption:";
    assert_eq!(
        p.branches()[0].ids(),
        merging.encode_raw(whole, "t").unwrap().as_slice()
    );
    // And the same row with the ordinary tokenizer does have its cuts.
    assert_eq!(kinds(&build(&st, &qs), 0).len(), 2);
}

#[test]
fn a_template_without_declared_tasks_has_no_cuts() {
    let t = tok();
    let q: Question = Choice::new(
        entry("Pick one."),
        vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
    )
    .unwrap()
    .into();
    let qs = Questions::new([("q", q)]).unwrap();
    let p = ChatMlLetters::new(None)
        .build(&state("hi"), &qs, &t, &limits())
        .unwrap();
    assert!(p.row_cuts(0).unwrap().is_empty());
}

#[test]
fn the_layout_texts_are_the_start_of_what_render_writes() {
    for layout in [Layout::State, Layout::StateCondition, Layout::StatePair] {
        let text = layout.render("S", "I");
        let head = format!("{} S", layout.head_text());
        assert!(text.starts_with(&head), "{layout:?}: {text:?}");
        if let Some(end) = layout.state_end_text() {
            let upto = format!("{head}{end}");
            assert!(text.starts_with(&upto), "{layout:?}: {text:?}");
            assert!(text[upto.len()..].starts_with(" I"), "{layout:?}: {text:?}");
        }
    }
}

#[test]
fn kev_has_no_cuts_in_the_branch_and_the_row_cut_is_the_end_of_the_state() {
    let t = testkit::kev();
    let q: Question = Choice::new(
        entry("Which one?"),
        vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
    )
    .unwrap()
    .into();
    let qs = Questions::new([("q", q)]).unwrap();
    let p = KEV_V1
        .build(&state("some state"), &qs, &t, &limits())
        .unwrap();
    assert!(p.branches()[0].cuts().is_empty());
    assert_eq!(
        p.row_cuts(0),
        Some(vec![Cut {
            pos: p.state().len(),
            kind: CutKind::Content,
        }])
    );
}
