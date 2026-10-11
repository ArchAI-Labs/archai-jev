//! Reusing what questions share must not change the answers (spec 007).
//!
//! * The token counts of the demo requests (level T1: they need the default `tokenizer.json`,
//!   `ARCHAI_JEV_TEST_DEFAULT_TOKENIZER`; `ARCHAI_JEV_REQUIRE_ASSETS=1` makes a missing one fail).
//! * The 48 golden prompts of the default model with the reuse off and on, against the table of
//!   tolerances of the spec and against the PyTorch oracle (level T2: needs the 1.65 GB GGUF,
//!   `ARCHAI_JEV_TEST_DEFAULT_GGUF`; `cargo test --release --features testing --test
//!   prefix_cache_parity -- --ignored --nocapture`).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use _core::Calibration;
use _core::calibration::calibrated_softmax;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::heads::LettersHead;
use _core::model_scorer::ModelScorer;
use _core::models::manifest::Target;
use _core::models::testing::assets;
use _core::models::testing::golden::{self, DefaultCase};
use _core::prefix::RequestPlan;
use _core::prompt::chatml::ChatMlLetters;
use _core::prompt::tasks::{self, default_tasks};
use _core::prompt::{NoLimits, PromptTokenizer, RequestLimits, Roles, Template};
use _core::schema::{Questions, State, StateValue, TextEntry, YesNo};
use _core::scorer::Scorer;

const TYPES: [&str; 2] = ["choice", "noul"];

/// The source of the "ten hypotheses" demo (written by the authors of archai-jev).
const SOURCE: &str = "Orion Labs released version 2.4 of its mapping app on 12 March. The update adds offline maps for 40 countries and reduces battery usage by improving location sampling. The app is free on Android and iOS, but offline maps require a Pro subscription costing 4 euros per month. The company is based in Lisbon and has about 60 employees.";

const HYPOTHESES: [&str; 10] = [
    "Version 2.4 came out on 12 March.",
    "The new offline maps cover 40 countries.",
    "Battery usage went down thanks to better location sampling.",
    "The Pro plan costs 4 euros a month.",
    "Orion Labs is headquartered in Lisbon.",
    "The offline maps are free for all users.",
    "Orion Labs has a team of around 600 people.",
    "The app is available only on iOS.",
    "The company was founded in Lisbon in 2012.",
    "The update also adds a built-in route planner for cyclists.",
];

/// The tickets of the "similar tickets" demo.
const TICKETS: [&str; 12] = [
    "Cannot log in with my Google account",
    "Google sign-in fails with an error",
    "Login via Google does not work",
    "Export to CSV produces an empty file",
    "The CSV export is empty",
    "Exported CSV file has no rows",
    "Please add a dark mode",
    "Dark theme support requested",
    "Need a night mode option",
    "Login with email and password fails",
    "The CSV import rejects my file",
    "Invoice shows the wrong VAT number",
];

fn roles() -> Roles {
    Roles::new([("im_start", 151_644u32), ("im_end", 151_645)])
}

fn template() -> ChatMlLetters {
    ChatMlLetters::new(Some(tasks::parse(&default_tasks(), 4).unwrap()))
}

fn target(label: &str, id: u32) -> Target {
    Target {
        label: label.to_string(),
        id,
    }
}

fn head() -> LettersHead {
    LettersHead::new(
        &[
            target("A", 32),
            target("B", 33),
            target("C", 34),
            target("D", 35),
        ],
        Some(&(target("FALSE", 30_351), target("TRUE", 20_611))),
    )
}

fn yes_no(condition: &str, yes: &str, no: &str) -> _core::schema::Question {
    YesNo::new(
        TextEntry::text(condition).unwrap(),
        Some(TextEntry::text(yes).unwrap()),
        Some(TextEntry::text(no).unwrap()),
    )
    .unwrap()
    .into()
}

fn entailment(condition: &str) -> _core::schema::Question {
    yes_no(
        condition,
        "The condition is valid and supported by state.",
        "The condition is invalid or contradicted.",
    )
}

fn similarity(other: &str) -> _core::schema::Question {
    yes_no(
        other,
        "The two texts have equivalent meaning or intent.",
        "The texts have different meanings.",
    )
}

fn state(s: &str) -> State {
    State::new(StateValue::string(s)).unwrap()
}

fn prompt(tok: &PromptTokenizer, st: &State, qs: &Questions) -> _core::prompt::Prompt {
    let limits = RequestLimits {
        model: "default",
        max_context: 512,
        question_types: &TYPES,
        support: &NoLimits,
    };
    template().build(st, qs, tok, &limits).unwrap()
}

/// Tokens the model decodes for the "ten hypotheses about one source" request, and for the 66
/// questions of the "similar tickets" demo (one request per ticket, a question per later ticket).
#[test]
fn the_demo_requests_decode_the_expected_tokens() {
    let Some(path) = assets::default_tokenizer() else {
        return;
    };
    let tok = PromptTokenizer::from_file(&path, roles()).unwrap();

    let qs = Questions::new(
        HYPOTHESES
            .iter()
            .enumerate()
            .map(|(i, h)| (format!("h{i}"), entailment(h)))
            .collect::<Vec<_>>()
            .iter()
            .map(|(n, q)| (n.as_str(), q.clone())),
    )
    .unwrap();
    let p = prompt(&tok, &state(SOURCE), &qs);
    let whole: usize = (0..HYPOTHESES.len()).map(|i| p.row_len(i).unwrap()).sum();
    let plan = RequestPlan::new(&p).unwrap();
    eprintln!(
        "demo 002 (10 hypotheses): {whole} tokens as 10 rows, {} decoded with reuse",
        plan.decoded_tokens()
    );
    assert_eq!(plan.decoded_tokens(), 322);

    let mut total = 0usize;
    let mut questions = 0usize;
    let mut whole_total = 0usize;
    for (i, ticket) in TICKETS.iter().enumerate().take(TICKETS.len() - 1) {
        let items: Vec<(String, _core::schema::Question)> = TICKETS
            .iter()
            .enumerate()
            .skip(i + 1)
            .map(|(j, t)| (format!("t{j}"), similarity(t)))
            .collect();
        let qs = Questions::new(items.iter().map(|(n, q)| (n.as_str(), q.clone()))).unwrap();
        let p = prompt(&tok, &state(ticket), &qs);
        whole_total += (0..items.len())
            .map(|r| p.row_len(r).unwrap())
            .sum::<usize>();
        total += RequestPlan::new(&p).unwrap().decoded_tokens();
        questions += items.len();
    }
    eprintln!(
        "demo 003 ({questions} questions): {whole_total} tokens as rows, {total} decoded with reuse"
    );
    assert_eq!(questions, 66);
    assert_eq!(total, 1_615);
}

fn restricted_probs(logits: &[f64], t: f64) -> Vec<f64> {
    calibrated_softmax(logits, t).unwrap()
}

fn top(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold(0, |b, (i, v)| if *v > p[b] { i } else { b })
}

/// The two paths of the same engine: one decode per row (the reuse off, as 0.1.0) and the row
/// split at the cuts of its template (the reuse on).
/// The cuts the template emits on the real tokenizer: for the 48 golden prompts and the requests
/// of the two demos. Each task has its own fixed head (the same tokens in every row of the task,
/// and the same position); the layouts that put the state first also cut right after it.
#[test]
fn cuts_are_valid_on_the_golden_and_the_sequences() {
    let Some(path) = assets::default_tokenizer() else {
        return;
    };
    let tok = PromptTokenizer::from_file(&path, roles()).unwrap();
    use _core::prompt::CutKind;
    use std::collections::BTreeMap;

    // task -> (fixed position, ids before it)
    let mut heads: BTreeMap<String, (usize, Vec<u32>)> = BTreeMap::new();
    let mut dropped = Vec::new();
    for case in golden::load_default() {
        let (st, qs) = golden::request_to_domain(&case.request).unwrap();
        let p = prompt(&tok, &st, &qs);
        let cuts = p.row_cuts(0).unwrap();
        let row = p.row(0).unwrap();
        let fixed = cuts.iter().find(|c| c.kind == CutKind::Fixed);
        let content = cuts.iter().find(|c| c.kind == CutKind::Content);
        let Some(fixed) = fixed else {
            dropped.push(case.id.clone());
            continue;
        };
        assert!(fixed.pos >= 16, "{}: the fixed head is too short", case.id);
        let entry = heads
            .entry(case.task.clone())
            .or_insert_with(|| (fixed.pos, row[..fixed.pos].to_vec()));
        assert_eq!(entry.0, fixed.pos, "{}: fixed cut moved", case.id);
        assert_eq!(entry.1, row[..fixed.pos], "{}: fixed ids differ", case.id);
        let has_state_end = matches!(case.task.as_str(), "entailment" | "similarity");
        assert_eq!(content.is_some(), has_state_end, "{}: content cut", case.id);
        if let Some(c) = content {
            // After the state, before the condition, and leaving the question to the row.
            assert!(c.pos > fixed.pos && c.pos < row.len() - 1, "{}", case.id);
        }
        // The head reads at the last token, after every cut.
        let last = cuts.last().unwrap().pos;
        assert!(p.branches()[0].decide() >= last, "{}", case.id);
    }
    eprintln!(
        "fixed heads (tokens): {:?}; rows without a fixed cut: {dropped:?}",
        heads
            .iter()
            .map(|(t, (n, _))| (t.as_str(), *n))
            .collect::<Vec<_>>()
    );
    assert!(
        dropped.is_empty(),
        "cuts dropped on the golden: {dropped:?}"
    );
    assert_eq!(heads.len(), 4);

    // The two demos: every row of a request has the same cuts, so all of them are common.
    let hyp = Questions::new(
        HYPOTHESES
            .iter()
            .enumerate()
            .map(|(i, h)| (format!("h{i}"), entailment(h)))
            .collect::<Vec<_>>()
            .iter()
            .map(|(n, q)| (n.as_str(), q.clone())),
    )
    .unwrap();
    let p = prompt(&tok, &state(SOURCE), &hyp);
    let plan = RequestPlan::new(&p).unwrap();
    assert_eq!(
        plan.common.len(),
        2,
        "fixed + state end are common to the ten rows"
    );
}

/// Short user messages (written by the authors; the same as in the benchmark).
const MESSAGES: [&str; 25] = [
    "How do I bake a chocolate cake?",
    "What is the capital of Australia?",
    "Can you recommend a good book about history?",
    "Translate 'good morning' into Spanish.",
    "How many ounces are in a pound?",
    "Write a short poem about the sea.",
    "Why is the sky blue during the day?",
    "What time does the library usually open?",
    "Explain what a mortgage is.",
    "I lost my card and need to block it.",
    "How can I change the PIN of my debit card?",
    "Where can I see my last transfers?",
    "What are the fees for an international transfer?",
    "My payment was declined, what should I do?",
    "How do I update my address?",
    "Is there a limit on cash withdrawals?",
    "I want to close my account.",
    "Why was I charged a monthly fee?",
    "How do I activate my new card?",
    "Can I set up a standing order?",
    "The ATM kept my card.",
    "How do I get a statement for last year?",
    "What is the exchange rate for dollars today?",
    "I do not recognise a transaction on my account.",
    "Can I increase my transfer limit?",
];

fn safety() -> _core::schema::Question {
    use _core::schema::{Choice, ChoiceOption};
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

fn intent() -> _core::schema::Question {
    use _core::schema::{Choice, ChoiceOption};
    Choice::new(
        TextEntry::text("You are a System One decision engine for intent routing.").unwrap(),
        [
            "Card & Account Issues",
            "Payments & Transfers",
            "Fees & Charges",
            "General Support",
        ]
        .into_iter()
        .map(ChoiceOption::key)
        .collect(),
    )
    .unwrap()
    .into()
}

type Request = (State, Questions);

fn one(name: &str, st: &str, q: _core::schema::Question) -> Request {
    (state(st), Questions::new([(name, q)]).unwrap())
}

/// The five request sequences of the benchmark, in order: (name, requests, tokens decoded
/// expected with the memory on, `None` where the spec leaves it to the admission rule).
fn sequences() -> Vec<(&'static str, Vec<Request>, Option<u64>)> {
    // Demo 003: a request per ticket, a question per later ticket.
    let demo003 = (0..TICKETS.len() - 1)
        .map(|i| {
            let items: Vec<(String, _core::schema::Question)> = (i + 1..TICKETS.len())
                .map(|j| (format!("t{j}"), similarity(TICKETS[j])))
                .collect();
            let qs = Questions::new(items.iter().map(|(n, q)| (n.as_str(), q.clone()))).unwrap();
            (state(TICKETS[i]), qs)
        })
        .collect();
    // Demo 002: one request, ten hypotheses about one source.
    let items: Vec<(String, _core::schema::Question)> = HYPOTHESES
        .iter()
        .enumerate()
        .map(|(i, h)| (format!("h{i}"), entailment(h)))
        .collect();
    let demo002 = vec![(
        state(SOURCE),
        Questions::new(items.iter().map(|(n, q)| (n.as_str(), q.clone()))).unwrap(),
    )];
    // The same source checked one hypothesis per ask.
    let stream = HYPOTHESES
        .iter()
        .map(|h| one("h", SOURCE, entailment(h)))
        .collect();
    // Twenty independent short messages, one safety question each.
    let safety_stream = MESSAGES[..20]
        .iter()
        .map(|m| one("q", m, safety()))
        .collect();
    // Mixed traffic: the four tasks in turn, one question per request.
    let mixed = (0..24)
        .map(|i| {
            let m = MESSAGES[i];
            let other = MESSAGES[(i + 9) % 25];
            match i % 4 {
                0 => one("q", m, safety()),
                1 => one("q", m, intent()),
                2 => one("q", m, entailment(other)),
                _ => one("q", m, similarity(other)),
            }
        })
        .collect();
    vec![
        ("demo003_66q_11asks", demo003, Some(1_185)),
        ("demo002_10hyp_1ask", demo002, Some(322)),
        ("safety_20asks", safety_stream, Some(388)),
        ("stream_10asks_1q", stream, None),
        ("mixed_24asks", mixed, None),
    ]
}

struct Engines {
    off: ModelScorer,
    on: ModelScorer,
    forward: Arc<LlamaForward>,
    tokenizer: std::path::PathBuf,
}

impl Engines {
    fn scorer(&self, limit: u64) -> ModelScorer {
        ModelScorer::new(
            "default",
            Box::new(template()),
            PromptTokenizer::from_file(&self.tokenizer, roles()).unwrap(),
            self.forward.clone(),
            Box::new(head()),
            Calibration::new(2.09, true).unwrap(),
            512,
            TYPES.iter().map(|t| (*t).to_string()).collect(),
        )
        .with_prefix_cache(limit)
    }
}

fn engines() -> Option<Engines> {
    let (tokenizer, gguf) = (assets::default_tokenizer()?, assets::default_gguf()?);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let forward = Arc::new(
        LlamaForward::load(
            &gguf,
            LlamaConfig {
                threads,
                max_context: 512,
                hidden_states: false,
            },
        )
        .unwrap(),
    );
    let make = |limit: u64| {
        ModelScorer::new(
            "default",
            Box::new(template()),
            PromptTokenizer::from_file(&tokenizer, roles()).unwrap(),
            forward.clone(),
            Box::new(head()),
            Calibration::new(2.09, true).unwrap(),
            512,
            TYPES.iter().map(|t| (*t).to_string()).collect(),
        )
        .with_prefix_cache(limit)
    };
    Some(Engines {
        off: make(0),
        on: make(128 << 20),
        forward,
        tokenizer,
    })
}

/// The table of tolerances between the reuse off and on: raw logits, probabilities at the
/// declared temperature, and the argmax unless the top two are closer than 0.06.
fn within_table(id: &str, a: &[f64], b: &[f64]) -> (f64, f64) {
    let dz = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    let (pa, pb) = (restricted_probs(a, 2.09), restricted_probs(b, 2.09));
    let dp = pa
        .iter()
        .zip(&pb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(dz <= 0.75, "{id}: max |dz| = {dz} exceeds 0.75");
    assert!(dp <= 0.06, "{id}: max |dp| = {dp} exceeds 0.06");
    if top(&pa) != top(&pb) {
        let mut sorted = pa.clone();
        sorted.sort_by(|x, y| y.total_cmp(x));
        let gap = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
        assert!(gap < 0.06, "{id}: argmax differs with a clear gap {gap}");
    }
    (dz, dp)
}

/// The 48 golden prompts on three paths: reuse off, memory cold (cleared before each prompt),
/// memory warm (the same request again), plus the whole list in a row with the memory filling up.
#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF and the default tokenizer"]
fn golden_three_paths_within_table_and_oracle() {
    let Some(e) = engines() else {
        return;
    };
    let (mut worst_z, mut worst_p, mut oracle) = (0f64, 0f64, [0f64; 3]);
    let cases: Vec<DefaultCase> = golden::load_default();
    assert_eq!(cases.len(), 48);
    let mut cold_all = Vec::new();
    for case in &cases {
        let (st, qs) = golden::request_to_domain(&case.request).unwrap();
        let off = e.off.run(&st, &qs).unwrap();
        e.on.clear_cache();
        let cold = e.on.run(&st, &qs).unwrap();
        let warm = e.on.run(&st, &qs).unwrap();
        assert_eq!(off.input_ids, case.ids, "{}: ids", case.id);
        assert_eq!(
            cold, warm,
            "{}: cold and warm must be identical bit for bit",
            case.id
        );
        let (dz, dp) = within_table(&case.id, &off.questions[0].1, &cold.questions[0].1);
        worst_z = worst_z.max(dz);
        worst_p = worst_p.max(dp);
        let theirs = restricted_probs(&case.oracle_logits, 1.0);
        for (k, run) in [&off, &cold, &warm].into_iter().enumerate() {
            let mine = restricted_probs(&run.questions[0].1, 1.0);
            let d = mine
                .iter()
                .zip(&theirs)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            oracle[k] = oracle[k].max(d);
            assert!(
                d <= 0.2,
                "{}: |dp| against the oracle on path {k} = {d}",
                case.id
            );
        }
        cold_all.push(cold);
    }
    // The same prompts in a row, the memory filling up as it goes: the same numbers again.
    e.on.clear_cache();
    for (case, cold) in cases.iter().zip(&cold_all) {
        let (st, qs) = golden::request_to_domain(&case.request).unwrap();
        assert_eq!(&e.on.run(&st, &qs).unwrap(), cold, "{}: in a row", case.id);
    }
    let info = e.on.cache_info().unwrap();
    eprintln!(
        "off vs cold: max |dz| = {worst_z:.4}, max |dp| = {worst_p:.4}; against the oracle: off {:.4}, cold {:.4}, warm {:.4}; in a row: {} hits, {} misses, {} entries",
        oracle[0], oracle[1], oracle[2], info.hits, info.misses, info.entries
    );
    assert!(info.hits > 0, "the fixed heads were restored");
}

/// The five sequences: answers against the reuse off within the table, and the tokens decoded.
#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF and the default tokenizer"]
fn sequences_match_and_decode_the_expected_tokens() {
    let Some(e) = engines() else {
        return;
    };
    for (name, requests, expected) in sequences() {
        e.on.clear_cache();
        e.on.reset_cache_counters();
        let (mut worst_z, mut worst_p) = (0f64, 0f64);
        for (i, (st, qs)) in requests.iter().enumerate() {
            let off = e.off.run(st, qs).unwrap();
            let on = e.on.run(st, qs).unwrap();
            for ((qn, a), (_, b)) in off.questions.iter().zip(&on.questions) {
                let (dz, dp) = within_table(&format!("{name}#{i}/{qn}"), a, b);
                worst_z = worst_z.max(dz);
                worst_p = worst_p.max(dp);
            }
        }
        let info = e.on.cache_info().unwrap();
        eprintln!(
            "{name}: decoded {} reused {} hits {} misses {} entries {} bytes {}; max |dz| {worst_z:.3} max |dp| {worst_p:.3}",
            info.tokens_decoded,
            info.tokens_reused,
            info.hits,
            info.misses,
            info.entries,
            info.bytes
        );
        if let Some(want) = expected {
            assert_eq!(info.tokens_decoded, want, "{name}: tokens decoded");
        }
    }
}

/// Several threads asking the real engine at once, with a memory small enough to evict: every
/// answer is the sequential one, bit for bit.
#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF and the default tokenizer"]
fn concurrent_real_engine() {
    let Some(e) = engines() else {
        return;
    };
    let on = e.scorer(8 << 20);
    let all: Vec<Request> = sequences()
        .into_iter()
        .flat_map(|(_, r, _)| r.into_iter().take(4))
        .collect();
    let want: Vec<_> = all
        .iter()
        .map(|(st, qs)| e.off.run(st, qs).unwrap())
        .collect();
    // Same numbers as the memory-off path need the same cuts: compare with a cold, one-by-one run.
    let reference: Vec<_> = all
        .iter()
        .map(|(st, qs)| {
            on.clear_cache();
            on.run(st, qs).unwrap()
        })
        .collect();
    for (a, b) in want.iter().zip(&reference) {
        for ((_, x), (_, y)) in a.questions.iter().zip(&b.questions) {
            within_table("concurrent", x, y);
        }
    }
    on.clear_cache();
    on.reset_cache_counters();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let (on, all, reference) = (&on, &all, &reference);
                scope.spawn(move || {
                    for i in 0..12 {
                        let k = (t * 5 + i * 3) % all.len();
                        let (st, qs) = &all[k];
                        assert_eq!(
                            &on.run(st, qs).unwrap(),
                            &reference[k],
                            "thread {t}, call {i}"
                        );
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });
    let info = on.cache_info().unwrap();
    assert!(info.bytes <= info.limit_bytes);
    assert_eq!(info.hits + info.misses, 48);
}
