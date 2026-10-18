//! What scoring the option keys costs on top of reading the prompt (spec 010, P1 and P2).
//!
//! For each request of a JSONL file (`{"id", "request": {state, questions}}`, the format of
//! `benches/quantization/measure_key_cost.py emit`) it builds the prompt with the real template and
//! runs the real engine on the same row, alternating:
//!
//! - **prefill**: the row is decoded and one logit is read (what any question costs at least);
//! - **keys**: the same row, the scores of all the option keys read (`KeyLogProbs`), once per
//!   `KEY_COST_WAVES` (default `1,4,8`): the number of keys that continue together in one decode.
//!   `1` is the reference path (one key at a time); the scores of the others must equal it.
//!
//! The ratio keys / prefill is what P1 (keys of one token, up to 8 options, threshold 1.15) and P2
//! (keys of 2-4 tokens, 8 options, threshold 1.5) bound. A second prefill run is the A/A control:
//! its ratio to the first is 1 on a quiet machine (D23). Two regimes: back to back and with a pause
//! of 0.5 s before each call. The cells rotate every round; the medians are of the per-round ratios.
//! Every run creates its context, as a request does, so the memory of a wave is in the time.
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_DEFAULT_TOKENIZER = "<base tokenizer.json>"
//! $env:ARCHAI_JEV_TEST_V2_GGUF = "<q8_0 gguf of the v2 weights>"
//! $env:ARCHAI_JEV_TEST_COST_REQUESTS = "<cost_requests.jsonl>"
//! cargo test --release --features testing --test key_cost -- --ignored --nocapture --test-threads=1
//! ```
//! `KEY_COST_ROUNDS` (default 7) and `KEY_COST_ONLY` (substrings of the ids to keep, separated by commas) tune the run.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_precision_loss
)]

use std::time::{Duration, Instant};

use _core::calibration::calibrated_softmax;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::engine::{Forward, Output, OutputSpec};
use _core::heads::KeyLogProbHead;
use _core::json_strict::{Json, parse};
use _core::models::testing::assets;
use _core::prompt::di::DecisionIndexChatMl;
use _core::prompt::{PromptTokenizer, RequestLimits, Roles, Template};
use _core::request::request_to_domain;

const MAX_KEY_TOKENS: usize = 8;
const MAX_OPTIONS: usize = 64;
const MAX_CONTEXT: u32 = 512; // the v2 of the registry

fn roles() -> Roles {
    Roles::new([("im_start", 151_644u32), ("im_end", 151_645)])
}

struct Cell {
    id: String,
    ids: Vec<u32>,
    keys: Vec<Vec<u32>>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn cells(tokenizer: &std::path::Path, requests: &std::path::Path) -> Vec<Cell> {
    let tok = PromptTokenizer::from_file(tokenizer, roles()).unwrap();
    let template = DecisionIndexChatMl::new(MAX_KEY_TOKENS);
    let limits = RequestLimits {
        model: "v2",
        max_context: 100_000,
        question_types: &["choice", "noul"],
        support: &KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS),
    };
    let only = std::env::var("KEY_COST_ONLY").ok();
    std::fs::read_to_string(requests)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let row = parse(line.as_bytes()).unwrap();
            let Some(Json::Str(id)) = row.get("id") else {
                panic!("id")
            };
            if only
                .as_deref()
                .is_some_and(|o| !o.split(',').any(|part| id.contains(part)))
            {
                return None;
            }
            let (state, qs) = request_to_domain(row.get("request").unwrap()).unwrap();
            let prompt = template
                .build(&state, &qs, &tok, &limits)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            let branch = &prompt.branches()[0];
            Some(Cell {
                id: id.clone(),
                ids: branch.ids().to_vec(),
                keys: branch.keys().to_vec(),
            })
        })
        .collect()
}

/// Seconds for one run of `cell` (context creation included) and the scores it read.
fn run(forward: &LlamaForward, cell: &Cell, with_keys: bool, pause: Duration) -> (f64, Vec<f64>) {
    if !pause.is_zero() {
        std::thread::sleep(pause);
    }
    let last = cell.ids.len() - 1;
    let spec = if with_keys {
        OutputSpec::KeyLogProbs {
            position: last,
            keys: cell.keys.clone(),
        }
    } else {
        OutputSpec::Logits {
            position: last,
            ids: vec![cell.keys[0][0]],
        }
    };
    let started = Instant::now();
    let mut session = forward.begin_empty().unwrap();
    let out = session.row(&cell.ids, &[spec]).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let scores = match out.into_iter().next() {
        Some(Output::KeyLogProbs(scores)) => scores,
        _ => Vec::new(),
    };
    (seconds, scores)
}

fn load(gguf: &std::path::Path, threads: i32, wave: u32) -> LlamaForward {
    LlamaForward::load(
        gguf,
        LlamaConfig {
            threads,
            max_context: MAX_CONTEXT,
            hidden_states: false,
        },
    )
    .unwrap()
    .with_key_scoring(MAX_KEY_TOKENS as u32)
    .with_key_wave(wave)
}

/// Keys that continue together must give the probabilities they give alone, for every wave size.
/// A one-token decode takes another kernel path in llama.cpp than a decode of several tokens, so
/// the raw scores of very unlikely keys differ in the second decimal (against the kit's, the
/// waves are the closer: 0.02 against 0.12 on the worst key), which is why the bound is on the
/// probabilities. Prints the worst key of each cell.
#[test]
#[ignore = "needs the v2 GGUF, the tokenizer and a requests file: see the module docs"]
fn a_wave_of_keys_scores_like_one_key_at_a_time() {
    let (Some(tokenizer), Ok(gguf), Ok(requests)) = (
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_V2_GGUF"),
        std::env::var("ARCHAI_JEV_TEST_COST_REQUESTS"),
    ) else {
        eprintln!(
            "SKIPPED: set the tokenizer, ARCHAI_JEV_TEST_V2_GGUF and ARCHAI_JEV_TEST_COST_REQUESTS"
        );
        return;
    };
    let tolerance: f64 = std::env::var("KEY_WAVE_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5e-3);
    let cells = cells(&tokenizer, std::path::Path::new(&requests));
    let gguf = std::path::Path::new(&gguf);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let reference = load(gguf, threads, 1);
    let waves = [2u32, 3, 4, 8];
    let forwards: Vec<LlamaForward> = waves.iter().map(|&w| load(gguf, threads, w)).collect();
    let (mut worst_score, mut worst_p) = (0.0f64, 0.0f64);
    for cell in &cells {
        let (_, base) = run(&reference, cell, true, Duration::ZERO);
        let base_p = calibrated_softmax(&base, 1.0).unwrap();
        for (w, f) in waves.iter().zip(&forwards) {
            let (_, scores) = run(f, cell, true, Duration::ZERO);
            assert_eq!(scores.len(), base.len(), "{}", cell.id);
            let (key, d) = scores
                .iter()
                .zip(&base)
                .enumerate()
                .map(|(k, (a, b))| (k, (a - b).abs()))
                .fold((0, 0.0), |m, x| if x.1 > m.1 { x } else { m });
            let dp = calibrated_softmax(&scores, 1.0)
                .unwrap()
                .iter()
                .zip(&base_p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if d > 1e-9 {
                println!(
                    "{} wave {w}: key {key} ({} tokens) score differs by {d:.3e} ({:.4} against {:.4}); largest probability gap {dp:.2e}",
                    cell.id,
                    cell.keys[key].len(),
                    scores[key],
                    base[key]
                );
            }
            worst_score = worst_score.max(d);
            worst_p = worst_p.max(dp);
        }
    }
    println!(
        "worst difference over every wave size and cell: score {worst_score:.3e}, probability {worst_p:.3e}"
    );
    assert!(
        worst_p <= tolerance,
        "a wave changed a probability by {worst_p:.3e}"
    );
}

#[test]
#[ignore = "needs the v2 GGUF, the tokenizer and a requests file: see the module docs"]
fn the_keys_cost_against_the_prefill() {
    let (Some(tokenizer), Ok(gguf), Ok(requests)) = (
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_V2_GGUF"),
        std::env::var("ARCHAI_JEV_TEST_COST_REQUESTS"),
    ) else {
        eprintln!(
            "SKIPPED: set the tokenizer, ARCHAI_JEV_TEST_V2_GGUF and ARCHAI_JEV_TEST_COST_REQUESTS"
        );
        return;
    };
    let cells = cells(&tokenizer, std::path::Path::new(&requests));
    assert!(!cells.is_empty(), "no request kept");
    let threads = std::env::var("KEY_COST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() as i32));
    let waves: Vec<u32> = std::env::var("KEY_COST_WAVES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 4, 8]);
    let gguf = std::path::Path::new(&gguf);
    let forwards: Vec<LlamaForward> = waves.iter().map(|&w| load(gguf, threads, w)).collect();
    let rounds: usize = std::env::var("KEY_COST_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);

    for cell in cells.iter().take(3) {
        for f in &forwards {
            run(f, cell, true, Duration::ZERO); // warm up the kernels and the clock
        }
    }
    println!(
        "threads {threads}, waves {waves:?}, {rounds} rounds; ratios are medians of the per-round ratios"
    );
    for (regime, pause) in [
        ("back to back", Duration::ZERO),
        ("pause 0.5 s", Duration::from_millis(500)),
    ] {
        println!("\n== {regime}");
        print!(
            "{:<18} {:>6} {:>6} {:>8}",
            "cell", "prompt", "keytok", "prefill"
        );
        for w in &waves {
            print!(" {:>9} {:>7}", format!("w{w} ms"), "x pre");
        }
        println!(" {:>6} {:>10}", "A/A", "max dscore");
        let n = cells.len();
        let m = waves.len();
        let mut pre = vec![Vec::new(); n];
        let mut aa = vec![Vec::new(); n];
        let mut keyed = vec![vec![Vec::new(); m]; n];
        let mut ratios = vec![vec![Vec::new(); m]; n];
        let mut worst = vec![0.0f64; n];
        for round in 0..rounds {
            let mut order: Vec<usize> = (0..n).map(|i| (i + round) % n).collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for i in order {
                let cell = &cells[i];
                // prefill, then every wave (in the order of the round), then prefill again
                let (first, _) = run(&forwards[0], cell, false, pause);
                let mut timed = vec![0.0; m];
                let mut scores: Vec<Vec<f64>> = vec![Vec::new(); m];
                let mut idx: Vec<usize> = (0..m).collect();
                if round % 2 == 1 {
                    idx.reverse();
                }
                for j in idx {
                    let (t, sc) = run(&forwards[j], cell, true, pause);
                    timed[j] = t;
                    scores[j] = sc;
                }
                let (second, _) = run(&forwards[0], cell, false, pause);
                let base = (first + second) / 2.0;
                pre[i].push(base * 1000.0);
                aa[i].push(second / first);
                for j in 0..m {
                    keyed[i][j].push(timed[j] * 1000.0);
                    ratios[i][j].push(timed[j] / base);
                    for (a, b) in scores[j].iter().zip(&scores[0]) {
                        worst[i] = worst[i].max((a - b).abs());
                    }
                }
            }
        }
        for (i, cell) in cells.iter().enumerate() {
            print!(
                "{:<18} {:>6} {:>6} {:>8.0}",
                cell.id,
                cell.ids.len(),
                cell.keys.iter().map(Vec::len).sum::<usize>(),
                median(pre[i].clone())
            );
            for j in 0..m {
                print!(
                    " {:>9.0} {:>7.2}",
                    median(keyed[i][j].clone()),
                    median(ratios[i][j].clone())
                );
            }
            println!(" {:>6.2} {:>10.2e}", median(aa[i].clone()), worst[i]);
        }
    }
}

/// Where the time of a request goes, and what changes it: for every GGUF (`CPU_SPEED_GGUFS`,
/// `label=path;label=path`) and number of threads (`CPU_SPEED_THREADS`, default `8,12,16,20`) the time to
/// create a context and the prefill of a short (~140 tokens) and a long (~460 tokens) prompt. Spec 010,
/// speed against the kit (S12). Cells rotate every round, medians of `CPU_SPEED_ROUNDS` (default 5).
#[test]
#[ignore = "needs GGUF files: CPU_SPEED_GGUFS, the tokenizer and a requests file"]
fn prefill_speed_by_threads_and_dtype() {
    let (Some(tokenizer), Ok(requests), Ok(ggufs)) = (
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_COST_REQUESTS"),
        std::env::var("CPU_SPEED_GGUFS"),
    ) else {
        eprintln!("SKIPPED: set the tokenizer, ARCHAI_JEV_TEST_COST_REQUESTS and CPU_SPEED_GGUFS");
        return;
    };
    let all = cells(&tokenizer, std::path::Path::new(&requests));
    let short = all.iter().find(|c| c.id.contains("letters-08")).unwrap();
    let long = all.iter().find(|c| c.id.contains("padded-48")).unwrap();
    let threads: Vec<i32> = std::env::var("CPU_SPEED_THREADS")
        .unwrap_or_else(|_| "8,12,16,20".to_string())
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let rounds: usize = std::env::var("CPU_SPEED_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let models: Vec<(String, std::path::PathBuf)> = ggufs
        .split(';')
        .filter_map(|e| e.split_once('='))
        .map(|(l, p)| (l.trim().to_string(), std::path::PathBuf::from(p.trim())))
        .collect();
    let mut combos: Vec<(String, i32, LlamaForward)> = Vec::new();
    for (label, path) in &models {
        for &t in &threads {
            let f = LlamaForward::load(
                path,
                LlamaConfig {
                    threads: t,
                    max_context: MAX_CONTEXT,
                    hidden_states: false,
                },
            )
            .unwrap();
            combos.push((label.clone(), t, f));
        }
    }
    let prefill = |f: &LlamaForward, cell: &Cell| -> f64 {
        let spec = OutputSpec::Logits {
            position: cell.ids.len() - 1,
            ids: vec![cell.keys[0][0]],
        };
        let started = Instant::now();
        let mut s = f.begin_empty().unwrap();
        std::hint::black_box(s.row(&cell.ids, &[spec]).unwrap());
        started.elapsed().as_secs_f64() * 1000.0
    };
    let create = |f: &LlamaForward| -> f64 {
        let started = Instant::now();
        let s = f.begin_empty().unwrap();
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        drop(s);
        ms
    };
    for (_, _, f) in combos.iter().take(2) {
        prefill(f, short); // warm up
    }
    let n = combos.len();
    let mut got = vec![(Vec::new(), Vec::new(), Vec::new()); n];
    for round in 0..rounds {
        let mut order: Vec<usize> = (0..n).map(|i| (i + round * 3) % n).collect();
        if round % 2 == 1 {
            order.reverse();
        }
        for i in order {
            let f = &combos[i].2;
            got[i].0.push(create(f));
            got[i].1.push(prefill(f, short));
            got[i].2.push(prefill(f, long));
        }
    }
    println!(
        "short prompt {} tokens, long prompt {} tokens; medians of {rounds} rounds, ms",
        short.ids.len(),
        long.ids.len()
    );
    println!(
        "{:<10} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "weights", "threads", "context", "short", "ms/token", "long", "ms/token"
    );
    for (i, (label, t, _)) in combos.iter().enumerate() {
        let (c, s, l) = (
            median(got[i].0.clone()),
            median(got[i].1.clone()),
            median(got[i].2.clone()),
        );
        println!(
            "{:<10} {:>7} {:>10.1} {:>10.0} {:>10.2} {:>10.0} {:>10.2}",
            label,
            t,
            c,
            s,
            (s - c) / short.ids.len() as f64,
            l,
            (l - c) / long.ids.len() as f64
        );
    }
}
