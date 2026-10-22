//! A fake engine: deterministic numbers that depend on the request, with knobs to break them.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::calibration::calibrated_softmax;
use crate::engine::device::BackendKind;
use crate::error::{Error, Result};
use crate::json_strict::Json;
use crate::mock::MockScorer;
use crate::models::backend::{Backend, Engine, ValidatedCheckpoint};
use crate::models::families;
use crate::models::verify::{RunOutput, VectorRunner};
use crate::schema::{Questions, State};
use crate::scorer::{Calibration, Scorer};

/// FNV-1a over bytes, then SplitMix64: stable everywhere.
fn seed_of(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Number of options of a question in TypeSafe JSON shape.
pub fn option_count(question: &Json) -> usize {
    match question.get("type") {
        Some(Json::Str(t)) if t == "noul" => 2,
        _ => match question.get("criteria") {
            Some(Json::Object(p)) => p.len(),
            Some(Json::Array(a)) => a.len(),
            _ => 1,
        },
    }
}

/// The prompt ids the fake engine "builds" for `request`.
pub fn fake_ids(request: &Json, n_vocab: u64) -> Vec<u32> {
    let mut s = seed_of(&request.to_canonical_string());
    (0..6)
        .map(|_| u32::try_from(splitmix(&mut s) % n_vocab.max(1)).unwrap_or(0))
        .collect()
}

/// The raw logits the fake engine "computes" for each question of `request`.
pub fn fake_logits(request: &Json) -> Vec<(String, Vec<f64>)> {
    let Some(Json::Object(questions)) = request.get("questions") else {
        return Vec::new();
    };
    questions
        .iter()
        .map(|(name, q)| {
            let mut s = seed_of(&format!("{}|{name}", request.to_canonical_string()));
            let logits = (0..option_count(q))
                .map(|_| (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0)
                .collect();
            (name.clone(), logits)
        })
        .collect()
}

/// Probabilities for the fake logits at `temperature`.
pub fn fake_probabilities(logits: &[f64], temperature: f64) -> Vec<f64> {
    calibrated_softmax(logits, temperature).unwrap()
}

/// How the fake engine misbehaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Perturb {
    /// Behaves exactly as the vectors expect.
    None,
    /// Builds a prompt with one token id changed.
    IdsOff,
    /// Adds this amount to every logit of the first option.
    LogitShift(f64),
    /// Returns NaN for the first logit.
    Nan,
    /// Returns +infinity for the first logit.
    Inf,
    /// Answers one question too few.
    DropQuestion,
}

/// Counts what the engine was asked to do.
#[derive(Debug, Default)]
pub struct Counters {
    pub loads: AtomicUsize,
    pub runs: AtomicUsize,
    pub scores: AtomicUsize,
    /// The identity of the device of every load, in order.
    pub devices: Mutex<Vec<String>>,
    /// While true, `score` fails like an engine that ran out of memory during a request.
    pub fail_scores: AtomicBool,
}

/// A loaded fake model: scorer and vector runner in one.
pub struct FakeModel {
    calibration: Calibration,
    n_vocab: u64,
    perturb: Perturb,
    /// The engine fails when asked to run a self-check vector (a context that does not fit).
    fail_run: bool,
    counters: Arc<Counters>,
    mock: MockScorer,
}

impl Scorer for FakeModel {
    fn calibration(&self) -> Calibration {
        self.calibration
    }
    fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>> {
        self.counters.scores.fetch_add(1, Ordering::SeqCst);
        if self.counters.fail_scores.load(Ordering::SeqCst) {
            return Err(Error::Inference {
                detail: "the fake engine ran out of memory during a request".to_string(),
            });
        }
        self.mock.score(state, questions)
    }
}

impl VectorRunner for FakeModel {
    fn run(&self, request: &Json) -> Result<RunOutput> {
        self.counters.runs.fetch_add(1, Ordering::SeqCst);
        if self.fail_run {
            return Err(Error::Inference {
                detail: "the fake engine ran out of memory creating the context".to_string(),
            });
        }
        let mut ids = fake_ids(request, self.n_vocab);
        let mut questions = fake_logits(request);
        match self.perturb {
            Perturb::None => {}
            Perturb::IdsOff => {
                if let Some(first) = ids.first_mut() {
                    *first =
                        first.wrapping_add(1) % u32::try_from(self.n_vocab).unwrap_or(1).max(1);
                }
            }
            Perturb::LogitShift(d) => {
                for (_, l) in &mut questions {
                    if let Some(x) = l.first_mut() {
                        *x += d;
                    }
                }
            }
            Perturb::Nan => {
                if let Some(x) = questions.first_mut().and_then(|(_, l)| l.first_mut()) {
                    *x = f64::NAN;
                }
            }
            Perturb::Inf => {
                if let Some(x) = questions.first_mut().and_then(|(_, l)| l.first_mut()) {
                    *x = f64::INFINITY;
                }
            }
            Perturb::DropQuestion => {
                questions.pop();
            }
        }
        Ok(RunOutput {
            input_ids: ids,
            questions,
        })
    }
}

/// A fake backend with counters and knobs, standing in for llama.cpp in tests.
#[derive(Clone)]
pub struct FakeBackend {
    pub counters: Arc<Counters>,
    pub perturb: Perturb,
    pub fail_load: bool,
    /// Numbers that differ on one backend only: a GPU that computes differently.
    pub perturb_on: Vec<(BackendKind, Perturb)>,
    /// Backends whose load fails like an engine that cannot use the device.
    pub fail_load_on: Vec<BackendKind>,
    /// Backends whose self-check run fails like an engine out of memory.
    pub fail_run_on: Vec<BackendKind>,
}

impl FakeBackend {
    pub fn new() -> Self {
        FakeBackend {
            counters: Arc::new(Counters::default()),
            perturb: Perturb::None,
            fail_load: false,
            perturb_on: Vec::new(),
            fail_load_on: Vec::new(),
            fail_run_on: Vec::new(),
        }
    }

    /// The numbers on `backend` are perturbed, those on every other backend are not.
    pub fn perturbed_on(mut self, backend: BackendKind, p: Perturb) -> Self {
        self.perturb_on.push((backend, p));
        self
    }

    /// Loading on `backend` fails with an engine error.
    pub fn failing_load_on(mut self, backend: BackendKind) -> Self {
        self.fail_load_on.push(backend);
        self
    }

    /// Running a vector on `backend` fails with an engine error.
    pub fn failing_run_on(mut self, backend: BackendKind) -> Self {
        self.fail_run_on.push(backend);
        self
    }

    /// Make every `score` fail (or stop failing): an error during a request.
    pub fn fail_scores(&self, on: bool) {
        self.counters.fail_scores.store(on, Ordering::SeqCst);
    }

    /// The identity of the device of every load so far.
    pub fn devices_loaded(&self) -> Vec<String> {
        self.counters.devices.lock().unwrap().clone()
    }

    pub fn perturbed(mut self, p: Perturb) -> Self {
        self.perturb = p;
        self
    }

    pub fn loads(&self) -> usize {
        self.counters.loads.load(Ordering::SeqCst)
    }

    pub fn runs(&self) -> usize {
        self.counters.runs.load(Ordering::SeqCst)
    }
}

impl Default for FakeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for FakeBackend {
    fn load(&self, checkpoint: &ValidatedCheckpoint) -> Result<Engine> {
        self.counters.loads.fetch_add(1, Ordering::SeqCst);
        let backend = checkpoint.device.backend;
        self.counters
            .devices
            .lock()
            .unwrap()
            .push(checkpoint.device.identity());
        if self.fail_load_on.contains(&backend) {
            return Err(Error::Inference {
                detail: format!("the fake engine cannot use the {backend} device"),
            });
        }
        if self.fail_load {
            return Err(crate::error::Error::IncompatibleModel(
                crate::models::incompat::Incompat::ConversionFailed {
                    detail: "the fake backend was told to fail".to_string(),
                },
            ));
        }
        let family = families::lookup(&checkpoint.manifest.family);
        let n_vocab = family
            .and_then(|f| {
                let spec = checkpoint.manifest.architecture.clone();
                f.read_params(&spec).ok().map(|p| (f.n_vocab)(&p))
            })
            .unwrap_or(32);
        let model = Arc::new(FakeModel {
            calibration: checkpoint.calibration.calibration,
            n_vocab,
            perturb: self
                .perturb_on
                .iter()
                .find(|(b, _)| *b == backend)
                .map_or(self.perturb, |(_, p)| *p),
            fail_run: self.fail_run_on.contains(&backend),
            counters: Arc::clone(&self.counters),
            mock: MockScorer::new(1).with_calibration(checkpoint.calibration.calibration),
        });
        Ok(Engine {
            scorer: model.clone(),
            runner: model,
        })
    }
}
