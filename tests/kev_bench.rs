//! Measurements of the conversion of Kev-0.8B (spec 018, P1-P7; test level T2, by hand, `#[ignore]`).
//!
//! These are not assertions about speed: they print numbers, to be run in alternating blocks
//! against the official script by `spec/playground/s7-convert/bench_conversion.py`, which also
//! watches the peak memory of the process from outside. Same environment variables as
//! `kev_parity.rs`; `ARCHAI_JEV_BENCH_OUT` names the folder of `bench_convert`.
//!
//! Run with `cargo test --release --features testing --test kev_bench -- --ignored --nocapture
//! --test-threads=1`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use _core::convert::Qwen35Converter;
use _core::convert::headpt::HeadPtReader;
use _core::convert::qwen35::convert_files;
use _core::engine::device::CpuOnlyProbe;
use _core::hub::config::HubConfig;
use _core::hub::download::{Response, Transport};
use _core::hub::events::{NeverCancel, NoObserver, Recorder};
use _core::models::convert::SourceFiles;
use _core::models::hash::sha256_file;
use _core::models::llama_backend::LlamaBackend;
use _core::models::load::{Context, LoadRequest, load_model};
use _core::models::registry::Registry;
use _core::models::resolve::{NameOrPath, Selection};
use _core::models::validate;

struct NoNetwork;

impl Transport for NoNetwork {
    fn get(&self, _: &str, _: Option<u64>, _: Duration) -> Result<Response, String> {
        Err("the network is disabled in this test".to_string())
    }
}

fn snapshot(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

fn originals() -> Vec<PathBuf> {
    let kev = snapshot("ARCHAI_JEV_TEST_KEV_SNAPSHOT");
    let base = snapshot("ARCHAI_JEV_TEST_BASE_SNAPSHOT");
    vec![
        kev.join("adapter_config.json"),
        kev.join("adapter_model.safetensors"),
        kev.join("head.pt"),
        kev.join("tokenizer.json"),
        base.join("config.json"),
        base.join("model.safetensors-00001-of-00001.safetensors"),
        base.join("tokenizer.json"),
        base.join("tokenizer_config.json"),
    ]
}

fn populate(cache: &Path) {
    let blobs = cache.join("v1").join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    for path in originals() {
        let (sha, _) = sha256_file(&path).unwrap();
        let target = blobs.join(&sha);
        if !target.exists() {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// P1, P2, P3: only the conversion, into `ARCHAI_JEV_BENCH_OUT` (the peak memory is read from
/// outside; the process does nothing else).
#[test]
#[ignore = "measurement: needs the original files of Kev-0.8B"]
fn bench_convert() {
    let dtype = std::env::var("ARCHAI_JEV_BENCH_DTYPE").unwrap_or_else(|_| "bf16".to_string());
    let out =
        PathBuf::from(std::env::var("ARCHAI_JEV_BENCH_OUT").expect("set ARCHAI_JEV_BENCH_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let (kev, base) = (
        snapshot("ARCHAI_JEV_TEST_KEV_SNAPSHOT"),
        snapshot("ARCHAI_JEV_TEST_BASE_SNAPSHOT"),
    );
    let files = SourceFiles {
        base_config: base.join("config.json"),
        base_weights: base.join("model.safetensors-00001-of-00001.safetensors"),
        base_tokenizer: base.join("tokenizer.json"),
        base_tokenizer_config: base.join("tokenizer_config.json"),
        adapter_config: kev.join("adapter_config.json"),
        adapter_weights: kev.join("adapter_model.safetensors"),
    };
    let registry = Registry::builtin().unwrap();
    let manifest = registry.resolve("jaredpalmer/kev-0.8b", None).unwrap();
    let resolved = validate::semantics(manifest, &["hf-lora".to_string()]).unwrap();
    let started = Instant::now();
    let done = convert_files(
        &files,
        &resolved.params,
        "Qwen/Qwen3.5-0.8B-Base",
        &dtype,
        &out,
        &NoObserver,
        &NeverCancel,
    )
    .unwrap();
    println!(
        "BENCH convert {dtype} seconds {:.3} bytes {}",
        started.elapsed().as_secs_f64(),
        done.size.unwrap()
    );
}

/// P7: SHA-256 of the eight original files, as the download verification does it.
#[test]
#[ignore = "measurement: needs the original files of Kev-0.8B"]
fn bench_hash_originals() {
    let mut total = 0u64;
    let started = Instant::now();
    for path in originals() {
        let (_, size) = sha256_file(&path).unwrap();
        total += size;
    }
    let s = started.elapsed().as_secs_f64();
    println!(
        "BENCH hash seconds {s:.3} bytes {total} MB/s {:.0}",
        total as f64 / 1e6 / s
    );
}

/// P4 and P5: the first load of a cache with only the originals (conversion + engine + the eight
/// self-check vectors), then the loads that follow with the record valid.
#[test]
#[ignore = "measurement: needs the original files of Kev-0.8B"]
fn bench_loads() {
    let dtype = std::env::var("ARCHAI_JEV_BENCH_DTYPE").unwrap_or_else(|_| "bf16".to_string());
    let cache = tempfile::tempdir().unwrap();
    populate(cache.path());
    let mut hub = HubConfig::new(cache.path().to_path_buf());
    hub.offline = true;
    let backend = LlamaBackend::new(_core::models::llama_backend::default_threads());
    let load = |recorder: &Recorder| {
        let ctx = Context {
            registry: Registry::builtin().unwrap(),
            hub: &hub,
            backend: &backend,
            devices: &CpuOnlyProbe,
            transport: &NoNetwork,
            observer: recorder,
            cancel: &NeverCancel,
            head_reader: Some(&HeadPtReader),
            converter: Some(&Qwen35Converter),
        };
        let request = LoadRequest {
            selection: Selection {
                name_or_path: NameOrPath::Str("jaredpalmer/kev-0.8b".to_string()),
                revision: None,
                device: "cpu".to_string(),
                dtype: Some(dtype.clone()),
                manifest: None,
            },
            temperature: None,
            allow_uncalibrated: false,
        };
        let started = Instant::now();
        let loaded = load_model(&request, &ctx).unwrap();
        let seconds = started.elapsed().as_secs_f64();
        drop(loaded);
        seconds
    };
    let first_events = Recorder::default();
    let first = load(&first_events);
    println!("BENCH first_load {dtype} seconds {first:.3}");
    for e in first_events.events() {
        let text = format!("{e:?}");
        if text.starts_with("Convert") {
            println!("  event: {text}");
        }
    }
    let mut later = Vec::new();
    for _ in 0..7 {
        later.push(load(&Recorder::default()));
    }
    println!(
        "BENCH later_loads {dtype} seconds {:?} p50 {:.3}",
        later.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>(),
        median(later.clone())
    );
}

/// P6: reading the built-in registry, and the weight of the Kev entry in it.
#[test]
#[ignore = "measurement"]
fn bench_registry() {
    let mut times = Vec::new();
    for _ in 0..50 {
        let started = Instant::now();
        let r = Registry::builtin().unwrap();
        times.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(r);
    }
    let entry = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/models/registry/jaredpalmer__kev-0.8b.json");
    println!(
        "BENCH registry builtin ms p50 {:.3} kev_entry_bytes {}",
        median(times),
        std::fs::metadata(entry).unwrap().len()
    );
}
