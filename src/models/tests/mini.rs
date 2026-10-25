//! The whole path with the real converter: a local model made of the original files of the tiny
//! checkpoint (`tests/data/qwen35-mini`), converted on first use, then loaded with the fake engine.
//! These are the loading-side criteria of spec 018: the record that vouches for a converted model,
//! the rebuild of a damaged one, the cache under concurrency, interruption and failures.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::Harness;
use crate::convert::headpt::HeadPtReader;
use crate::convert::{Qwen35Converter, RECIPE};
use crate::error::{Error, Result};
use crate::hub::events::{CancelFlag, Event};
use crate::json_strict::Json;
use crate::models::convert::{ConvertJob, Converted, Converter};
use crate::models::hash::sha256_hex;
use crate::models::head::HeadReader;
use crate::models::incompat::Incompat;
use crate::models::materialize;
use crate::models::testing::builder::{Builder, MANIFEST_FILE};
use crate::models::testing::jsonedit::{arr, b, f, n, obj, s, set};

const BASE_REPO: &str = "Org/Mini-Base";
const BASE_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("qwen35-mini")
}

/// A converter that counts, can be told to stop after converting, and has a version of its own.
struct Spy {
    inner: Qwen35Converter,
    calls: AtomicUsize,
    version: String,
    fail_after: bool,
    /// Simulates another process that finishes the same conversion first (altered afterwards).
    rival: bool,
}

impl Spy {
    fn new() -> Arc<Spy> {
        Arc::new(Spy {
            inner: Qwen35Converter,
            calls: AtomicUsize::new(0),
            version: RECIPE.to_string(),
            fail_after: false,
            rival: false,
        })
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Converter for Spy {
    fn kinds(&self) -> Vec<String> {
        self.inner.kinds()
    }
    fn version(&self) -> String {
        self.version.clone()
    }
    fn convert(&self, job: &ConvertJob<'_>, out_dir: &Path) -> Result<Converted> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.rival && call == 1 {
            // Another process finishes first, with a file that is not what its record says.
            let (other, _) = materialize::build(&self.inner, job, &job.destination)?;
            let mut bytes = std::fs::read(&other.model).unwrap();
            let at = bytes.len() / 2;
            bytes[at] ^= 0xFF;
            std::fs::write(&other.model, bytes).unwrap();
        }
        let done = self.inner.convert(job, out_dir)?;
        if self.fail_after {
            return Err(Error::IncompatibleModel(Incompat::ConversionFailed {
                detail: "stopped after writing the file".to_string(),
            }));
        }
        Ok(done)
    }
}

fn entry(dir: &Path, path: &str) -> Json {
    let bytes = std::fs::read(dir.join(path)).unwrap();
    obj(vec![
        ("path", s(path)),
        ("size", n(bytes.len() as u64)),
        ("sha256", s(&sha256_hex(&bytes))),
    ])
}

/// A local model folder: the originals of the tiny checkpoint and a manifest with both variants.
fn mini_model() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for sub in ["base", "adapter"] {
        std::fs::create_dir_all(dir.path().join(sub)).unwrap();
        for e in std::fs::read_dir(fixture().join(sub)).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), dir.path().join(sub).join(e.file_name())).unwrap();
        }
    }
    for file in ["head.pt", "prompt-tokenizer.json"] {
        std::fs::copy(fixture().join(file), dir.path().join(file)).unwrap();
    }
    let d = dir.path();
    let vector = Builder::tiny_kev().vocab(40).vector(2.0);
    let source = obj(vec![
        ("kind", s("hf-lora")),
        (
            "base",
            obj(vec![
                ("repo", s(BASE_REPO)),
                ("revision", s(BASE_REVISION)),
                ("config", entry(d, "base/config.json")),
                ("weights", arr(vec![entry(d, "base/model.safetensors")])),
                ("tokenizer", entry(d, "base/tokenizer.json")),
                ("tokenizer_config", entry(d, "base/tokenizer_config.json")),
            ]),
        ),
        (
            "adapter",
            obj(vec![
                ("config", entry(d, "adapter/adapter_config.json")),
                ("weights", entry(d, "adapter/adapter_model.safetensors")),
            ]),
        ),
    ]);
    let variant = |dtype: &str| {
        obj(vec![
            ("dtype", s(dtype)),
            ("source", source.clone()),
            (
                "calibration",
                obj(vec![
                    ("declared", b(true)),
                    ("temperature", f(2.0)),
                    (
                        "evidence",
                        s("the temperature stored in head.pt of the fixture"),
                    ),
                ]),
            ),
            (
                "selfcheck",
                obj(vec![("vectors", arr(vec![vector.clone()]))]),
            ),
        ])
    };
    let role = |text: &str, id: u64| obj(vec![("text", s(text)), ("id", n(id))]);
    let int = |x: u64| n(x);
    let manifest = obj(vec![
        ("schema_version", n(1)),
        ("name", s("local/mini-kev")),
        ("revision", s("v1")),
        ("family", s("qwen35-pointer")),
        (
            "architecture",
            obj(vec![
                ("name", s("qwen35")),
                ("n_layers", int(4)),
                ("n_mtp_layers", int(1)),
                ("n_embd", int(16)),
                ("n_ff", int(32)),
                ("n_heads", int(2)),
                ("n_kv_heads", int(1)),
                ("head_dim", int(8)),
                ("n_vocab", int(40)),
                ("tie_embeddings", b(true)),
                ("full_attention_interval", int(4)),
                ("ssm_conv_kernel", int(4)),
                ("ssm_state_size", int(4)),
                ("ssm_group_count", int(2)),
                ("ssm_time_step_rank", int(2)),
                ("ssm_inner_size", int(8)),
                ("rope_theta", int(10_000_000)),
                ("rope_dim", int(2)),
            ]),
        ),
        (
            "head",
            obj(vec![
                ("kind", s("pointer")),
                ("d_model", n(16)),
                ("proj_dim", n(4)),
                ("weights", entry(d, "head.pt")),
            ]),
        ),
        ("template", obj(vec![("id", s("kev")), ("version", n(1))])),
        (
            "tokenizer",
            obj(vec![
                ("file", entry(d, "prompt-tokenizer.json")),
                (
                    "special_tokens",
                    obj(vec![
                        ("fim_prefix", role("<|fim_prefix|>", 32)),
                        ("fim_middle", role("<|fim_middle|>", 33)),
                        ("fim_suffix", role("<|fim_suffix|>", 34)),
                        ("box_start", role("<|box_start|>", 35)),
                        ("box_end", role("<|box_end|>", 36)),
                    ]),
                ),
            ]),
        ),
        ("max_context", n(128)),
        (
            "license",
            obj(vec![
                ("spdx", s("Apache-2.0")),
                ("url", s("https://example.test/license")),
            ]),
        ),
        ("tasks", Json::Null),
        ("default_dtype", s("bf16")),
        ("variants", arr(vec![variant("bf16"), variant("f32")])),
    ]);
    std::fs::write(d.join(MANIFEST_FILE), manifest.to_canonical_string()).unwrap();
    dir
}

fn harness(spy: &Arc<Spy>) -> Harness {
    let mut h = Harness::new();
    h.converter = Some(spy.clone());
    let reader: Arc<dyn HeadReader> = Arc::new(HeadPtReader);
    h.head_reader = Some(reader);
    h
}

fn materialized(h: &Harness) -> PathBuf {
    h.cache.path().join("v1").join("materialized")
}

fn count_events(h: &Harness, pick: impl Fn(&Event) -> bool) -> usize {
    h.recorder.events().iter().filter(|e| pick(e)).count()
}

#[test]
fn the_originals_are_converted_on_first_use_and_the_model_loads() {
    for dtype in ["bf16", "f32"] {
        let dir = mini_model();
        let spy = Spy::new();
        let h = harness(&spy);
        let mut req = h.local(dir.path());
        req.selection.dtype = Some(dtype.to_string());
        let m = h.load(&req).unwrap();
        assert_eq!(m.data.family, "qwen35-pointer");
        assert_eq!(m.data.dtype, dtype);
        assert!(m.data.calibrated);
        assert_eq!(spy.calls(), 1);
        assert_eq!(h.backend.loads(), 1);
        assert_eq!(h.backend.runs(), 1, "the self-check ran once");
        assert_eq!(h.records(), 1);
        assert_eq!(
            count_events(&h, |e| matches!(e, Event::ConvertStart { .. })),
            1
        );
        assert_eq!(
            count_events(&h, |e| matches!(e, Event::ConvertDone { .. })),
            1
        );
    }
}

#[test]
fn the_second_load_converts_nothing_and_checks_nothing_again() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    h.load(&h.local(dir.path())).unwrap();
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(spy.calls(), 1, "no second conversion");
    assert_eq!(h.backend.runs(), 1, "no second self-check");
    assert_eq!(h.backend.loads(), 2, "the engine loads every time");
    assert_eq!(h.records(), 1);
    assert_eq!(
        count_events(&h, |e| matches!(e, Event::ConvertStart { .. })),
        1
    );
    // the fast path trusts a file whose size and modification time did not change: corrupt a
    // byte of an original and restore the time, and nothing re-hashes it (documented limit)
    let weights = dir.path().join("base/model.safetensors");
    let before = std::fs::metadata(&weights).unwrap().modified().unwrap();
    let mut bytes = std::fs::read(&weights).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&weights, bytes).unwrap();
    std::fs::File::options()
        .write(true)
        .open(&weights)
        .unwrap()
        .set_modified(before)
        .unwrap();
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(spy.calls(), 1);
}

#[test]
fn an_original_that_changed_is_found_before_anything_else_happens() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    h.load(&h.local(dir.path())).unwrap();
    // now the file changes (and its time with it): the record no longer applies
    let weights = dir.path().join("adapter/adapter_model.safetensors");
    let mut bytes = std::fs::read(&weights).unwrap();
    bytes[40] ^= 0x01;
    std::fs::write(&weights, bytes).unwrap();
    let loads = h.backend.loads();
    let err = h.load(&h.local(dir.path())).err().unwrap();
    assert!(
        matches!(err, Error::IncompatibleModel(Incompat::HashMismatch { .. })),
        "{err:?}"
    );
    assert_eq!(h.backend.loads(), loads, "the engine was not touched");
    assert_eq!(spy.calls(), 1);
}

#[test]
fn a_converted_file_that_no_longer_matches_its_record_is_rebuilt_once() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    h.load(&h.local(dir.path())).unwrap();
    // damage the GGUF in the cache (the time changes with the write, so the record does not vouch)
    let folder = std::fs::read_dir(materialized(&h))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let gguf = std::fs::read_dir(folder.join("bf16"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
        .join("model.gguf");
    let mut bytes = std::fs::read(&gguf).unwrap();
    let at = bytes.len() / 2;
    bytes[at] ^= 0xFF;
    std::fs::write(&gguf, &bytes).unwrap();
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(
        spy.calls(),
        2,
        "converted again from the verified originals"
    );
    let warnings = h
        .recorder
        .events()
        .iter()
        .filter(|e| matches!(e, Event::Warning(w) if w.contains("converting again")))
        .count();
    assert_eq!(warnings, 1);
    // and the rebuilt file is the right one: a third load is quiet
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(spy.calls(), 2);
}

#[test]
fn a_new_version_of_the_recipe_converts_again_and_keeps_the_old_folder() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    h.load(&h.local(dir.path())).unwrap();
    let mut v2 = Spy {
        inner: Qwen35Converter,
        calls: AtomicUsize::new(0),
        version: "qwen35-lora/2".to_string(),
        fail_after: false,
        rival: false,
    };
    v2.fail_after = false;
    let v2 = Arc::new(v2);
    let mut h2 = harness(&v2);
    h2.cache = h.cache;
    h2.load(&h2.local(dir.path())).unwrap();
    assert_eq!(v2.calls(), 1);
    let versions = std::fs::read_dir(
        std::fs::read_dir(materialized(&h2))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .join("bf16"),
    )
    .unwrap()
    .count();
    assert_eq!(versions, 2, "both versions are in the cache");
}

#[test]
fn an_interrupted_conversion_leaves_nothing_with_a_final_name_and_the_retry_works() {
    let dir = mini_model();
    let mut flaky = Spy {
        inner: Qwen35Converter,
        calls: AtomicUsize::new(0),
        version: RECIPE.to_string(),
        fail_after: true,
        rival: false,
    };
    flaky.fail_after = true;
    let flaky = Arc::new(flaky);
    let h = harness(&flaky);
    assert!(h.load(&h.local(dir.path())).is_err());
    let leftovers: Vec<_> = walk(&materialized(&h));
    assert!(
        leftovers
            .iter()
            .all(|p| !p.ends_with(materialize::RECORD)
                && !p.to_string_lossy().contains("model.gguf")),
        "{leftovers:?}"
    );
    // the same cache, a converter that works
    let spy = Spy::new();
    let mut h2 = harness(&spy);
    h2.cache = h.cache;
    h2.load(&h2.local(dir.path())).unwrap();
    assert_eq!(spy.calls(), 1);
}

#[test]
fn a_result_found_in_the_cache_after_a_lost_race_is_checked_like_any_other_file() {
    let dir = mini_model();
    let rival = Arc::new(Spy {
        inner: Qwen35Converter,
        calls: AtomicUsize::new(0),
        version: RECIPE.to_string(),
        fail_after: false,
        rival: true,
    });
    let h = harness(&rival);
    // the other process' file does not match its record: it is not trusted because it was not
    // written here, and the model is converted again (not loaded as it is)
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(
        rival.calls(),
        2,
        "the altered file of the rival was replaced"
    );
    let warnings = h
        .recorder
        .events()
        .iter()
        .filter(|e| matches!(e, Event::Warning(w) if w.contains("converting again")))
        .count();
    assert_eq!(warnings, 1);
    // and what is in the cache now is good: the next load is quiet
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(rival.calls(), 2);
}

#[test]
fn a_folder_without_a_record_in_the_way_is_replaced_once() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    h.load(&h.local(dir.path())).unwrap();
    // an unusable folder with the final name (no record), as a crashed older version could leave
    let folder = walk(&materialized(&h))
        .into_iter()
        .find(|p| p.join(materialize::RECORD).is_file())
        .unwrap();
    std::fs::remove_file(folder.join(materialize::RECORD)).unwrap();
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(spy.calls(), 2, "converted again");
    assert!(folder.join(materialize::RECORD).is_file());
    h.load(&h.local(dir.path())).unwrap();
    assert_eq!(spy.calls(), 2);
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&p) else {
            continue;
        };
        for e in rd.flatten() {
            out.push(e.path());
            if e.path().is_dir() {
                stack.push(e.path());
            }
        }
    }
    out
}

#[test]
fn two_threads_loading_for_the_first_time_both_succeed_with_one_valid_result() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| scope.spawn(|| h.load(&h.local(dir.path())).map(|m| m.data.dtype)))
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap().unwrap(), "bf16");
        }
    });
    assert!((1..=2).contains(&spy.calls()));
    // exactly one folder per (manifest, dtype, version), and no temporary folder left over
    let all = walk(&materialized(&h));
    assert!(
        all.iter().all(|p| !p.to_string_lossy().contains(".tmp-")),
        "{all:?}"
    );
    h.load(&h.local(dir.path())).unwrap();
    assert!((1..=2).contains(&spy.calls()));
}

#[test]
fn a_request_to_stop_ends_the_conversion_and_leaves_no_folder() {
    let dir = mini_model();
    let spy = Spy::new();
    let mut h = harness(&spy);
    let flag = Arc::new(CancelFlag::default());
    flag.cancel();
    h.cancel = Some(flag);
    let err = h.load(&h.local(dir.path())).err().unwrap();
    assert!(matches!(err, Error::Cancelled), "{err:?}");
    assert_eq!(h.backend.loads(), 0);
    let all = walk(&materialized(&h));
    assert!(
        all.iter().all(|p| p.is_dir()),
        "only empty folders may remain: {all:?}"
    );
    assert!(
        all.iter().all(|p| !p.to_string_lossy().contains(".tmp-")),
        "{all:?}"
    );
}

#[test]
fn an_unwritable_cache_is_an_io_error_and_nothing_is_loaded() {
    let dir = mini_model();
    let spy = Spy::new();
    let h = harness(&spy);
    // a file where the folder of converted models should be
    std::fs::create_dir_all(h.cache.path().join("v1")).unwrap();
    std::fs::write(
        h.cache.path().join("v1").join("materialized"),
        b"in the way",
    )
    .unwrap();
    let err = h.load(&h.local(dir.path())).err().unwrap();
    assert!(matches!(err, Error::ModelDownload(_)), "{err:?}");
    assert_eq!(h.backend.loads(), 0);
}

#[test]
fn what_head_pt_says_about_itself_must_agree_with_the_manifest() {
    // the temperature stored in the file, the base it was trained on
    for (path, value, fragment) in [
        ("variants.0.calibration.temperature", f(1.5), "temperature"),
        ("variants.1.calibration.temperature", f(1.5), "temperature"),
        ("head.proj_dim", n(8), "tensor"),
    ] {
        let dir = mini_model();
        let mp = dir.path().join(MANIFEST_FILE);
        let mut j = crate::json_strict::parse(&std::fs::read(&mp).unwrap()).unwrap();
        set(&mut j, path, value);
        std::fs::write(&mp, j.to_canonical_string()).unwrap();
        let spy = Spy::new();
        let h = harness(&spy);
        let mut req = h.local(dir.path());
        req.selection.dtype = Some(
            if path.contains("variants.1") {
                "f32"
            } else {
                "bf16"
            }
            .to_string(),
        );
        let err = h.load(&req).err().unwrap();
        let Error::IncompatibleModel(Incompat::HeadWeights { detail }) = &err else {
            panic!("{path}: {err:?}")
        };
        assert!(detail.contains(fragment), "{path}: {detail}");
    }
    // a head trained on another base
    let dir = mini_model();
    let mp = dir.path().join(MANIFEST_FILE);
    let mut j = crate::json_strict::parse(&std::fs::read(&mp).unwrap()).unwrap();
    for v in 0..2 {
        set(
            &mut j,
            &format!("variants.{v}.source.base.revision"),
            s(&"b".repeat(40)),
        );
    }
    std::fs::write(&mp, j.to_canonical_string()).unwrap();
    let spy = Spy::new();
    let h = harness(&spy);
    let err = h.load(&h.local(dir.path())).err().unwrap();
    let Error::IncompatibleModel(Incompat::HeadWeights { detail }) = &err else {
        panic!("{err:?}")
    };
    assert!(detail.contains("base revision"), "{detail}");
}

#[test]
fn the_mini_model_has_the_files_the_tests_assume() {
    assert!(fixture().join("prompt-tokenizer.json").is_file());
    assert!(fixture().join("head.pt").is_file());
}
