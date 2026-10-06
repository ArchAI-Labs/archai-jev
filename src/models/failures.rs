//! The other families of model-loading errors: failed self-check, download problems, and
//! wrong arguments to `from_pretrained`.

use std::fmt;

/// A self-check vector disagreed with the model (surfaces as `ModelVerificationError`).
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyFailure {
    /// The prompt built for a vector has different token ids than expected.
    Ids {
        model: String,
        vector: String,
        position: usize,
        expected: Option<u32>,
        got: Option<u32>,
        expected_len: usize,
        got_len: usize,
    },
    /// A probability differs from the expected one by more than the tolerance.
    Probability {
        model: String,
        vector: String,
        question: String,
        option: usize,
        expected: f64,
        got: f64,
        tolerance: f64,
    },
    /// A logit differs from the expected one by more than the tolerance.
    Logit {
        model: String,
        vector: String,
        question: String,
        option: usize,
        expected: f64,
        got: f64,
        tolerance: f64,
    },
    /// The model produced a non-finite number.
    NonFinite {
        model: String,
        vector: String,
        question: String,
        option: usize,
        value: f64,
    },
    /// The model answered with the wrong number of questions or options.
    Shape {
        model: String,
        vector: String,
        detail: String,
    },
}

impl fmt::Display for VerifyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const HINT: &str = "the files passed the integrity checks, so a template, tokenizer, backend or platform mismatch is likely; the model is not usable";
        match self {
            VerifyFailure::Ids {
                model,
                vector,
                position,
                expected,
                got,
                expected_len,
                got_len,
            } => write!(
                f,
                "self-check failed for {model} on vector '{vector}': the prompt token ids differ at position {position} (expected {}, got {}; lengths {expected_len} and {got_len}); {HINT}",
                show(*expected),
                show(*got)
            ),
            VerifyFailure::Probability {
                model,
                vector,
                question,
                option,
                expected,
                got,
                tolerance,
            } => write!(
                f,
                "self-check failed for {model} on vector '{vector}', question '{question}', option {option}: probability {got} differs from the expected {expected} by {:.3e}, more than the tolerance {tolerance:.1e}; {HINT}",
                (got - expected).abs()
            ),
            VerifyFailure::Logit {
                model,
                vector,
                question,
                option,
                expected,
                got,
                tolerance,
            } => write!(
                f,
                "self-check failed for {model} on vector '{vector}', question '{question}', option {option}: logit {got} differs from the expected {expected} by {:.3e}, more than the tolerance {tolerance:.1e}; {HINT}",
                (got - expected).abs()
            ),
            VerifyFailure::NonFinite {
                model,
                vector,
                question,
                option,
                value,
            } => write!(
                f,
                "self-check failed for {model} on vector '{vector}', question '{question}', option {option}: non-finite value {value}; {HINT}"
            ),
            VerifyFailure::Shape {
                model,
                vector,
                detail,
            } => write!(
                f,
                "self-check failed for {model} on vector '{vector}': {detail}; {HINT}"
            ),
        }
    }
}

fn show(id: Option<u32>) -> String {
    id.map_or_else(|| "nothing".to_string(), |i| i.to_string())
}

/// A model file could not be fetched or found (surfaces as `ModelDownloadError`).
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum DownloadFailure {
    Network {
        file: String,
        url: String,
        cause: String,
    },
    DiskFull {
        file: String,
        dir: String,
        need: u64,
    },
    NotFound {
        file: String,
        url: String,
        status: u16,
    },
    Refused {
        file: String,
        url: String,
        status: u16,
    },
    Offline {
        file: String,
        size: u64,
        dir: String,
    },
    NoCacheDir,
    Io {
        path: String,
        cause: String,
    },
}

impl fmt::Display for DownloadFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadFailure::Network { file, url, cause } => {
                write!(f, "cannot download {file} from {url}: {cause}")
            }
            DownloadFailure::DiskFull { file, dir, need } => write!(
                f,
                "not enough disk space to store {file} in {dir}: {need} more bytes needed"
            ),
            DownloadFailure::NotFound { file, url, status } => write!(
                f,
                "cannot download {file}: repository or revision not found (HTTP {status}) at {url}"
            ),
            DownloadFailure::Refused { file, url, status } => write!(
                f,
                "cannot download {file}: access refused (HTTP {status}) at {url}; only public repositories are supported, no token is used"
            ),
            DownloadFailure::Offline { file, size, dir } => write!(
                f,
                "offline mode: {file} ({size} bytes) is not in the cache at {dir}"
            ),
            DownloadFailure::NoCacheDir => write!(
                f,
                "cannot determine a cache directory; set ARCHAI_JEV_CACHE or pass cache_dir"
            ),
            DownloadFailure::Io { path, cause } => {
                write!(f, "cannot read or write {path}: {cause}")
            }
        }
    }
}

/// A wrong argument to `from_pretrained` (surfaces as `ValueError`, or `FileNotFoundError`).
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgFailure {
    Ambiguous { name: String },
    PathNotFound { path: String },
    Device { got: String },
    Dtype { got: String },
    RevisionForLocal,
    ManifestWithRegistryName { name: String },
}

impl fmt::Display for ArgFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgFailure::Ambiguous { name } => write!(
                f,
                "'{name}' is both the name of a built-in model and an existing folder; pass a pathlib.Path for the folder, or rename it"
            ),
            ArgFailure::PathNotFound { path } => {
                write!(f, "model folder '{path}' does not exist")
            }
            ArgFailure::Device { got } => write!(
                f,
                "device '{got}' is not supported; supported: 'cpu' (GPU support is planned)"
            ),
            ArgFailure::Dtype { got } => write!(
                f,
                "'{got}' is not a known dtype; known: f32, bf16, q8_0 (float32 and bfloat16 are accepted as synonyms)"
            ),
            ArgFailure::RevisionForLocal => write!(
                f,
                "revision cannot be used with a local folder: the revision of a local model is the one declared in its manifest"
            ),
            ArgFailure::ManifestWithRegistryName { name } => write!(
                f,
                "manifest= is for local folders and cannot be combined with the built-in model name '{name}'"
            ),
        }
    }
}
