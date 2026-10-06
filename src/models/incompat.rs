//! Every reason a checkpoint is refused, with the message the user sees (spec 005, §6).
//!
//! One variant per reason: each has a test that provokes exactly that reason. The messages say
//! which check failed, which file or field, the value received and the value expected.

use std::fmt;

use crate::json_strict::{FieldProblem, JsonProblem};

/// Why a model could not be loaded (surfaces as `IncompatibleModelError`).
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum Incompat {
    // ---- stage 0: which model ----
    UnknownName {
        name: String,
        known: Vec<String>,
    },
    RevisionNotAccepted {
        name: String,
        revision: String,
        available: Vec<String>,
    },
    DtypeNotOffered {
        dtype: String,
        available: Vec<String>,
    },
    // ---- stage 1: reading the manifest ----
    ManifestAbsent {
        expected: String,
    },
    ManifestSyntax {
        detail: String,
    },
    ManifestLimit {
        detail: String,
    },
    ManifestDuplicateKey {
        key: String,
    },
    ManifestUnknownField {
        path: String,
        field: String,
    },
    ManifestMissingField {
        path: String,
    },
    ManifestBadValue {
        path: String,
        detail: String,
    },
    SchemaVersion {
        got: String,
        supported: String,
    },
    NameCollision {
        name: String,
    },
    // ---- stage 2: what the manifest declares ----
    UnknownFamily {
        got: String,
        supported: Vec<String>,
    },
    ArchitectureMismatch {
        family: String,
        expected: String,
        got: String,
    },
    HeadMismatch {
        family: String,
        expected: String,
        got: String,
    },
    TemplateMismatch {
        family: String,
        expected: String,
        got: String,
    },
    NoTolerance {
        family: String,
        dtype: String,
    },
    SourceWithoutConverter {
        kind: String,
    },
    MissingVectors {
        dtype: String,
        kind: String,
    },
    BadVector {
        id: String,
        detail: String,
    },
    BadCalibration {
        detail: String,
    },
    MissingLicense,
    /// A family with a pointer head has no way to read a closed set of tasks (spec 006a).
    TasksNotSupported {
        family: String,
    },
    /// The pointer head and the backbone disagree on the hidden size.
    HeadDimension {
        d_model: u64,
        n_embd: u64,
    },
    BadTasks {
        index: usize,
        detail: String,
    },
    Uncalibrated {
        name: String,
        dtype: String,
    },
    // ---- stage 3: files ----
    FileMissing {
        path: String,
    },
    SizeMismatch {
        path: String,
        expected: u64,
        got: u64,
    },
    HashMismatch {
        path: String,
        expected: String,
        got: String,
        size: u64,
    },
    // ---- stage 4: GGUF ----
    GgufUnreadable {
        path: String,
        detail: String,
    },
    GgufImpossible {
        path: String,
        detail: String,
    },
    GgufMetadata {
        key: String,
        expected: String,
        got: String,
    },
    ContextTooLong {
        max_context: u64,
        context_length: u64,
    },
    TensorMissing {
        name: String,
    },
    TensorUnexpected {
        name: String,
    },
    TensorShape {
        name: String,
        expected: Vec<u64>,
        got: Vec<u64>,
    },
    TensorType {
        name: String,
        expected: String,
        got: String,
    },
    // ---- stage 5: tokenizer ----
    TokenizerUnreadable {
        path: String,
        detail: String,
    },
    TokenizerTruncationOrPadding {
        which: String,
    },
    SpecialRoles {
        expected: Vec<String>,
        got: Vec<String>,
    },
    SpecialTokenText {
        role: String,
        text: String,
        expected_id: u32,
        found: Option<u32>,
    },
    SpecialTokenNotSingle {
        role: String,
        text: String,
        ids: Vec<u32>,
    },
    SpecialSameId {
        a: String,
        b: String,
        id: u32,
    },
    LettersTarget {
        label: String,
        id: u32,
        detail: String,
    },
    LettersLabelNotSingle {
        label: String,
        ids: Vec<u32>,
    },
    // ---- stage 6: head weights ----
    HeadWeights {
        detail: String,
    },
    // ---- conversion (018) ----
    ConversionFailed {
        detail: String,
    },
}

fn list(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

impl fmt::Display for Incompat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Incompat as I;
        match self {
            I::UnknownName { name, known } => write!(
                f,
                "'{name}' is not a known model and not a local folder; known models: {}; to load a local folder pass a pathlib.Path",
                list(known)
            ),
            I::RevisionNotAccepted {
                name,
                revision,
                available,
            } => write!(
                f,
                "revision '{revision}' of '{name}' is not accepted: only pinned revisions are supported ({}); branch and tag names are not pinned",
                list(available)
            ),
            I::DtypeNotOffered { dtype, available } => write!(
                f,
                "dtype '{dtype}' is not available for this model; available: {}",
                list(available)
            ),
            I::ManifestAbsent { expected } => write!(
                f,
                "no manifest found: expected {expected}; models need a manifest (see the documentation on custom models)"
            ),
            I::ManifestSyntax { detail } => write!(f, "the manifest is not valid: {detail}"),
            I::ManifestLimit { detail } => write!(f, "the manifest exceeds a limit: {detail}"),
            I::ManifestDuplicateKey { key } => write!(
                f,
                "the manifest is not valid: the key \"{key}\" appears more than once in the same object"
            ),
            I::ManifestUnknownField { path, field } => write!(
                f,
                "the manifest has an unknown field \"{field}\" in {path}; unknown fields are rejected, check the spelling"
            ),
            I::ManifestMissingField { path } => write!(
                f,
                "the manifest is missing the required field {path}; nothing is assumed for missing fields"
            ),
            I::ManifestBadValue { path, detail } => {
                write!(f, "the manifest field {path} is not valid: {detail}")
            }
            I::SchemaVersion { got, supported } => write!(
                f,
                "manifest schema_version {got} is not supported by this archai-jev (supports {supported}); upgrade archai-jev or use a manifest of a supported version"
            ),
            I::NameCollision { name } => write!(
                f,
                "the local manifest is named '{name}', which is a model of the built-in registry; choose another name"
            ),
            I::UnknownFamily { got, supported } => write!(
                f,
                "manifest field family='{got}' is not supported by this archai-jev; supported: {}",
                list(supported)
            ),
            I::ArchitectureMismatch {
                family,
                expected,
                got,
            } => write!(
                f,
                "manifest architecture '{got}' does not match family '{family}', which needs '{expected}'"
            ),
            I::HeadMismatch {
                family,
                expected,
                got,
            } => write!(
                f,
                "manifest head '{got}' does not match family '{family}', which needs '{expected}'"
            ),
            I::TemplateMismatch {
                family,
                expected,
                got,
            } => write!(
                f,
                "manifest template '{got}' does not match family '{family}', which needs '{expected}'"
            ),
            I::NoTolerance { family, dtype } => write!(
                f,
                "no verification tolerance is defined for family '{family}' with dtype '{dtype}', so its numbers cannot be verified; the variant is refused (tolerances are in the library, not in manifests)"
            ),
            I::SourceWithoutConverter { kind } => write!(
                f,
                "source kind '{kind}' needs the checkpoint converter, which is not available in this build"
            ),
            I::MissingVectors { dtype, kind } => write!(
                f,
                "variant '{dtype}' has no self-check vector for {kind} questions; every supported question kind needs at least one"
            ),
            I::BadVector { id, detail } => {
                write!(f, "self-check vector '{id}' is not valid: {detail}")
            }
            I::BadCalibration { detail } => {
                write!(f, "the calibration declaration is not valid: {detail}")
            }
            I::MissingLicense => write!(
                f,
                "the manifest must declare a license (license.spdx is missing or empty)"
            ),
            I::TasksNotSupported { family } => write!(
                f,
                "family '{family}' reads no closed set of tasks: set \"tasks\" to null"
            ),
            I::HeadDimension { d_model, n_embd } => write!(
                f,
                "head.d_model is {d_model} but architecture.n_embd is {n_embd}; the head reads the backbone's hidden state, so they must be equal"
            ),
            I::BadTasks { index, detail } => write!(f, "tasks[{index}] is not valid: {detail}"),
            I::Uncalibrated { name, dtype } => write!(
                f,
                "model '{name}' ({dtype}) has no declared calibration: pass temperature=<value> (see archai_jev.fit_temperature) or allow_uncalibrated=True to accept uncalibrated probabilities (answers will carry calibrated=False)"
            ),
            I::FileMissing { path } => write!(f, "file '{path}' does not exist"),
            I::SizeMismatch {
                path,
                expected,
                got,
            } => write!(
                f,
                "file '{path}' is {got} bytes, expected {expected}; the file is truncated or is not the pinned revision; delete it and retry"
            ),
            I::HashMismatch {
                path,
                expected,
                got,
                size,
            } => write!(
                f,
                "file '{path}' has sha256 {got}, expected {expected} (size {size}); the file is corrupted or is not the pinned revision; delete it and retry"
            ),
            I::GgufUnreadable { path, detail } => {
                write!(f, "'{path}' is not a readable GGUF file: {detail}")
            }
            I::GgufImpossible { path, detail } => {
                write!(f, "GGUF file '{path}' is inconsistent: {detail}")
            }
            I::GgufMetadata { key, expected, got } => write!(
                f,
                "GGUF metadata {key} is {got}, but the manifest declares {expected}"
            ),
            I::ContextTooLong {
                max_context,
                context_length,
            } => write!(
                f,
                "manifest max_context {max_context} is larger than the context length {context_length} of the model file"
            ),
            I::TensorMissing { name } => {
                write!(f, "tensor '{name}' is missing from the model file")
            }
            I::TensorUnexpected { name } => {
                write!(
                    f,
                    "tensor '{name}' is in the model file but not expected for this family"
                )
            }
            I::TensorShape {
                name,
                expected,
                got,
            } => write!(
                f,
                "tensor '{name}' has shape {got:?}, expected {expected:?} for the declared architecture"
            ),
            I::TensorType {
                name,
                expected,
                got,
            } => write!(
                f,
                "tensor '{name}' has type {got}, expected {expected} for the declared dtype"
            ),
            I::TokenizerUnreadable { path, detail } => {
                write!(
                    f,
                    "tokenizer file '{path}' cannot be read as a tokenizer: {detail}"
                )
            }
            I::TokenizerTruncationOrPadding { which } => write!(
                f,
                "tokenizer.json has {which} enabled; it would truncate or pad prompts silently, so it is refused (use the tokenizer.json of the base model)"
            ),
            I::SpecialRoles { expected, got } => write!(
                f,
                "special token roles are [{}], but the template needs exactly [{}]",
                list(got),
                list(expected)
            ),
            I::SpecialTokenText {
                role,
                text,
                expected_id,
                found,
            } => match found {
                None => write!(
                    f,
                    "special token '{role}' ({text:?}) is not in the tokenizer vocabulary (expected id {expected_id})"
                ),
                Some(real) => write!(
                    f,
                    "special token '{role}' ({text:?}) has id {real} in the tokenizer, but the manifest declares {expected_id}"
                ),
            },
            I::SpecialTokenNotSingle { role, text, ids } => write!(
                f,
                "special token '{role}' ({text:?}) encodes to {} tokens {ids:?}, expected exactly one token",
                ids.len()
            ),
            I::SpecialSameId { a, b, id } => write!(
                f,
                "special tokens '{a}' and '{b}' have the same id {id}; every role needs its own token"
            ),
            I::LettersTarget { label, id, detail } => {
                write!(
                    f,
                    "answer target '{label}' (id {id}) is not valid: {detail}"
                )
            }
            I::LettersLabelNotSingle { label, ids } => write!(
                f,
                "answer label '{label}' encodes to {} tokens {ids:?}, expected exactly one token",
                ids.len()
            ),
            I::HeadWeights { detail } => {
                write!(f, "the decision head weights are not valid: {detail}")
            }
            I::ConversionFailed { detail } => write!(f, "checkpoint conversion failed: {detail}"),
        }
    }
}

impl From<FieldProblem> for Incompat {
    fn from(p: FieldProblem) -> Self {
        match p {
            FieldProblem::Missing { path } => Incompat::ManifestMissingField { path },
            FieldProblem::Unknown { path, field } => Incompat::ManifestUnknownField { path, field },
            FieldProblem::Bad { path, detail } => Incompat::ManifestBadValue { path, detail },
        }
    }
}

impl From<JsonProblem> for Incompat {
    fn from(p: JsonProblem) -> Self {
        match p {
            JsonProblem::Syntax(detail) => Incompat::ManifestSyntax { detail },
            JsonProblem::Limit(detail) => Incompat::ManifestLimit { detail },
            JsonProblem::DuplicateKey(key) => Incompat::ManifestDuplicateKey { key },
        }
    }
}
