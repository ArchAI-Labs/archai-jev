//! Builds a tiny, complete fake checkpoint in a temporary folder: a GGUF with a real header,
//! a tokenizer, a manifest with hashes and self-check vectors. Every defect a test needs is a
//! method of the builder; hashes are recomputed after the defect unless it is an integrity one.

use std::path::{Path, PathBuf};

use super::fake::{fake_ids, fake_logits, fake_probabilities};
use super::gguf_writer::{Meta, TensorDesc, write_with_offsets};
use super::jsonedit::{arr, b, f, n, obj, s, set};
use crate::json_strict::Json;
use crate::models::families::{self, ArchParams, ParamValue, Role};
use crate::models::hash::sha256_hex;

/// Name of the manifest file in a local model folder.
pub const MANIFEST_FILE: &str = "archai-jev-manifest.json";
/// A commit hash used by registry-style fixtures.
pub const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

type JsonEdit = Box<dyn Fn(&mut Json)>;

/// A built fake checkpoint.
pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub manifest: Json,
    pub n_vocab: u64,
}

impl Fixture {
    pub fn path(&self) -> &Path {
        self.dir.path()
    }
    pub fn manifest_path(&self) -> PathBuf {
        self.dir.path().join(MANIFEST_FILE)
    }
    pub fn gguf_path(&self) -> PathBuf {
        self.dir.path().join("model.gguf")
    }
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.path().join("tokenizer.json")
    }
}

/// Describes the fake checkpoint to build.
#[allow(clippy::type_complexity)]
pub struct Builder {
    family: String,
    dtype: String,
    registry: bool,
    n_layers: u64,
    n_embd: u64,
    n_ff: u64,
    n_heads: u64,
    n_kv_heads: u64,
    n_vocab: u64,
    tie: bool,
    context_length: u32,
    max_context: u64,
    declared: Option<f64>,
    tensor_edits: Vec<Box<dyn Fn(&mut Vec<TensorDesc>)>>,
    meta_edits: Vec<Box<dyn Fn(&mut Vec<(String, Meta)>)>>,
    gguf_edits: Vec<Box<dyn Fn(&mut Vec<u8>)>>,
    tokenizer_edits: Vec<JsonEdit>,
    manifest_edits: Vec<JsonEdit>,
    after_seal: Vec<Box<dyn Fn(&Path)>>,
    offsets: Option<Vec<u64>>,
}

fn role_text(role: &str) -> String {
    if role.starts_with("im_") {
        format!("<|{role}|>")
    } else {
        format!("<{role}>")
    }
}

impl Builder {
    /// A tiny `qwen2-letters` checkpoint, local, `q8_0`, no declared calibration.
    pub fn tiny() -> Self {
        Builder {
            family: "qwen2-letters".to_string(),
            dtype: "q8_0".to_string(),
            registry: false,
            n_layers: 1,
            n_embd: 32,
            n_ff: 64,
            n_heads: 2,
            n_kv_heads: 1,
            n_vocab: 32,
            tie: true,
            context_length: 2048,
            max_context: 512,
            declared: None,
            tensor_edits: Vec::new(),
            meta_edits: Vec::new(),
            gguf_edits: Vec::new(),
            tokenizer_edits: Vec::new(),
            manifest_edits: Vec::new(),
            after_seal: Vec::new(),
            offsets: None,
        }
    }

    /// A tiny checkpoint of the test-only family `test-fake`.
    pub fn tiny_fake() -> Self {
        let mut b = Builder::tiny();
        b.family = "test-fake".to_string();
        b.dtype = "f32".to_string();
        b
    }

    /// A tiny checkpoint of the test-only family `test-kev` (pointer head, `kev` template).
    pub fn tiny_kev() -> Self {
        let mut b = Builder::tiny();
        b.family = "test-kev".to_string();
        b.dtype = "f32".to_string();
        b
    }

    pub fn dtype(mut self, d: &str) -> Self {
        self.dtype = d.to_string();
        self
    }
    /// Make the manifest a registry one: files get an `origin`, revision is a commit hash.
    pub fn registry(mut self) -> Self {
        self.registry = true;
        self
    }
    pub fn declared(mut self, temperature: f64) -> Self {
        self.declared = Some(temperature);
        self
    }
    pub fn context_length(mut self, n: u32) -> Self {
        self.context_length = n;
        self
    }
    pub fn layers(mut self, n: u64) -> Self {
        self.n_layers = n;
        self
    }
    pub fn edit_manifest(mut self, f: impl Fn(&mut Json) + 'static) -> Self {
        self.manifest_edits.push(Box::new(f));
        self
    }
    pub fn set(self, path: &'static str, value: Json) -> Self {
        self.edit_manifest(move |j| set(j, path, value.clone()))
    }
    pub fn remove(self, path: &'static str) -> Self {
        self.edit_manifest(move |j| super::jsonedit::remove(j, path))
    }
    pub fn edit_tensors(mut self, f: impl Fn(&mut Vec<TensorDesc>) + 'static) -> Self {
        self.tensor_edits.push(Box::new(f));
        self
    }
    pub fn edit_meta(mut self, f: impl Fn(&mut Vec<(String, Meta)>) + 'static) -> Self {
        self.meta_edits.push(Box::new(f));
        self
    }
    /// Edit the GGUF bytes before hashing (so the defect is structural, not an integrity one).
    pub fn edit_gguf(mut self, f: impl Fn(&mut Vec<u8>) + 'static) -> Self {
        self.gguf_edits.push(Box::new(f));
        self
    }
    pub fn edit_tokenizer(mut self, f: impl Fn(&mut Json) + 'static) -> Self {
        self.tokenizer_edits.push(Box::new(f));
        self
    }
    /// Run `f(dir)` after the hashes are written (integrity defects).
    pub fn after_seal(mut self, f: impl Fn(&Path) + 'static) -> Self {
        self.after_seal.push(Box::new(f));
        self
    }
    /// Force the offsets written in the tensor descriptors (in tensor order).
    pub fn tensor_offsets(mut self, offsets: Vec<u64>) -> Self {
        self.offsets = Some(offsets);
        self
    }
    pub fn drop_tensor(self, name: &'static str) -> Self {
        self.edit_tensors(move |t| t.retain(|x| x.name != name))
    }
    pub fn add_tensor(self, name: &'static str, dims: Vec<u64>) -> Self {
        self.edit_tensors(move |t| {
            t.push(TensorDesc {
                name: name.to_string(),
                dims: dims.clone(),
                ty: 0,
            });
        })
    }
    pub fn set_tensor_dims(self, name: &'static str, dims: Vec<u64>) -> Self {
        self.edit_tensors(move |t| {
            for x in t.iter_mut().filter(|x| x.name == name) {
                x.dims = dims.clone();
            }
        })
    }
    pub fn set_tensor_type(self, name: &'static str, ty: u32) -> Self {
        self.edit_tensors(move |t| {
            for x in t.iter_mut().filter(|x| x.name == name) {
                x.ty = ty;
            }
        })
    }

    fn params(&self) -> ArchParams {
        let i = |x: u64| ParamValue::Int(x);
        if self.family == "test-fake" || self.family == "test-kev" {
            ArchParams::from_pairs(vec![
                ("n_embd".to_string(), i(self.n_embd)),
                ("n_vocab".to_string(), i(self.n_vocab)),
            ])
        } else {
            ArchParams::from_pairs(vec![
                ("n_layers".to_string(), i(self.n_layers)),
                ("n_embd".to_string(), i(self.n_embd)),
                ("n_ff".to_string(), i(self.n_ff)),
                ("n_heads".to_string(), i(self.n_heads)),
                ("n_kv_heads".to_string(), i(self.n_kv_heads)),
                ("n_vocab".to_string(), i(self.n_vocab)),
                ("tie_embeddings".to_string(), ParamValue::Bool(self.tie)),
            ])
        }
    }

    fn architecture_json(&self, arch: &str) -> Json {
        let mut pairs = vec![("name", s(arch))];
        if self.family == "test-fake" || self.family == "test-kev" {
            pairs.push(("n_embd", n(self.n_embd)));
            pairs.push(("n_vocab", n(self.n_vocab)));
        } else {
            pairs.push(("n_layers", n(self.n_layers)));
            pairs.push(("n_embd", n(self.n_embd)));
            pairs.push(("n_ff", n(self.n_ff)));
            pairs.push(("n_heads", n(self.n_heads)));
            pairs.push(("n_kv_heads", n(self.n_kv_heads)));
            pairs.push(("n_vocab", n(self.n_vocab)));
            pairs.push(("tie_embeddings", b(self.tie)));
        }
        obj(pairs)
    }

    fn gguf_bytes(&self) -> Vec<u8> {
        let family = families::lookup(&self.family).expect("family");
        let params = self.params();
        let matrix_ty = families::matrix_type(&self.dtype).map_or(0, |t| t.0);
        let mut tensors: Vec<TensorDesc> = (family.tensors)(&params)
            .into_iter()
            .map(|t| TensorDesc {
                name: t.name,
                dims: t.dims,
                ty: if t.role == Role::Matrix { matrix_ty } else { 0 },
            })
            .collect();
        let mut meta: Vec<(String, Meta)> = (family.metadata)(&params)
            .into_iter()
            .map(|(k, v)| {
                let m = match v {
                    families::MetaExpect::Str(x) => Meta::Str(x),
                    families::MetaExpect::UInt(x) => Meta::U32(u32::try_from(x).unwrap()),
                };
                (k, m)
            })
            .collect();
        meta.push((
            family.context_key.to_string(),
            Meta::U32(self.context_length),
        ));
        meta.push((
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArray((0..self.n_vocab).map(|i| format!("t{i}")).collect()),
        ));
        for e in &self.tensor_edits {
            e(&mut tensors);
        }
        for e in &self.meta_edits {
            e(&mut meta);
        }
        let mut bytes = write_with_offsets(&meta, &tensors, 32, self.offsets.as_deref());
        for e in &self.gguf_edits {
            e(&mut bytes);
        }
        bytes
    }

    fn tokenizer_json(&self) -> Json {
        let family = families::lookup(&self.family).expect("family");
        let mut vocab: Vec<(String, Json)> = vec![("<unk>".to_string(), n(0))];
        for (i, w) in ["A", "B", "C", "D", "TRUE", "FALSE"].iter().enumerate() {
            vocab.push(((*w).to_string(), n(3 + i as u64)));
        }
        for (i, role) in family.special_roles.iter().enumerate() {
            vocab.push((role_text(role), n(1 + i as u64)));
        }
        for i in 9..self.n_vocab {
            vocab.push((format!("w{i}"), n(i)));
        }
        let added: Vec<Json> = family
            .special_roles
            .iter()
            .enumerate()
            .map(|(i, role)| {
                obj(vec![
                    ("id", n(1 + i as u64)),
                    ("content", s(&role_text(role))),
                    ("single_word", b(false)),
                    ("lstrip", b(false)),
                    ("rstrip", b(false)),
                    ("normalized", b(false)),
                    ("special", b(true)),
                ])
            })
            .collect();
        let mut j = obj(vec![
            ("version", s("1.0")),
            ("truncation", Json::Null),
            ("padding", Json::Null),
            ("added_tokens", arr(added)),
            ("normalizer", Json::Null),
            ("pre_tokenizer", obj(vec![("type", s("Whitespace"))])),
            ("post_processor", Json::Null),
            ("decoder", Json::Null),
            (
                "model",
                obj(vec![
                    ("type", s("WordLevel")),
                    ("vocab", Json::Object(vocab)),
                    ("unk_token", s("<unk>")),
                ]),
            ),
        ]);
        for e in &self.tokenizer_edits {
            e(&mut j);
        }
        j
    }

    fn head_json(&self, kind: &str, head_bytes: &[u8]) -> Json {
        if kind == "pointer" {
            return obj(vec![
                ("kind", s("pointer")),
                ("d_model", n(self.n_embd)),
                ("proj_dim", n(4)),
                ("weights", self.file_entry("head.bin", head_bytes)),
            ]);
        }
        let target = |label: &str, id: u64| obj(vec![("label", s(label)), ("id", n(id))]);
        obj(vec![
            ("kind", s("letters")),
            ("rule", s("restricted_softmax")),
            (
                "choice_targets",
                arr(vec![
                    target("A", 3),
                    target("B", 4),
                    target("C", 5),
                    target("D", 6),
                ]),
            ),
            (
                "yes_no_targets",
                obj(vec![("no", target("FALSE", 8)), ("yes", target("TRUE", 7))]),
            ),
        ])
    }

    fn file_entry(&self, path: &str, bytes: &[u8]) -> Json {
        let mut pairs = vec![
            ("path", s(path)),
            ("size", n(bytes.len() as u64)),
            ("sha256", s(&sha256_hex(bytes))),
        ];
        if self.registry {
            pairs.push((
                "origin",
                obj(vec![("repo", s("test/tiny")), ("revision", s(COMMIT))]),
            ));
        }
        obj(pairs)
    }

    /// The self-check request used by every fixture.
    pub fn request(&self) -> Json {
        let mut questions = vec![
            (
                "team",
                obj(vec![
                    ("type", s("choice")),
                    ("instructions", s("Which team?")),
                    (
                        "criteria",
                        obj(vec![
                            ("returns", s("refunds")),
                            ("billing", Json::Null),
                            ("shipping", s("delays")),
                        ]),
                    ),
                ]),
            ),
            (
                "angry",
                obj(vec![
                    ("type", s("noul")),
                    ("instructions", s("Is the customer angry?")),
                ]),
            ),
        ];
        if self.family == "test-fake" || self.family == "test-kev" {
            questions.push((
                "urgency",
                obj(vec![
                    ("type", s("score")),
                    ("instructions", s("How urgent?")),
                    ("criteria", arr(vec![s("low"), s("mid"), s("high")])),
                ]),
            ));
        }
        obj(vec![
            ("state", obj(vec![("ticket", s("Shoes arrived late."))])),
            ("questions", obj(questions)),
        ])
    }

    fn vector(&self, temperature: f64) -> Json {
        let request = self.request();
        let ids: Vec<Json> = fake_ids(&request, self.n_vocab)
            .into_iter()
            .map(|i| n(u64::from(i)))
            .collect();
        let questions = Json::Object(
            fake_logits(&request)
                .into_iter()
                .map(|(name, logits)| {
                    let probs = fake_probabilities(&logits, temperature);
                    let q = obj(vec![
                        ("logits", arr(logits.iter().map(|x| f(*x)).collect())),
                        ("probabilities", arr(probs.iter().map(|x| f(*x)).collect())),
                    ]);
                    (name, q)
                })
                .collect(),
        );
        obj(vec![
            ("id", s("tiny-1")),
            ("request", request),
            (
                "expected",
                obj(vec![("input_ids", arr(ids)), ("questions", questions)]),
            ),
        ])
    }

    /// Write the checkpoint and return it.
    pub fn build(self) -> Fixture {
        let dir = tempfile::tempdir().expect("tempdir");
        let family = families::lookup(&self.family).expect("family");

        let gguf = self.gguf_bytes();
        let tok_json = self.tokenizer_json();
        let tok_bytes = tok_json.to_canonical_string().into_bytes();
        std::fs::write(dir.path().join("model.gguf"), &gguf).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), &tok_bytes).unwrap();
        let head_bytes = vec![0u8; 64];
        if family.head_kind == "pointer" {
            std::fs::write(dir.path().join("head.bin"), &head_bytes).unwrap();
        }

        let temperature = self.declared.unwrap_or(1.0);
        let calibration = match self.declared {
            Some(t) => obj(vec![
                ("declared", b(true)),
                ("temperature", f(t)),
                ("evidence", s("fixture")),
            ]),
            None => obj(vec![("declared", b(false))]),
        };
        let roles: Vec<(&str, Json)> = family
            .special_roles
            .iter()
            .enumerate()
            .map(|(i, r)| {
                (
                    *r,
                    obj(vec![("text", s(&role_text(r))), ("id", n(1 + i as u64))]),
                )
            })
            .collect();
        let revision = if self.registry { COMMIT } else { "v1" };
        let mut manifest = obj(vec![
            ("schema_version", n(1)),
            (
                "name",
                s(if self.registry {
                    "test/tiny"
                } else {
                    "local/tiny"
                }),
            ),
            ("revision", s(revision)),
            ("family", s(&self.family)),
            ("architecture", self.architecture_json(family.arch)),
            ("head", self.head_json(family.head_kind, &head_bytes)),
            (
                "template",
                obj(vec![
                    ("id", s(family.template_id)),
                    ("version", n(family.template_version)),
                ]),
            ),
            (
                "tokenizer",
                obj(vec![
                    ("file", self.file_entry("tokenizer.json", &tok_bytes)),
                    ("special_tokens", obj(roles)),
                ]),
            ),
            ("max_context", n(self.max_context)),
            (
                "license",
                obj(vec![
                    ("spdx", s("Apache-2.0")),
                    ("url", s("https://example.test/license")),
                ]),
            ),
            ("tasks", Json::Null),
            ("default_dtype", s(&self.dtype)),
            (
                "variants",
                arr(vec![obj(vec![
                    ("dtype", s(&self.dtype)),
                    (
                        "source",
                        obj(vec![
                            ("kind", s("gguf")),
                            ("file", self.file_entry("model.gguf", &gguf)),
                        ]),
                    ),
                    ("calibration", calibration),
                    (
                        "selfcheck",
                        obj(vec![("vectors", arr(vec![self.vector(temperature)]))]),
                    ),
                ])]),
            ),
        ]);
        for e in &self.manifest_edits {
            e(&mut manifest);
        }
        for e in &self.after_seal {
            e(dir.path());
        }
        std::fs::write(
            dir.path().join(MANIFEST_FILE),
            manifest.to_canonical_string(),
        )
        .unwrap();
        Fixture {
            dir,
            manifest,
            n_vocab: self.n_vocab,
        }
    }
}
