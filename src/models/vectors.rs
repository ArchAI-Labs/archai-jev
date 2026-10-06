//! Self-check vectors of a manifest: a request and the numbers the model must reproduce.

use super::incompat::Incompat;
use crate::json_strict::{Json, Obj};

/// Most vectors a variant may carry.
pub const MAX_VECTORS: usize = 256;
/// Most prompt token ids across all the vectors of a variant.
pub const MAX_TOTAL_IDS: usize = 1_000_000;

/// Expected numbers for one question of a vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Expected {
    /// Raw logits after the head, before temperature.
    pub logits: Vec<f64>,
    /// Probabilities at the manifest's temperature.
    pub probabilities: Vec<f64>,
}

/// One self-check vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// Name of the vector (for error messages).
    pub id: String,
    /// The request, in the TypeSafe JSON shape (`state` and `questions`), kept opaque here.
    pub request: Json,
    /// Token ids of the prompt the model must build, exactly.
    pub input_ids: Vec<u32>,
    /// Expected numbers per question name, in request order.
    pub questions: Vec<(String, Expected)>,
}

impl Vector {
    /// Kinds (`"choice"`, `"score"`, `"noul"`) of the questions in the request.
    pub fn question_types(&self) -> Vec<String> {
        match self.request.get("questions") {
            Some(Json::Object(qs)) => qs
                .iter()
                .filter_map(|(_, q)| match q.get("type") {
                    Some(Json::Str(t)) => Some(t.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn numbers(path: &str, items: &[Json]) -> Result<Vec<f64>, Incompat> {
    items
        .iter()
        .enumerate()
        .map(|(i, j)| match j {
            Json::Int(n) => Ok(*n as f64),
            Json::Float(f) => Ok(*f),
            other => Err(Incompat::ManifestBadValue {
                path: format!("{path}[{i}]"),
                detail: format!("must be a number, got {}", other.kind()),
            }),
        })
        .collect()
}

/// Read `selfcheck.vectors` from the array `items`, called `path` in messages.
pub(crate) fn parse_vectors(items: &[Json], path: &str) -> Result<Vec<Vector>, Incompat> {
    if items.len() > MAX_VECTORS {
        return Err(Incompat::ManifestLimit {
            detail: format!(
                "{path} has {} vectors; the maximum is {MAX_VECTORS}",
                items.len()
            ),
        });
    }
    let mut total_ids = 0usize;
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let vp = format!("{path}[{i}]");
        let mut o = Obj::new(item, &vp)?;
        let id = o.str("id")?.to_string();
        if id.is_empty() {
            return Err(Incompat::ManifestBadValue {
                path: format!("{vp}.id"),
                detail: "must not be empty".to_string(),
            });
        }
        let request = o.req("request")?.clone();
        if !matches!(request, Json::Object(_)) {
            return Err(Incompat::BadVector {
                id,
                detail: "request must be an object with state and questions".to_string(),
            });
        }
        let mut e = o.obj("expected")?;
        let ids_json = e.arr("input_ids")?;
        let mut input_ids = Vec::with_capacity(ids_json.len());
        for (k, j) in ids_json.iter().enumerate() {
            match j {
                Json::Int(n) if (0..=i128::from(u32::MAX)).contains(n) => {
                    input_ids.push(u32::try_from(*n).unwrap_or(0));
                }
                _ => {
                    return Err(Incompat::BadVector {
                        id,
                        detail: format!("input_ids[{k}] must be a token id (an integer)"),
                    });
                }
            }
        }
        total_ids += input_ids.len();
        if total_ids > MAX_TOTAL_IDS {
            return Err(Incompat::ManifestLimit {
                detail: format!("{path} has more than {MAX_TOTAL_IDS} prompt token ids in all"),
            });
        }
        let qjson = e.req("questions")?;
        let Json::Object(qpairs) = qjson else {
            return Err(Incompat::BadVector {
                id,
                detail: "expected.questions must be an object".to_string(),
            });
        };
        let mut questions = Vec::with_capacity(qpairs.len());
        for (name, q) in qpairs {
            let qp = format!("{vp}.expected.questions.{name}");
            let mut qo = Obj::new(q, &qp)?;
            let logits = numbers(&format!("{qp}.logits"), qo.arr("logits")?)?;
            let probabilities = numbers(&format!("{qp}.probabilities"), qo.arr("probabilities")?)?;
            qo.finish()?;
            questions.push((
                name.clone(),
                Expected {
                    logits,
                    probabilities,
                },
            ));
        }
        e.finish()?;
        o.finish()?;
        out.push(Vector {
            id,
            request,
            input_ids,
            questions,
        });
    }
    Ok(out)
}

/// Sanity checks that need no model: well-formed distributions, plausible ids.
pub(crate) fn check_vectors(vectors: &[Vector], n_vocab: u64) -> Result<(), Incompat> {
    for v in vectors {
        let bad = |detail: String| Incompat::BadVector {
            id: v.id.clone(),
            detail,
        };
        if v.input_ids.is_empty() {
            return Err(bad("input_ids is empty".to_string()));
        }
        if let Some(id) = v.input_ids.iter().find(|id| u64::from(**id) >= n_vocab) {
            return Err(bad(format!(
                "input id {id} is outside the vocabulary (size {n_vocab})"
            )));
        }
        if v.questions.is_empty() {
            return Err(bad("expected.questions is empty".to_string()));
        }
        for (name, q) in &v.questions {
            if q.logits.is_empty() || q.logits.len() != q.probabilities.len() {
                return Err(bad(format!(
                    "question '{name}' needs the same, non-zero number of logits and probabilities"
                )));
            }
            if q.logits
                .iter()
                .chain(&q.probabilities)
                .any(|x| !x.is_finite())
            {
                return Err(bad(format!("question '{name}' has a non-finite number")));
            }
            if q.probabilities.iter().any(|p| !(0.0..=1.0).contains(p)) {
                return Err(bad(format!(
                    "question '{name}' has a probability outside [0, 1]"
                )));
            }
            let sum: f64 = q.probabilities.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(bad(format!(
                    "the probabilities of question '{name}' sum to {sum}, expected 1 within 1e-6"
                )));
            }
        }
    }
    Ok(())
}
