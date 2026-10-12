//! Binding of the evaluation feature (spec 017): the opaque `_Dataset`, the conversion of
//! Python records, and the three model operations. Thin: it converts, validates scalars, releases
//! the GIL and maps errors. The classes people use are in `python/archai_jev/_evaluation.py`.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyString, PyTuple};

use super::calibration::{ReportTuple, parse_bins, report_tuple};
use super::classes::{
    PyCancelToken, PyControl, PyModel, PyQuestion, build_questions, refuse_reentrant_call,
};
use super::convert::state_value;
use super::rows::{describe, is_integer_like};
use crate::calibration::fit::check_bounds;
use crate::error::Error;
use crate::evaluation::calibrate::{StudyOptions, calibrate};
use crate::evaluation::dataset::{Dataset, LabelledRecord, RawLabel, check_id, not_an_id};
use crate::evaluation::evaluate::{EvaluationData, Options, evaluate};
use crate::evaluation::fingerprint::{hex, kind_name};
use crate::evaluation::groups::split_groups;
use crate::evaluation::problem::{ArgumentProblem, DatasetProblem, MAX_BOOTSTRAP_SAMPLES};
use crate::evaluation::read::{read_jsonl, record_from_json};
use crate::evaluation::recalibrated::Recalibrated;
use crate::evaluation::scopes::ScopeReport;
use crate::json_strict::Json;
use crate::schema::{QuestionKind, Questions, State, StateValue};
use crate::scorer::Calibration;

type Obj<'py> = Bound<'py, PyAny>;

/// `(state, [(name, _Question)], [(name, label)], id, group)`.
type RecordParts<'py> = (
    Obj<'py>,
    Vec<(String, Py<PyQuestion>)>,
    Vec<(String, Obj<'py>)>,
    Obj<'py>,
    Obj<'py>,
);

/// A validated dataset. Opaque: the Python layer holds one to avoid validating twice.
#[pyclass(frozen, name = "_Dataset", module = "archai_jev._core")]
pub struct PyDataset {
    pub(crate) inner: Arc<Dataset>,
}

#[pymethods]
impl PyDataset {
    /// Number of records.
    fn __len__(&self) -> usize {
        self.inner.records().len()
    }

    /// The fingerprint of the dataset, 16 hexadecimal digits.
    #[getter]
    fn fingerprint(&self) -> String {
        hex(self.inner.fingerprint())
    }

    /// Number of questions over all records.
    #[getter]
    fn n_questions(&self) -> usize {
        self.inner.question_count()
    }

    /// Number of groups.
    #[getter]
    fn n_groups(&self) -> usize {
        self.inner.groups()
    }
}

// ----------------------------------------------------------------------- scalars

fn int_argument(obj: &Obj<'_>) -> Option<i128> {
    if obj.is_instance_of::<PyBool>() || !is_integer_like(obj).unwrap_or(false) {
        return None;
    }
    obj.extract::<i128>().ok()
}

fn parse_seed(obj: &Obj<'_>) -> PyResult<u64> {
    int_argument(obj)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| Error::EvalArgument(ArgumentProblem::Seed { got: describe(obj) }).into())
}

fn parse_bootstrap_samples(obj: &Obj<'_>) -> PyResult<usize> {
    int_argument(obj)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n <= MAX_BOOTSTRAP_SAMPLES)
        .ok_or_else(|| {
            Error::EvalArgument(ArgumentProblem::BootstrapSamples { got: describe(obj) }).into()
        })
}

fn parse_folds(obj: &Obj<'_>) -> PyResult<usize> {
    int_argument(obj)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| (2..=crate::evaluation::groups::MAX_FOLDS).contains(n))
        .ok_or_else(|| Error::EvalArgument(ArgumentProblem::Folds { got: describe(obj) }).into())
}

fn kind_of(name: &str) -> Option<QuestionKind> {
    match name {
        "choice" => Some(QuestionKind::Choice),
        "score" => Some(QuestionKind::Score),
        "yes_no" => Some(QuestionKind::YesNo),
        _ => None,
    }
}

// ----------------------------------------------------------------------- records

fn raw_label(obj: &Obj<'_>) -> RawLabel {
    if obj.is_instance_of::<PyBool>() {
        return RawLabel::Bool(obj.extract::<bool>().unwrap_or(false));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return match s.to_str() {
            Ok(text) => RawLabel::Str(text.to_string()),
            Err(_) => RawLabel::Other("a string with a lone surrogate".to_string()),
        };
    }
    if obj.is_instance_of::<PyInt>() {
        return match obj.extract::<i128>() {
            Ok(n) => RawLabel::Int(n),
            Err(_) => RawLabel::Other(describe(obj)),
        };
    }
    RawLabel::Other(describe(obj))
}

fn id_of(place: &str, field: &'static str, obj: Option<&Obj<'_>>) -> Result<Option<String>, Error> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => match o.cast::<PyString>() {
            Ok(s) => match s.to_str() {
                Ok(text) => check_id(place, field, Some(text)),
                Err(_) => Err(not_an_id(place, field)),
            },
            Err(_) => Err(not_an_id(place, field)),
        },
    }
}

/// A `StateValue` as the JSON the file reader would have produced.
fn json_of(value: &StateValue, place: &str) -> Result<Json, Error> {
    let not_json = |detail: &str| {
        Error::Dataset(DatasetProblem::NotJson {
            place: place.to_string(),
            detail: detail.to_string(),
        })
    };
    Ok(match value {
        StateValue::Null => Json::Null,
        StateValue::Bool(b) => Json::Bool(*b),
        StateValue::Int(text) => text
            .parse::<i128>()
            .map(Json::Int)
            .unwrap_or_else(|_| Json::BigInt(text.clone())),
        StateValue::Float(f) if f.is_finite() => Json::Float(*f),
        StateValue::Float(_) => return Err(not_json("NaN and infinity are not allowed")),
        StateValue::Str(s) => Json::Str(s.clone()),
        StateValue::List(items) => Json::Array(
            items
                .iter()
                .map(|v| json_of(v, place))
                .collect::<Result<_, _>>()?,
        ),
        StateValue::Object(pairs) => Json::Object(
            pairs
                .iter()
                .map(|(k, v)| Ok((k.clone(), json_of(v, place)?)))
                .collect::<Result<_, Error>>()?,
        ),
    })
}

/// A record given as a Python `dict`: read with the rules of the file.
fn record_from_dict(obj: &Obj<'_>, place: &str) -> Result<LabelledRecord, Error> {
    if !obj.is_instance_of::<PyDict>() {
        return Err(Error::Dataset(DatasetProblem::NotAnObject {
            place: place.to_string(),
            got: obj
                .get_type()
                .name()
                .map_or_else(|_| "an object".to_string(), |n| format!("a {n}")),
        }));
    }
    let value = state_value(obj).map_err(|e| Error::at(place, e))?;
    let json = json_of(&value, place)?;
    record_from_json(&json, place)
}

/// A record given as the tuple the Python layer makes of a `LabelledRecord`:
/// `(state, [(name, _Question)], [(name, label)], id, group)`.
fn record_from_tuple(t: &Bound<'_, PyTuple>, place: &str) -> PyResult<LabelledRecord> {
    let (state, questions, labels, id, group): RecordParts<'_> = t.extract()?;
    let state = state_value(&state)
        .and_then(State::new)
        .map_err(|e| Error::at(place, e))?;
    let questions: Questions = build_questions(&questions).map_err(|e| Error::at(place, e))?;
    let raw: Vec<(String, RawLabel)> = labels
        .iter()
        .map(|(n, l)| (n.clone(), raw_label(l)))
        .collect();
    let id = id_of(place, "id", Some(&id))?;
    let group = id_of(place, "group", Some(&group))?;
    Ok(LabelledRecord::build(
        place, state, questions, &raw, id, group,
    )?)
}

fn dataset_of(items: &[Obj<'_>]) -> PyResult<PyDataset> {
    let mut records = Vec::with_capacity(items.len());
    for (n, item) in items.iter().enumerate() {
        let place = format!("record {n}");
        let record = match item.cast::<PyTuple>() {
            Ok(t) => record_from_tuple(t, &place)?,
            Err(_) => record_from_dict(item, &place)?,
        };
        records.push(record);
    }
    Ok(PyDataset {
        inner: Arc::new(Dataset::new(records)?),
    })
}

/// Validate and convert the records of a dataset.
#[pyfunction]
#[pyo3(name = "_dataset_from_items")]
pub fn dataset_from_items(py: Python<'_>, items: Vec<Obj<'_>>) -> PyResult<PyDataset> {
    let _ = py;
    dataset_of(&items)
}

/// Validate the bytes of a JSONL file (all of it, or an error); `name` is what the messages show.
#[pyfunction]
#[pyo3(name = "_dataset_from_jsonl")]
pub fn dataset_from_jsonl(py: Python<'_>, data: &[u8], name: &str) -> PyResult<PyDataset> {
    let inner = py.detach(|| read_jsonl(data, name))?;
    Ok(PyDataset {
        inner: Arc::new(inner),
    })
}

/// Check the options of `evaluate` (so `compare` can refuse them before it asks any model).
#[pyfunction]
#[pyo3(name = "_check_evaluate_options")]
pub fn check_evaluate_options(
    n_bins: &Obj<'_>,
    bootstrap_samples: &Obj<'_>,
    seed: &Obj<'_>,
) -> PyResult<()> {
    parse_bins(n_bins)?;
    parse_bootstrap_samples(bootstrap_samples)?;
    parse_seed(seed)?;
    Ok(())
}

/// The indices of the records of the train and test sides of a split by group.
#[pyfunction]
#[pyo3(name = "_split")]
pub fn split(
    dataset: &PyDataset,
    test_fraction: f64,
    seed: &Obj<'_>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let seed = parse_seed(seed)?;
    let sizes = dataset.inner.group_sizes();
    let is_test = split_groups(&sizes, test_fraction, seed)?;
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, g) in dataset.inner.group_of().iter().enumerate() {
        if is_test.get(*g).copied().unwrap_or(false) {
            test.push(i);
        } else {
            train.push(i);
        }
    }
    Ok((train, test))
}

// ----------------------------------------------------------------------- reports

type ScopeTuple = (
    ReportTuple,
    Vec<(&'static str, f64)>,
    Option<Vec<(f64, f64)>>,
);

fn scope_tuple(s: &ScopeReport) -> ScopeTuple {
    (
        report_tuple(&s.report),
        s.extras.pairs(),
        s.ci95.map(|c| c.to_vec()),
    )
}

type EvaluationTuple = (
    (usize, usize),
    ScopeTuple,
    Vec<(&'static str, ScopeTuple)>,
    Vec<(String, &'static str, ScopeTuple)>,
    (String, Option<usize>, Vec<String>),
    (f64, bool),
);

fn evaluation_tuple(d: &EvaluationData) -> EvaluationTuple {
    (
        (d.n_records, d.n_questions),
        scope_tuple(&d.overall),
        d.by_type
            .iter()
            .map(|(k, s)| (kind_name(*k), scope_tuple(s)))
            .collect(),
        d.by_question
            .iter()
            .map(|(n, k, s)| (n.clone(), kind_name(*k), scope_tuple(s)))
            .collect(),
        (
            hex(d.dataset_fingerprint),
            d.calibration_overlap,
            d.notes.clone(),
        ),
        (d.calibration.temperature(), d.calibration.calibrated()),
    )
}

type StudyTuple = (
    (f64, (f64, f64, f64, usize, usize), Vec<f64>),
    (
        ReportTuple,
        ReportTuple,
        ReportTuple,
        Vec<(&'static str, ReportTuple, ReportTuple)>,
    ),
    (usize, usize, String, Vec<u64>),
    (Vec<String>, (f64, bool)),
);

// ----------------------------------------------------------------------- model operations

impl PyModel {
    /// Evaluate on a dataset (spec 017 §2).
    pub(crate) fn evaluate_dataset<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
        n_bins: &Obj<'py>,
        bootstrap_samples: &Obj<'py>,
        seed: &Obj<'py>,
        cancel: Option<Py<PyCancelToken>>,
    ) -> PyResult<EvaluationTuple> {
        refuse_reentrant_call()?;
        let options = Options {
            n_bins: parse_bins(n_bins)?,
            bootstrap_samples: parse_bootstrap_samples(bootstrap_samples)?,
            seed: parse_seed(seed)?,
        };
        let scorer = Arc::clone(&self.scorer);
        let provenance = self.provenance.clone();
        let data = Arc::clone(&dataset.inner);
        let control = PyControl::new(py, cancel.as_ref().map(|t| t.get().flag()))?;
        let result = py.detach(|| {
            evaluate(
                scorer.as_ref(),
                &data,
                &options,
                &control,
                provenance.as_deref(),
            )
        });
        let data = control.finish(py, result)?;
        Ok(evaluation_tuple(&data))
    }

    /// Study the calibration on a dataset (spec 017 §3.1).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn calibrate_dataset<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
        folds: &Obj<'py>,
        seed: &Obj<'py>,
        question_types: Option<Vec<String>>,
        bounds: (f64, f64),
        cancel: Option<Py<PyCancelToken>>,
    ) -> PyResult<StudyTuple> {
        refuse_reentrant_call()?;
        let types = match question_types {
            None => None,
            Some(names) => Some(
                names
                    .iter()
                    .map(|n| {
                        kind_of(n).ok_or_else(|| {
                            Error::EvalArgument(ArgumentProblem::QuestionType {
                                got: format!("'{n}'"),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        check_bounds(bounds)?;
        let options = StudyOptions {
            folds: parse_folds(folds)?,
            seed: parse_seed(seed)?,
            question_types: types,
            bounds,
        };
        let scorer = Arc::clone(&self.scorer);
        let data = Arc::clone(&dataset.inner);
        let control = PyControl::new(py, cancel.as_ref().map(|t| t.get().flag()))?;
        let result = py.detach(|| calibrate(scorer.as_ref(), &data, &options, &control));
        let s = control.finish(py, result)?;
        Ok((
            (
                s.temperature,
                (
                    s.fit.temperature,
                    s.fit.loss_before,
                    s.fit.loss_after,
                    s.fit.n,
                    s.fit.iterations,
                ),
                s.fold_temperatures.clone(),
            ),
            (
                report_tuple(&s.current),
                report_tuple(&s.in_sample),
                report_tuple(&s.out_of_fold),
                s.by_type
                    .iter()
                    .map(|(k, c, o)| (kind_name(*k), report_tuple(c), report_tuple(o)))
                    .collect(),
            ),
            (
                s.n,
                s.n_groups,
                hex(s.dataset_fingerprint),
                s.record_fingerprints.clone(),
            ),
            (
                s.notes.clone(),
                (s.calibration.temperature(), s.calibration.calibrated()),
            ),
        ))
    }

    /// The same model with another temperature (a calibration given by the user).
    pub(crate) fn with_temperature_value(
        &self,
        temperature: f64,
        provenance: Option<Vec<u64>>,
    ) -> PyResult<PyModel> {
        let calibration = Calibration::new(temperature, true)?;
        Ok(PyModel {
            scorer: Arc::new(Recalibrated::new(Arc::clone(&self.base), calibration)),
            base: Arc::clone(&self.base),
            provenance: provenance.map(Arc::from),
        })
    }
}
