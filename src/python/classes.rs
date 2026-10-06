//! Python classes and functions of `archai_jev._core`: thin wrappers around the core.

use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyTuple};

use super::convert::{entry_value, state_value};
use crate::answers::Answer;
use crate::error::Error;
use crate::mock::{Fault, MockScorer};
use crate::schema::{Choice, ChoiceOption, Question, Questions, Score, State, TextEntry, YesNo};
use crate::scorer::{Calibration, Scorer};

type Obj<'py> = Bound<'py, PyAny>;

/// A validated question. Opaque: the public classes of `archai_jev` wrap it.
#[pyclass(frozen, name = "_Question", module = "archai_jev._core")]
pub struct PyQuestion {
    inner: Question,
}

fn entry(obj: &Obj<'_>, field: &str) -> Result<TextEntry, Error> {
    TextEntry::new(field, entry_value(obj, field)?)
}

fn type_name(obj: &Obj<'_>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

fn bad_criteria(message: String) -> Error {
    Error::EntryConversion { message }
}

fn choice_options(criteria: &Obj<'_>) -> Result<Vec<ChoiceOption>, Error> {
    if let Ok(d) = criteria.cast::<PyDict>() {
        let mut out = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key = k.cast::<PyString>().map_err(|_| {
                bad_criteria(format!(
                    "criteria keys of choice must be str, got {}",
                    type_name(&k)
                ))
            })?;
            let key = key
                .to_str()
                .map_err(|_| bad_criteria("criteria key contains a lone surrogate".to_string()))?
                .to_string();
            let description = if v.is_none() {
                None
            } else {
                Some(entry(&v, &format!("description of option \"{key}\""))?)
            };
            out.push(ChoiceOption { key, description });
        }
        return Ok(out);
    }
    if criteria.is_instance_of::<PyString>() {
        return Err(bad_criteria(
            "criteria of choice must be a dict or a list of keys, got str; a str would be read one character at a time".to_string(),
        ));
    }
    let items: Option<Vec<Obj<'_>>> = if let Ok(l) = criteria.cast::<PyList>() {
        Some(l.iter().collect())
    } else if let Ok(t) = criteria.cast::<PyTuple>() {
        Some(t.iter().collect())
    } else {
        None
    };
    match items {
        Some(items) => items
            .iter()
            .map(|k| {
                let key = k.cast::<PyString>().map_err(|_| {
                    bad_criteria(format!(
                        "criteria keys of choice must be str, got {}",
                        type_name(k)
                    ))
                })?;
                let key = key.to_str().map_err(|_| {
                    bad_criteria("criteria key contains a lone surrogate".to_string())
                })?;
                Ok(ChoiceOption::key(key))
            })
            .collect(),
        None => Err(bad_criteria(format!(
            "criteria of choice must be a dict or a list of keys, got {}",
            type_name(criteria)
        ))),
    }
}

fn score_levels(criteria: &Obj<'_>) -> Result<Vec<TextEntry>, Error> {
    if criteria.is_instance_of::<PyString>() {
        return Err(bad_criteria(
            "criteria of score must be a list or tuple of levels, got str; a str would be read one character at a time".to_string(),
        ));
    }
    let items: Vec<Obj<'_>> = if let Ok(l) = criteria.cast::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = criteria.cast::<PyTuple>() {
        t.iter().collect()
    } else {
        return Err(bad_criteria(format!(
            "criteria of score must be a list or tuple of levels, got {}",
            type_name(criteria)
        )));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, level)| entry(level, &format!("level {i} of score")))
        .collect()
}

fn yes_no_descriptions(
    criteria: &Obj<'_>,
) -> Result<(Option<TextEntry>, Option<TextEntry>), Error> {
    if criteria.is_none() {
        return Ok((None, None));
    }
    let d = criteria.cast::<PyDict>().map_err(|_| {
        bad_criteria(format!(
            "criteria of yes/no must be a dict with the keys \"true\" and \"false\", or None, got {}",
            type_name(criteria)
        ))
    })?;
    let (mut yes, mut no) = (None, None);
    for (k, v) in d.iter() {
        let key = k
            .cast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok().map(str::to_string))
            .ok_or_else(|| {
                bad_criteria(format!(
                    "YesNo criteria keys must be str, got {}",
                    type_name(&k)
                ))
            })?;
        let slot = match key.as_str() {
            "true" => &mut yes,
            "false" => &mut no,
            other => {
                return Err(bad_criteria(format!(
                    "YesNo criteria only accepts the keys \"true\" and \"false\", got \"{other}\"; check the spelling"
                )));
            }
        };
        if !v.is_none() {
            *slot = Some(entry(&v, &format!("description of the \"{key}\" answer"))?);
        }
    }
    Ok((yes, no))
}

/// Build and validate a question (`kind` is "choice", "score" or "yes_no").
#[pyfunction]
pub fn make_question(
    kind: &str,
    instructions: &Obj<'_>,
    criteria: &Obj<'_>,
) -> PyResult<PyQuestion> {
    let instructions = entry(instructions, "instructions")?;
    let inner: Question = match kind {
        "choice" => Choice::new(instructions, choice_options(criteria)?)?.into(),
        "score" => Score::new(instructions, score_levels(criteria)?)?.into(),
        "yes_no" => {
            let (yes, no) = yes_no_descriptions(criteria)?;
            YesNo::new(instructions, yes, no)?.into()
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown question kind {other:?}"
            )));
        }
    };
    Ok(PyQuestion { inner })
}

/// The canonical text of a state (what the model will read).
#[pyfunction]
pub fn render_state(state: &Obj<'_>) -> PyResult<String> {
    Ok(State::new(state_value(state)?)?
        .canonical_text()
        .to_string())
}

/// Temperature-scaled softmax of raw logits.
#[pyfunction]
#[pyo3(signature = (logits, temperature = 1.0))]
pub fn calibrated_softmax(logits: Vec<f64>, temperature: f64) -> PyResult<Vec<f64>> {
    Ok(crate::calibration::calibrated_softmax(
        &logits,
        temperature,
    )?)
}

/// A deliberate failure injected into a `MockScorer`.
#[pyclass(frozen, name = "Fault", module = "archai_jev._core")]
pub struct PyFault {
    inner: Fault,
}

#[pymethods]
impl PyFault {
    /// NaN at (`question`, `option`).
    #[staticmethod]
    fn nan(question: &str, option: usize) -> Self {
        PyFault {
            inner: Fault::nan(question, option),
        }
    }
    /// +infinity at (`question`, `option`).
    #[staticmethod]
    fn pos_inf(question: &str, option: usize) -> Self {
        PyFault {
            inner: Fault::pos_inf(question, option),
        }
    }
    /// -infinity at (`question`, `option`).
    #[staticmethod]
    fn neg_inf(question: &str, option: usize) -> Self {
        PyFault {
            inner: Fault::neg_inf(question, option),
        }
    }
    /// `count` logits for `question` instead of one per option.
    #[staticmethod]
    fn wrong_option_count(question: &str, count: usize) -> Self {
        PyFault {
            inner: Fault::wrong_option_count(question, count),
        }
    }
    /// Logits for `count` questions instead of one per question.
    #[staticmethod]
    fn wrong_question_count(count: usize) -> Self {
        PyFault {
            inner: Fault::wrong_question_count(count),
        }
    }
}

fn check_latency(latency: f64) -> PyResult<Duration> {
    if !latency.is_finite() || latency < 0.0 {
        return Err(PyValueError::new_err(format!(
            "latency must be a finite number >= 0 seconds, got {latency}"
        )));
    }
    Ok(Duration::from_secs_f64(latency))
}

/// The deterministic fake scorer. It is not a model.
#[pyclass(frozen, name = "MockScorer", module = "archai_jev._core")]
pub struct PyMockScorer {
    pub(crate) inner: Arc<MockScorer>,
}

impl PyMockScorer {
    fn configured(
        base: MockScorer,
        temperature: f64,
        calibrated: bool,
        latency: f64,
    ) -> PyResult<Self> {
        let scorer = base
            .with_calibration(Calibration::new(temperature, calibrated)?)
            .with_latency(check_latency(latency)?);
        Ok(PyMockScorer {
            inner: Arc::new(scorer),
        })
    }
}

#[pymethods]
impl PyMockScorer {
    #[new]
    #[pyo3(signature = (seed = None, *, temperature = 1.0, calibrated = false, latency = 0.0))]
    fn new(
        seed: Option<&Obj<'_>>,
        temperature: f64,
        calibrated: bool,
        latency: f64,
    ) -> PyResult<Self> {
        let seed = match seed {
            None => 0,
            Some(s) => {
                if !s.is_instance_of::<PyInt>() || s.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err(format!(
                        "seed must be an int, got {}",
                        type_name(s)
                    )));
                }
                s.extract::<u64>().map_err(|_| {
                    PyValueError::new_err("seed must be in the range 0 <= seed < 2**64")
                })?
            }
        };
        Self::configured(MockScorer::new(seed), temperature, calibrated, latency)
    }

    /// A scorer that returns exactly these logits, whatever they are.
    #[staticmethod]
    #[pyo3(signature = (logits, *, temperature = 1.0, calibrated = false, latency = 0.0))]
    fn scripted(
        logits: Vec<Vec<f64>>,
        temperature: f64,
        calibrated: bool,
        latency: f64,
    ) -> PyResult<Self> {
        Self::configured(
            MockScorer::scripted(logits),
            temperature,
            calibrated,
            latency,
        )
    }

    /// A new scorer with one more injected fault; the original is unchanged.
    fn with_fault(&self, fault: &PyFault) -> Self {
        PyMockScorer {
            inner: Arc::new(self.inner.with_fault(fault.inner.clone())),
        }
    }

    /// How many times this scorer has been called.
    #[getter]
    fn calls(&self) -> u64 {
        self.inner.calls()
    }

    /// The temperature this scorer declares.
    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.calibration().temperature()
    }

    /// Whether this scorer declares a calibration.
    #[getter]
    fn calibrated(&self) -> bool {
        self.inner.calibration().calibrated()
    }
}

/// A model ready to answer: wraps a scorer. Python's `Jev` holds one.
#[pyclass(frozen, name = "_Model", module = "archai_jev._core")]
pub struct PyModel {
    scorer: Arc<dyn Scorer>,
}

impl PyModel {
    /// Wrap a loaded model's scorer.
    pub(crate) fn wrap(scorer: Arc<dyn Scorer>) -> Self {
        PyModel { scorer }
    }
}

fn build_questions(questions: &[(String, Py<PyQuestion>)]) -> Result<Questions, Error> {
    Questions::new(
        questions
            .iter()
            .map(|(n, q)| (n.clone(), q.get().inner.clone()))
            .collect::<Vec<_>>(),
    )
}

fn answer_to_python<'py>(py: Python<'py>, name: &str, a: &Answer) -> PyResult<Obj<'py>> {
    Ok(match a {
        Answer::Choice(c) => (
            "choice",
            name,
            c.value().to_string(),
            c.index(),
            c.probabilities().to_vec(),
            c.confidence(),
            c.calibrated(),
        )
            .into_pyobject(py)?
            .into_any(),
        Answer::Score(s) => (
            "score",
            name,
            s.value(),
            s.probabilities().to_vec(),
            s.confidence(),
            s.calibrated(),
        )
            .into_pyobject(py)?
            .into_any(),
        Answer::YesNo(y) => ("yes_no", name, y.probability(), y.calibrated())
            .into_pyobject(py)?
            .into_any(),
    })
}

#[pymethods]
impl PyModel {
    /// A model backed by the mock scorer.
    #[staticmethod]
    fn from_mock(scorer: &PyMockScorer) -> Self {
        PyModel {
            scorer: scorer.inner.clone(),
        }
    }

    /// Ask every question about `state`; returns one tuple per answer, in question order.
    fn ask<'py>(
        &self,
        py: Python<'py>,
        state: &Obj<'py>,
        questions: Vec<(String, Py<PyQuestion>)>,
    ) -> PyResult<Bound<'py, PyList>> {
        let state = State::new(state_value(state)?)?;
        let questions = build_questions(&questions)?;
        let scorer = Arc::clone(&self.scorer);
        let answers = py.detach(|| crate::ask::ask(scorer.as_ref(), &state, &questions))?;
        let out = PyList::empty(py);
        for (name, a) in answers.iter() {
            out.append(answer_to_python(py, name, a)?)?;
        }
        Ok(out)
    }

    /// Same questions on several states; all states are validated before any calculation.
    fn ask_many<'py>(
        &self,
        py: Python<'py>,
        states: Vec<Obj<'py>>,
        questions: Vec<(String, Py<PyQuestion>)>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut parsed = Vec::with_capacity(states.len());
        for (i, s) in states.iter().enumerate() {
            let state = state_value(s)
                .and_then(State::new)
                .map_err(|e| prefix(e, i))?;
            parsed.push(state);
        }
        let questions = build_questions(&questions)?;
        let scorer = Arc::clone(&self.scorer);
        let all = py.detach(|| {
            parsed
                .iter()
                .map(|s| crate::ask::ask(scorer.as_ref(), s, &questions))
                .collect::<Result<Vec<_>, Error>>()
        })?;
        let out = PyList::empty(py);
        for answers in &all {
            let one = PyList::empty(py);
            for (name, a) in answers.iter() {
                one.append(answer_to_python(py, name, a)?)?;
            }
            out.append(one)?;
        }
        Ok(out)
    }
}

/// Prefix a state error of `ask_many` with the index of the state (`states[i]: ...`).
fn prefix(err: Error, index: usize) -> Error {
    let message = format!("states[{index}]: {err}");
    match err {
        Error::StateConversion { .. } | Error::StateType { .. } => {
            Error::StateConversion { message }
        }
        Error::StateTooDeep { .. }
        | Error::StateTooLarge
        | Error::NonFiniteNumber { .. }
        | Error::DuplicateStateKey { .. } => Error::StateConversion { message },
        other => other,
    }
}
