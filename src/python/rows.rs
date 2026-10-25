//! Python sequences of numbers → [`Rows`] and labels, once per call (spec 004 §3.5).
//!
//! The accepted elements are `int` and `float` (subclasses and numpy scalars included). `bool`,
//! `str`, `bytes` and `None` are `TypeError`s with the message of the spec: a `True` that slips in
//! as `1.0` is the kind of mistake that gives plausible but wrong numbers. Any other object is
//! accepted when its type has `__float__` or `__index__`; if that method raises, the exception is
//! the user's and propagates unchanged. Rows may have different lengths.
//!
//! Exact `list` and `tuple` containers are read natively; anything else (numpy arrays, other
//! sequences, generators) is iterated, without a fast lane.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::calibration::rows::{Rows, RowsBuilder};

/// What a value was, for a message: exact builtin values by `repr` (no user code runs), anything
/// else by its type.
pub fn describe(obj: &Bound<'_, PyAny>) -> String {
    let exact = obj.is_exact_instance_of::<PyInt>()
        || obj.is_exact_instance_of::<PyBool>()
        || obj.is_exact_instance_of::<PyFloat>()
        || obj.is_exact_instance_of::<PyString>()
        || obj.is_none();
    match (exact, obj.repr()) {
        (true, Ok(r)) => r.to_string(),
        _ => format!("<{} object>", type_name(obj)),
    }
}

/// An integer for the purposes of labels and counts: an `int` (subclasses too) or any object with
/// `__index__` (numpy integers). `bool` and `float` are not.
pub fn is_integer_like(item: &Bound<'_, PyAny>) -> PyResult<bool> {
    if item.is_instance_of::<PyBool>() || item.is_instance_of::<PyFloat>() {
        return Ok(false);
    }
    Ok(item.is_instance_of::<PyInt>() || item.get_type().hasattr("__index__")?)
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

/// Calls `f` with every item of `obj`. `not_iterable` builds the error when `obj` has no `iter`.
fn visit<'py>(
    obj: &Bound<'py, PyAny>,
    not_iterable: impl FnOnce() -> PyErr,
    mut f: impl FnMut(Bound<'py, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    if let Ok(list) = obj.cast_exact::<PyList>() {
        for item in list.iter() {
            f(item)?;
        }
    } else if let Ok(tuple) = obj.cast_exact::<PyTuple>() {
        for item in tuple.iter() {
            f(item)?;
        }
    } else {
        let iterator = obj.try_iter().map_err(|e| {
            if e.is_instance_of::<PyTypeError>(obj.py()) {
                not_iterable()
            } else {
                e
            }
        })?;
        for item in iterator {
            f(item?)?;
        }
    }
    Ok(())
}

fn number(item: &Bound<'_, PyAny>, name: &str, row: usize, option: usize) -> PyResult<f64> {
    if let Ok(float) = item.cast_exact::<PyFloat>() {
        return Ok(float.value());
    }
    if item.is_exact_instance_of::<PyInt>() {
        return item.extract::<f64>();
    }
    // `bool_` is numpy's boolean scalar: it has `__float__`, but it is still a bool.
    let rejected = item.is_instance_of::<PyBool>()
        || item.is_instance_of::<PyString>()
        || item.is_instance_of::<PyBytes>()
        || item.is_none()
        || type_name(item).starts_with("bool");
    if !rejected {
        if item.is_instance_of::<PyFloat>() || item.is_instance_of::<PyInt>() {
            return item.extract::<f64>();
        }
        let ty = item.get_type();
        if ty.hasattr("__float__")? || ty.hasattr("__index__")? {
            return item.extract::<f64>();
        }
    }
    Err(PyTypeError::new_err(format!(
        "{name} row {row} option {option} must be a number, got {}",
        type_name(item)
    )))
}

fn read_row_into(
    row: &Bound<'_, PyAny>,
    name: &str,
    index: usize,
    builder: &mut RowsBuilder,
    not_iterable: impl FnOnce() -> PyErr,
) -> PyResult<()> {
    let mut option = 0usize;
    visit(row, not_iterable, |item| {
        builder.push(number(&item, name, index, option)?);
        option += 1;
        Ok(())
    })
}

/// `rows`: a sequence of sequences of numbers.
pub fn read_rows(obj: &Bound<'_, PyAny>, name: &'static str) -> PyResult<Rows> {
    // For a list of lists the size is known: one allocation instead of a growing buffer.
    let mut builder = match obj.cast_exact::<PyList>() {
        Ok(list) => {
            let first = list
                .get_item(0)
                .and_then(|row| row.len())
                .unwrap_or_default();
            // The first row is only a hint: cap it, a long first row must not reserve gigabytes.
            let hint = list.len().saturating_mul(first).min(1 << 22);
            RowsBuilder::with_capacity(list.len(), hint)
        }
        Err(_) => RowsBuilder::new(),
    };
    let mut index = 0usize;
    visit(
        obj,
        || {
            PyTypeError::new_err(format!(
                "{name} must be a sequence of rows, got {}",
                type_name(obj)
            ))
        },
        |row| {
            let not_iterable = || {
                PyTypeError::new_err(format!(
                    "{name} row {index} must be a sequence of numbers, got {}",
                    type_name(&row)
                ))
            };
            read_row_into(&row, name, index, &mut builder, not_iterable)?;
            builder.end_row();
            index += 1;
            Ok(())
        },
    )?;
    Ok(builder.finish())
}

/// One row: a sequence of numbers (reported as row 0).
pub fn read_row(obj: &Bound<'_, PyAny>, name: &'static str) -> PyResult<Rows> {
    let mut builder = RowsBuilder::new();
    let not_iterable = || {
        PyTypeError::new_err(format!(
            "{name} must be a sequence of numbers, got {}",
            type_name(obj)
        ))
    };
    read_row_into(obj, name, 0, &mut builder, not_iterable)?;
    builder.end_row();
    Ok(builder.finish())
}

/// `labels`: a sequence of integers (numpy integers included; `bool` and `float` are rejected).
pub fn read_labels(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let mut labels = Vec::new();
    visit(
        obj,
        || {
            PyTypeError::new_err(format!(
                "labels must be a sequence of integers, got {}",
                type_name(obj)
            ))
        },
        |item| {
            let position = labels.len();
            let bad = || {
                PyTypeError::new_err(format!(
                    "labels must be integers, got {} at position {position}",
                    type_name(&item)
                ))
            };
            if !is_integer_like(&item)? {
                return Err(bad());
            }
            match item.extract::<i64>() {
                Ok(label) => labels.push(label),
                // Too big for 64 bits: certainly no option of any row. A ValueError, like every
                // other label outside its row.
                Err(e) if e.is_instance_of::<PyOverflowError>(item.py()) => {
                    return Err(PyValueError::new_err(format!(
                        "label at position {position} is {}, which is outside every row (valid labels are 0-based option indices)",
                        describe(&item)
                    )));
                }
                Err(e) => return Err(e),
            }
            Ok(())
        },
    )?;
    Ok(labels)
}
