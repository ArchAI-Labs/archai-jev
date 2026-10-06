//! Python values → core values. Strict by design (D20): only the types of the table in the
//! spec are accepted, and **no user code runs** during the conversion (no `__str__`, no
//! `__iter__`, no overridden `keys()`): containers are read as native structures.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString, PyTuple};

use crate::error::Error;
use crate::schema::{StateValue, render::MAX_DEPTH};

enum Seg {
    Key(String),
    Index(usize),
}

fn path_string(path: &[Seg]) -> String {
    if path.is_empty() {
        return "<root>".to_string();
    }
    let mut out = String::new();
    for seg in path {
        match seg {
            Seg::Key(k) => {
                if !out.is_empty() {
                    out.push('.');
                }
                out.push_str(k);
            }
            Seg::Index(i) => out.push_str(&format!("[{i}]")),
        }
    }
    if out.chars().count() > 64 {
        let cut: String = out.chars().take(64).collect();
        format!("{cut}...")
    } else {
        out
    }
}

/// Why a Python value could not be converted.
enum Fail {
    Unsupported {
        path: String,
        ty: String,
    },
    NonStrKey {
        path: String,
        key: String,
        ty: String,
    },
    Surrogate {
        path: String,
        code_point: u32,
    },
    BigInt {
        path: String,
    },
    TooDeep {
        path: String,
    },
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

/// Text of a key for messages, without running user code: only exact builtin types are `repr`'d.
fn safe_key_repr(key: &Bound<'_, PyAny>) -> String {
    let exact = key.is_exact_instance_of::<PyInt>()
        || key.is_exact_instance_of::<PyFloat>()
        || key.is_exact_instance_of::<PyBool>()
        || key.is_exact_instance_of::<PyNone>();
    if exact && let Ok(r) = key.repr() {
        return r.to_string();
    }
    format!("<{} object>", type_name(key))
}

fn convert_str(obj: &Bound<'_, PyString>, path: &[Seg]) -> Result<String, Fail> {
    match obj.to_str() {
        Ok(s) => Ok(s.to_string()),
        Err(_) => {
            // Re-encode with `surrogatepass` to find the first lone surrogate (ED A0..BF xx).
            let py = obj.py();
            let code_point = py
                .get_type::<PyString>()
                .call_method1("encode", (obj, "utf-8", "surrogatepass"))
                .and_then(|b| b.extract::<Vec<u8>>())
                .ok()
                .and_then(|bytes| {
                    bytes.windows(3).find_map(|w| match w {
                        [0xED, b1, b2] if *b1 >= 0xA0 => {
                            Some((0xD000u32) | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F))
                        }
                        _ => None,
                    })
                })
                .unwrap_or(0xD800);
            Err(Fail::Surrogate {
                path: path_string(path),
                code_point,
            })
        }
    }
}

fn convert(obj: &Bound<'_, PyAny>, depth: usize, path: &mut Vec<Seg>) -> Result<StateValue, Fail> {
    // bool before int: bool is a subclass of int
    if obj.is_instance_of::<PyBool>() {
        return Ok(StateValue::Bool(obj.extract::<bool>().unwrap_or(false)));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return convert_str(s, path).map(StateValue::Str);
    }
    if obj.is_instance_of::<PyInt>() {
        // `int.__repr__` called on the type: a subclass override (IntEnum, ...) never runs.
        let py = obj.py();
        let text = py
            .get_type::<PyInt>()
            .call_method1("__repr__", (obj,))
            .and_then(|t| t.extract::<String>());
        return match text {
            Ok(t) => StateValue::int_text(&t).map_err(|_| Fail::Unsupported {
                path: path_string(path),
                ty: type_name(obj),
            }),
            Err(_) => Err(Fail::BigInt {
                path: path_string(path),
            }),
        };
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return Ok(StateValue::Float(f.value()));
    }
    if obj.is_none() {
        return Ok(StateValue::Null);
    }
    let container = obj.is_instance_of::<PyDict>()
        || obj.is_instance_of::<PyList>()
        || obj.is_instance_of::<PyTuple>();
    if container && depth + 1 > MAX_DEPTH {
        return Err(Fail::TooDeep {
            path: path_string(path),
        });
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut pairs = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key = match k.cast::<PyString>() {
                Ok(ks) => convert_str(ks, path)?,
                Err(_) => {
                    return Err(Fail::NonStrKey {
                        path: path_string(path),
                        key: safe_key_repr(&k),
                        ty: type_name(&k),
                    });
                }
            };
            path.push(Seg::Key(key.clone()));
            let value = convert(&v, depth + 1, path);
            path.pop();
            pairs.push((key, value?));
        }
        return Ok(StateValue::Object(pairs));
    }
    let items: Option<Vec<Bound<'_, PyAny>>> = if let Ok(l) = obj.cast::<PyList>() {
        Some(l.iter().collect())
    } else if let Ok(t) = obj.cast::<PyTuple>() {
        Some(t.iter().collect())
    } else {
        None
    };
    if let Some(items) = items {
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            path.push(Seg::Index(i));
            let value = convert(item, depth + 1, path);
            path.pop();
            out.push(value?);
        }
        return Ok(StateValue::List(out));
    }
    Err(Fail::Unsupported {
        path: path_string(path),
        ty: type_name(obj),
    })
}

/// Convert a Python object to a state value (errors are `InvalidStateError`s).
pub(crate) fn state_value(obj: &Bound<'_, PyAny>) -> Result<StateValue, Error> {
    convert(obj, 0, &mut Vec::new()).map_err(|f| match f {
        Fail::Unsupported { path, ty } => Error::StateConversion {
            message: format!(
                "state value at {path} has unsupported type {ty}; supported types are str, int, float, bool, None, dict (str keys) and list or tuple; convert it first, e.g. with str() or list()"
            ),
        },
        Fail::NonStrKey { path, key, ty } => Error::StateConversion {
            message: format!(
                "state object key {key} at {path} is {ty}, not str; object keys must be strings"
            ),
        },
        Fail::Surrogate { path, code_point } => Error::StateConversion {
            message: format!(
                "state string at {path} contains the lone surrogate U+{code_point:04X} and cannot be encoded as UTF-8; remove or replace it"
            ),
        },
        Fail::BigInt { path } => Error::StateConversion {
            message: format!(
                "state integer at {path} has more than 4300 digits; pass it as a string instead"
            ),
        },
        Fail::TooDeep { path } => Error::StateTooDeep { path },
    })
}

/// Convert a Python object to a question's text entry called `field` (errors are
/// `InvalidQuestionError`s).
pub(crate) fn entry_value(obj: &Bound<'_, PyAny>, field: &str) -> Result<StateValue, Error> {
    convert(obj, 0, &mut Vec::new()).map_err(|f| {
        let message = match f {
            Fail::Unsupported { path, ty } => {
                if path == "<root>" {
                    return Error::InvalidFieldType {
                        field: field.to_string(),
                        got: ty,
                    };
                }
                format!("{field} contains {ty} at {path}, which is not supported; use str, int, float, bool, None, dict (str keys) or list")
            }
            Fail::NonStrKey { path, key, ty } => {
                format!("{field}: object key {key} at {path} is {ty}, not str")
            }
            Fail::Surrogate { path, code_point } => format!(
                "{field}: string at {path} contains the lone surrogate U+{code_point:04X}"
            ),
            Fail::BigInt { path } => format!("{field}: integer at {path} has more than 4300 digits"),
            Fail::TooDeep { path } => format!("{field} is nested more than 32 levels deep at {path}"),
        };
        Error::EntryConversion { message }
    })
}
