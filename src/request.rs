//! A request in the TypeSafe shape (`state` plus `questions` with `type`, `instructions`,
//! `criteria`) read into the domain types. Used by the self-check vectors of a model manifest;
//! the full JSON API of spec 003 builds on it.
//!
//! The reader is as strict as the library (D30, D32): what Kev accepts but the domain types
//! refuse (a bare number as the state, `null` instructions, unknown keys in the criteria of a
//! yes/no question) is an error, not a guess.

use crate::error::Error;
use crate::json_strict::Json;
use crate::schema::{
    Choice, ChoiceOption, Question, Questions, Score, State, StateValue, TextEntry, YesNo,
};

/// Turn a JSON value into a state value. `string_hook` may reinterpret a string (the golden
/// reader uses it for integers of more than 64 bits); it returns `None` to leave it as text.
pub(crate) fn value_of(
    j: &Json,
    hook: &dyn Fn(&str) -> Option<StateValue>,
) -> Result<StateValue, Error> {
    Ok(match j {
        Json::Null => StateValue::Null,
        Json::Bool(b) => StateValue::Bool(*b),
        Json::Int(i) => StateValue::int_text(&i.to_string())?,
        Json::BigInt(text) => StateValue::int_text(text)?,
        Json::Float(f) => StateValue::Float(*f),
        Json::Str(s) => hook(s).unwrap_or_else(|| StateValue::string(s.clone())),
        Json::Array(items) => StateValue::list(
            items
                .iter()
                .map(|v| value_of(v, hook))
                .collect::<Result<Vec<_>, Error>>()?,
        ),
        Json::Object(pairs) => StateValue::object(
            pairs
                .iter()
                .map(|(k, v)| Ok((k.clone(), value_of(v, hook)?)))
                .collect::<Result<Vec<_>, Error>>()?,
        ),
    })
}

fn entry(
    field: &str,
    j: &Json,
    hook: &dyn Fn(&str) -> Option<StateValue>,
) -> Result<TextEntry, Error> {
    TextEntry::new(field, value_of(j, hook)?)
}

fn optional_entry(
    field: &str,
    j: Option<&Json>,
    hook: &dyn Fn(&str) -> Option<StateValue>,
) -> Result<Option<TextEntry>, Error> {
    match j {
        None | Some(Json::Null) => Ok(None),
        Some(other) => entry(field, other, hook).map(Some),
    }
}

pub(crate) fn question_of(
    name: &str,
    q: &Json,
    hook: &dyn Fn(&str) -> Option<StateValue>,
) -> Result<Question, Error> {
    let Some(Json::Str(kind)) = q.get("type") else {
        return Err(Error::EntryConversion {
            message: format!("question \"{name}\" has no type"),
        });
    };
    let instructions = entry(
        &format!("instructions of question \"{name}\""),
        q.get("instructions").unwrap_or(&Json::Null),
        hook,
    )?;
    match (kind.as_str(), q.get("criteria")) {
        ("choice", Some(Json::Object(pairs))) => {
            let mut options = Vec::with_capacity(pairs.len());
            for (key, desc) in pairs {
                options.push(ChoiceOption {
                    key: key.clone(),
                    description: optional_entry("description", Some(desc), hook)?,
                });
            }
            Choice::new(instructions, options).map(Question::from)
        }
        ("score", Some(Json::Array(levels))) => {
            let levels = levels
                .iter()
                .map(|l| entry("level", l, hook))
                .collect::<Result<Vec<_>, _>>()?;
            Score::new(instructions, levels).map(Question::from)
        }
        ("noul", None | Some(Json::Null)) => {
            YesNo::new(instructions, None, None).map(Question::from)
        }
        ("noul", Some(Json::Object(pairs))) => {
            if let Some((k, _)) = pairs.iter().find(|(k, _)| k != "true" && k != "false") {
                return Err(Error::EntryConversion {
                    message: format!(
                        "the criteria of yes/no question \"{name}\" have the unknown key \"{k}\"; only \"true\" and \"false\" are allowed"
                    ),
                });
            }
            let t = optional_entry("true", q.get("criteria").and_then(|c| c.get("true")), hook)?;
            let f = optional_entry(
                "false",
                q.get("criteria").and_then(|c| c.get("false")),
                hook,
            )?;
            YesNo::new(instructions, t, f).map(Question::from)
        }
        (other, _) => Err(Error::EntryConversion {
            message: format!(
                "question \"{name}\": cannot read a {other} question with these criteria"
            ),
        }),
    }
}

/// Read a request into the domain types, or say why the library refuses it.
///
/// # Errors
/// The reason, as text.
pub fn request_to_domain(request: &Json) -> Result<(State, Questions), String> {
    request_to_domain_with(request, &|_| None)
}

/// Like [`request_to_domain`], with a hook that may reinterpret strings (see [`value_of`]).
///
/// # Errors
/// The reason, as text.
pub fn request_to_domain_with(
    request: &Json,
    hook: &dyn Fn(&str) -> Option<StateValue>,
) -> Result<(State, Questions), String> {
    request_to_domain_typed(request, hook).map_err(|e| e.to_string())
}

/// Like [`request_to_domain_with`], with the typed error (its category is that of the part of
/// the request that was refused).
///
/// # Errors
/// A state or question error of the domain types.
pub fn request_to_domain_typed(
    request: &Json,
    hook: &dyn Fn(&str) -> Option<StateValue>,
) -> Result<(State, Questions), Error> {
    let state = State::new(value_of(request.get("state").unwrap_or(&Json::Null), hook)?)?;
    let Some(Json::Object(qs)) = request.get("questions") else {
        return Err(Error::NoQuestions);
    };
    let mut items = Vec::with_capacity(qs.len());
    for (name, q) in qs {
        items.push((name.clone(), question_of(name, q, hook)?));
    }
    Ok((state, Questions::new(items)?))
}
