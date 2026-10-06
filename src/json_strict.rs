//! A strict JSON reader for manifests and other files we do not trust.
//!
//! Stricter than plain `serde_json`: duplicate object keys are an error (a parser that keeps
//! the last one would let a repeated `"family"` change meaning silently), nesting and size are
//! limited, a byte-order mark is refused, and objects keep their key order. Numbers must fit
//! an integer of 64 bits or a finite float.
//!
//! [`Reader`] then walks the tree field by field, remembering the path (`variants[0].calibration`)
//! so every error names the field and unknown keys are rejected instead of ignored.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};

/// Largest accepted document, in bytes.
pub const MAX_BYTES: usize = 8 * 1024 * 1024;
/// Deepest accepted nesting of objects and arrays.
pub const MAX_DEPTH: usize = 16;

/// A parsed JSON value. Objects are lists of pairs in document order.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// An integer that fits `i64` or `u64`.
    Int(i128),
    /// A finite float.
    Float(f64),
    /// A string.
    Str(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, keys unique, in document order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Name of the kind, for messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Int(_) | Json::Float(_) => "a number",
            Json::Str(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }

    /// The value of `key` if this is an object that has it.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Compact text of this value, with keys in document order (stable, used for hashing).
    pub fn to_canonical_string(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::Float(f) => out.push_str(
                &serde_json::Number::from_f64(*f)
                    .map_or_else(|| "null".to_string(), |n| n.to_string()),
            ),
            Json::Str(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_canonical(out);
                }
                out.push(']');
            }
            Json::Object(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push(':');
                    v.write_canonical(out);
                }
                out.push('}');
            }
        }
    }
}

/// Why a document was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonProblem {
    /// Broken syntax or an unsupported number (message already says where).
    Syntax(String),
    /// A byte-order mark, size or depth limit.
    Limit(String),
    /// An object with the same key twice.
    DuplicateKey(String),
}

impl fmt::Display for JsonProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonProblem::Syntax(m) | JsonProblem::Limit(m) => write!(f, "{m}"),
            JsonProblem::DuplicateKey(k) => write!(f, "the key \"{k}\" appears more than once"),
        }
    }
}

struct Node {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for Node {
    type Value = Json;
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Node {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a JSON value")
    }
    fn visit_unit<E: de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }
    fn visit_none<E: de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Json, E> {
        Ok(Json::Bool(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Json, E> {
        Ok(Json::Int(i128::from(v)))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Json, E> {
        Ok(Json::Int(i128::from(v)))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Json, E> {
        if v.is_finite() {
            Ok(Json::Float(v))
        } else {
            Err(E::custom("number out of range"))
        }
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Json, E> {
        Ok(Json::Str(v.to_string()))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<Json, E> {
        Ok(Json::Str(v))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        if self.depth + 1 > MAX_DEPTH {
            return Err(de::Error::custom(format!(
                "nesting is deeper than {MAX_DEPTH} levels"
            )));
        }
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(Node {
            depth: self.depth + 1,
        })? {
            items.push(item);
        }
        Ok(Json::Array(items))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        if self.depth + 1 > MAX_DEPTH {
            return Err(de::Error::custom(format!(
                "nesting is deeper than {MAX_DEPTH} levels"
            )));
        }
        let mut pairs: Vec<(String, Json)> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if pairs.iter().any(|(k, _)| *k == key) {
                return Err(de::Error::custom(format!("duplicate key \"{key}\"")));
            }
            let value = map.next_value_seed(Node {
                depth: self.depth + 1,
            })?;
            pairs.push((key, value));
        }
        Ok(Json::Object(pairs))
    }
}

/// Parse `bytes` strictly.
///
/// # Errors
/// [`JsonProblem`] for broken syntax, a byte-order mark, more than 8 MiB, nesting deeper than
/// 16 levels, duplicate keys, or a number that is not an integer of 64 bits or a finite float.
pub fn parse(bytes: &[u8]) -> Result<Json, JsonProblem> {
    if bytes.len() > MAX_BYTES {
        return Err(JsonProblem::Limit(format!(
            "the file is {} bytes; the maximum is {MAX_BYTES}",
            bytes.len()
        )));
    }
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(JsonProblem::Limit(
            "the file starts with a byte order mark; save it as UTF-8 without BOM".to_string(),
        ));
    }
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let value = Node { depth: 0 }.deserialize(&mut de).and_then(|v| {
        de.end()?;
        Ok(v)
    });
    value.map_err(|e| {
        let text = e.to_string();
        if let Some(rest) = text.strip_prefix("duplicate key \"") {
            let key = rest.split('"').next().unwrap_or("").to_string();
            return JsonProblem::DuplicateKey(key);
        }
        if text.starts_with("nesting is deeper") {
            return JsonProblem::Limit(text);
        }
        JsonProblem::Syntax(format!("invalid JSON: {text}"))
    })
}

/// A field-by-field reader of one JSON object that remembers its path.
///
/// Every `take_*` marks the field as read; [`Obj::finish`] rejects any field that was not.
pub struct Obj<'a> {
    path: String,
    fields: Vec<(&'a str, &'a Json, bool)>,
}

/// What went wrong while reading a field.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldProblem {
    /// A required field is absent.
    Missing { path: String },
    /// A field the schema does not know.
    Unknown { path: String, field: String },
    /// A field of the wrong kind or out of its domain.
    Bad { path: String, detail: String },
}

impl<'a> Obj<'a> {
    /// Start reading `json`, which must be an object, called `path` in messages
    /// (`"<manifest>"` for the root).
    ///
    /// # Errors
    /// [`FieldProblem::Bad`] if `json` is not an object.
    pub fn new(json: &'a Json, path: &str) -> Result<Obj<'a>, FieldProblem> {
        match json {
            Json::Object(pairs) => Ok(Obj {
                path: path.to_string(),
                fields: pairs.iter().map(|(k, v)| (k.as_str(), v, false)).collect(),
            }),
            other => Err(FieldProblem::Bad {
                path: path.to_string(),
                detail: format!("must be an object, got {}", other.kind()),
            }),
        }
    }

    /// The path of this object.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The path of one of its fields.
    pub fn child_path(&self, key: &str) -> String {
        if self.path == "<manifest>" {
            key.to_string()
        } else {
            format!("{}.{key}", self.path)
        }
    }

    fn take(&mut self, key: &str) -> Option<&'a Json> {
        self.fields
            .iter_mut()
            .find(|(k, _, _)| *k == key)
            .map(|(_, v, used)| {
                *used = true;
                *v
            })
    }

    /// The field `key`, or [`FieldProblem::Missing`].
    pub fn req(&mut self, key: &str) -> Result<&'a Json, FieldProblem> {
        let path = self.child_path(key);
        self.take(key).ok_or(FieldProblem::Missing { path })
    }

    /// The field `key` if present.
    pub fn opt(&mut self, key: &str) -> Option<&'a Json> {
        self.take(key)
    }

    fn bad(&self, key: &str, detail: String) -> FieldProblem {
        FieldProblem::Bad {
            path: self.child_path(key),
            detail,
        }
    }

    /// A required string.
    pub fn str(&mut self, key: &str) -> Result<&'a str, FieldProblem> {
        match self.req(key)? {
            Json::Str(s) => Ok(s),
            other => Err(self.bad(key, format!("must be a string, got {}", other.kind()))),
        }
    }

    /// A required non-negative integer.
    pub fn u64(&mut self, key: &str) -> Result<u64, FieldProblem> {
        match self.req(key)? {
            Json::Int(i) if *i >= 0 => {
                u64::try_from(*i).map_err(|_| self.bad(key, "is too large".to_string()))
            }
            other => Err(self.bad(
                key,
                format!("must be a non-negative integer, got {}", describe(other)),
            )),
        }
    }

    /// A required boolean.
    pub fn bool(&mut self, key: &str) -> Result<bool, FieldProblem> {
        match self.req(key)? {
            Json::Bool(b) => Ok(*b),
            other => Err(self.bad(key, format!("must be a boolean, got {}", other.kind()))),
        }
    }

    /// A required finite number (integer or float).
    pub fn f64(&mut self, key: &str) -> Result<f64, FieldProblem> {
        match self.req(key)? {
            Json::Int(i) => Ok(*i as f64),
            Json::Float(f) => Ok(*f),
            other => Err(self.bad(key, format!("must be a number, got {}", other.kind()))),
        }
    }

    /// A required object, to be read with its own [`Obj`].
    pub fn obj(&mut self, key: &str) -> Result<Obj<'a>, FieldProblem> {
        let json = self.req(key)?;
        Obj::new(json, &self.child_path(key))
    }

    /// A required array.
    pub fn arr(&mut self, key: &str) -> Result<&'a [Json], FieldProblem> {
        match self.req(key)? {
            Json::Array(items) => Ok(items),
            other => Err(self.bad(key, format!("must be an array, got {}", other.kind()))),
        }
    }

    /// The fields nobody read yet, consuming the reader (for free-form sub-objects whose
    /// allowed keys are decided later).
    pub fn rest(self) -> Vec<(&'a str, &'a Json)> {
        self.fields
            .into_iter()
            .filter(|(_, _, used)| !used)
            .map(|(k, v, _)| (k, v))
            .collect()
    }

    /// Fail on the first field nobody read.
    ///
    /// # Errors
    /// [`FieldProblem::Unknown`] naming the field.
    pub fn finish(self) -> Result<(), FieldProblem> {
        match self.fields.iter().find(|(_, _, used)| !used) {
            None => Ok(()),
            Some((k, _, _)) => Err(FieldProblem::Unknown {
                path: self.path.clone(),
                field: (*k).to_string(),
            }),
        }
    }
}

fn describe(j: &Json) -> String {
    match j {
        Json::Int(i) => format!("the integer {i}"),
        other => other.kind().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> Json {
        parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn parses_values_and_keeps_key_order() {
        let j = ok(r#"{"b": 1, "a": [true, null, 1.5, "x"], "c": {"z": 0}}"#);
        let Json::Object(p) = &j else { panic!() };
        assert_eq!(
            p.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["b", "a", "c"]
        );
        assert_eq!(
            j.get("a"),
            Some(&Json::Array(vec![
                Json::Bool(true),
                Json::Null,
                Json::Float(1.5),
                Json::Str("x".into())
            ]))
        );
        assert_eq!(
            j.to_canonical_string(),
            r#"{"b":1,"a":[true,null,1.5,"x"],"c":{"z":0}}"#
        );
    }

    #[test]
    fn integers_up_to_u64_and_negative_are_exact() {
        assert_eq!(
            ok("18446744073709551615"),
            Json::Int(18_446_744_073_709_551_615)
        );
        assert_eq!(
            ok("-9223372036854775808"),
            Json::Int(-9_223_372_036_854_775_808)
        );
        // beyond u64 serde_json reads a float: fields that need an integer then refuse it
        let big = ok("{\"n\": 18446744073709551616}");
        assert!(matches!(big.get("n"), Some(Json::Float(_))));
        assert!(Obj::new(&big, "x").unwrap().u64("n").is_err());
        assert!(matches!(parse(b"1e999"), Err(JsonProblem::Syntax(_))));
    }

    #[test]
    fn rejects_the_things_plain_parsers_let_through() {
        assert_eq!(
            parse(br#"{"a": 1, "a": 2}"#),
            Err(JsonProblem::DuplicateKey("a".into()))
        );
        assert_eq!(
            parse(br#"{"x": {"a": 1, "b": 2, "a": 3}}"#),
            Err(JsonProblem::DuplicateKey("a".into()))
        );
        assert!(matches!(
            parse(b"\xEF\xBB\xBF{}"),
            Err(JsonProblem::Limit(_))
        ));
        assert!(matches!(parse(b"{"), Err(JsonProblem::Syntax(_))));
        assert!(matches!(parse(b"NaN"), Err(JsonProblem::Syntax(_))));
        assert!(matches!(parse(b"Infinity"), Err(JsonProblem::Syntax(_))));
        assert!(matches!(parse(b"{} {}"), Err(JsonProblem::Syntax(_))));
        assert!(matches!(parse(b"\"\xFF\""), Err(JsonProblem::Syntax(_))));
        assert!(matches!(parse(b""), Err(JsonProblem::Syntax(_))));
    }

    #[test]
    fn depth_limit_is_exact() {
        let nest = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(parse(nest(MAX_DEPTH).as_bytes()).is_ok());
        assert!(matches!(
            parse(nest(MAX_DEPTH + 1).as_bytes()),
            Err(JsonProblem::Limit(_))
        ));
        assert!(matches!(
            parse(nest(5000).as_bytes()),
            Err(JsonProblem::Limit(_))
        ));
    }

    #[test]
    fn size_limit_is_exact() {
        let pad = |n: usize| format!("\"{}\"", "x".repeat(n - 2));
        assert!(parse(pad(MAX_BYTES).as_bytes()).is_ok());
        assert!(matches!(
            parse(pad(MAX_BYTES + 1).as_bytes()),
            Err(JsonProblem::Limit(_))
        ));
    }

    #[test]
    fn reader_tracks_paths_and_rejects_unknown_fields() {
        let j = ok(r#"{"name": "n", "inner": {"count": 3, "extra": 1}, "list": []}"#);
        let mut root = Obj::new(&j, "<manifest>").unwrap();
        assert_eq!(root.str("name").unwrap(), "n");
        let mut inner = root.obj("inner").unwrap();
        assert_eq!(inner.u64("count").unwrap(), 3);
        assert_eq!(
            inner.finish(),
            Err(FieldProblem::Unknown {
                path: "inner".into(),
                field: "extra".into()
            })
        );
        assert_eq!(
            root.req("nope"),
            Err(FieldProblem::Missing {
                path: "nope".into()
            })
        );
        assert!(root.arr("list").unwrap().is_empty());
        assert_eq!(root.finish(), Ok(()));
    }

    #[test]
    fn reader_type_errors_name_the_field() {
        let j = ok(r#"{"a": "x", "b": -1, "c": 1.5}"#);
        let mut r = Obj::new(&j, "<manifest>").unwrap();
        let Err(FieldProblem::Bad { path, detail }) = r.u64("a") else {
            panic!()
        };
        assert_eq!(
            (path.as_str(), detail.as_str()),
            ("a", "must be a non-negative integer, got a string")
        );
        let Err(FieldProblem::Bad { detail, .. }) = r.u64("b") else {
            panic!()
        };
        assert!(detail.contains("-1"));
        assert_eq!(r.f64("c").unwrap(), 1.5);
        assert!(Obj::new(&Json::Null, "x").is_err());
    }
}
