//! Small helpers to build and edit [`Json`] trees in tests.

use crate::json_strict::Json;

pub fn s(x: &str) -> Json {
    Json::Str(x.to_string())
}

pub fn n(x: u64) -> Json {
    Json::Int(i128::from(x))
}

pub fn f(x: f64) -> Json {
    Json::Float(x)
}

pub fn b(x: bool) -> Json {
    Json::Bool(x)
}

pub fn arr(items: Vec<Json>) -> Json {
    Json::Array(items)
}

pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn step<'a>(node: &'a mut Json, key: &str) -> &'a mut Json {
    match node {
        Json::Object(pairs) => {
            let i = pairs
                .iter()
                .position(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("no key {key:?}"));
            &mut pairs[i].1
        }
        Json::Array(items) => {
            let i: usize = key
                .parse()
                .unwrap_or_else(|_| panic!("not an index: {key:?}"));
            &mut items[i]
        }
        other => panic!("cannot step into {other:?} with {key:?}"),
    }
}

/// Set (or add, for an object key) the value at a dotted `path` such as `variants.0.dtype`.
pub fn set(root: &mut Json, path: &str, value: Json) {
    let parts: Vec<&str> = path.split('.').collect();
    let (last, init) = parts.split_last().unwrap();
    let mut node = root;
    for p in init {
        node = step(node, p);
    }
    match node {
        Json::Object(pairs) => {
            if let Some(slot) = pairs.iter_mut().find(|(k, _)| k == last) {
                slot.1 = value;
            } else {
                pairs.push(((*last).to_string(), value));
            }
        }
        Json::Array(items) => {
            let i: usize = last.parse().unwrap();
            items[i] = value;
        }
        other => panic!("cannot set into {other:?}"),
    }
}

/// Remove the key (or array element) at a dotted `path`.
pub fn remove(root: &mut Json, path: &str) {
    let parts: Vec<&str> = path.split('.').collect();
    let (last, init) = parts.split_last().unwrap();
    let mut node = root;
    for p in init {
        node = step(node, p);
    }
    match node {
        Json::Object(pairs) => {
            let i = pairs
                .iter()
                .position(|(k, _)| k == last)
                .unwrap_or_else(|| panic!("no key {last:?} to remove"));
            pairs.remove(i);
        }
        Json::Array(items) => {
            items.remove(last.parse::<usize>().unwrap());
        }
        other => panic!("cannot remove from {other:?}"),
    }
}

/// The value at a dotted path, if any.
pub fn get<'a>(root: &'a Json, path: &str) -> Option<&'a Json> {
    let mut node = root;
    for p in path.split('.') {
        node = match node {
            Json::Object(pairs) => &pairs.iter().find(|(k, _)| k == p)?.1,
            Json::Array(items) => items.get(p.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(node)
}
