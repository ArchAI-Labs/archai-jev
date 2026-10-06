//! Canonical text of a state: Kev's `render`, byte for byte (S2 NOTES 2.2).
//!
//! Three details a "natural" translation would get wrong, all covered by tests:
//! Python's `lstrip` set of spaces, Python's `str(float)`, and the odd blank lines.

use std::collections::HashSet;

use super::value::StateValue;
use crate::error::{Error, Result};

/// Maximum nesting of objects/lists.
pub const MAX_DEPTH: usize = 32;

/// Characters for which Python's `str.isspace()` is true (29 code points).
pub fn is_py_space(c: char) -> bool {
    matches!(
        c,
        '\u{9}'..='\u{d}'
            | '\u{1c}'..='\u{1f}'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// True if `s` is empty or made only of Python whitespace.
pub fn is_blank(s: &str) -> bool {
    s.chars().all(is_py_space)
}

/// One step of the path to a node, for error messages.
enum Seg<'a> {
    Key(&'a str),
    Index(usize),
}

fn path_string(path: &[Seg<'_>]) -> String {
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
            Seg::Index(i) => {
                out.push('[');
                out.push_str(&i.to_string());
                out.push(']');
            }
        }
    }
    if out.chars().count() > 64 {
        let cut: String = out.chars().take(64).collect();
        format!("{cut}...")
    } else {
        out
    }
}

/// Output buffer with a size limit and Python-style `lstrip` support.
struct Out {
    buf: String,
    limit: usize,
    /// While true, leading Python whitespace is dropped: the text being written is the
    /// rendered text of a list item, whose `lstrip` Kev applies.
    strip: bool,
}

impl Out {
    fn push(&mut self, s: &str) -> Result<()> {
        let mut s = s;
        if self.strip {
            s = s.trim_start_matches(is_py_space);
            if s.is_empty() {
                return Ok(());
            }
            self.strip = false;
        }
        self.buf.push_str(s);
        if self.buf.len() > self.limit {
            return Err(Error::StateTooLarge);
        }
        Ok(())
    }

    fn pad(&mut self, indent: usize) -> Result<()> {
        for _ in 0..indent {
            self.push("  ")?;
        }
        Ok(())
    }
}

/// Render `value` as Kev does, failing as soon as the output would exceed `limit` bytes.
///
/// With `validate` false this is the low-level renderer: any node is accepted as root and
/// nothing is checked. With `validate` true it also enforces what a real state needs: finite
/// floats, unique keys and the depth limit.
pub(crate) fn render_checked(value: &StateValue, limit: usize, validate: bool) -> Result<String> {
    let mut out = Out {
        buf: String::new(),
        limit,
        strip: false,
    };
    let mut path: Vec<Seg<'_>> = Vec::new();
    render_node(value, 0, 0, &mut out, &mut path, validate)?;
    Ok(out.buf)
}

fn render_node<'a>(
    value: &'a StateValue,
    indent: usize,
    depth: usize,
    out: &mut Out,
    path: &mut Vec<Seg<'a>>,
    validate: bool,
) -> Result<()> {
    match value {
        StateValue::Null => Ok(()),
        StateValue::Bool(b) => out.push(if *b { "True" } else { "False" }),
        StateValue::Int(text) => out.push(text),
        StateValue::Float(x) => {
            if !x.is_finite() && validate {
                let value = if x.is_nan() {
                    "NaN"
                } else if *x > 0.0 {
                    "Infinity"
                } else {
                    "-Infinity"
                };
                return Err(Error::NonFiniteNumber {
                    value: value.to_string(),
                    path: path_string(path),
                });
            }
            out.push(&py_float_repr(*x))
        }
        StateValue::Str(s) => out.push(s),
        StateValue::List(items) => {
            if validate && depth + 1 > MAX_DEPTH {
                return Err(Error::StateTooDeep {
                    path: path_string(path),
                });
            }
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push("\n")?;
                }
                out.pad(indent)?;
                out.push("- ")?;
                out.strip = true;
                path.push(Seg::Index(i));
                let r = render_node(item, indent + 1, depth + 1, out, path, validate);
                path.pop();
                out.strip = false;
                r?;
            }
            Ok(())
        }
        StateValue::Object(pairs) => {
            if validate {
                if depth + 1 > MAX_DEPTH {
                    return Err(Error::StateTooDeep {
                        path: path_string(path),
                    });
                }
                let mut seen: HashSet<&str> = HashSet::with_capacity(pairs.len());
                for (k, _) in pairs {
                    if !seen.insert(k.as_str()) {
                        return Err(Error::DuplicateStateKey {
                            key: k.clone(),
                            path: path_string(path),
                        });
                    }
                }
            }
            for (i, (key, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push("\n")?;
                }
                out.pad(indent)?;
                out.push(key)?;
                path.push(Seg::Key(key));
                let sep = if val.is_container() { ":\n" } else { ": " };
                let r = out
                    .push(sep)
                    .and_then(|()| render_node(val, indent + 1, depth + 1, out, path, validate));
                path.pop();
                r?;
            }
            Ok(())
        }
    }
}

/// Python's `repr(float)`: shortest round-trip digits, laid out like CPython does.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // `{:e}` prints the shortest digits that round-trip, e.g. "1.2345678901234567e19".
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp10: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if digits.bytes().all(|b| b == b'0') {
        return format!("{sign}0.0");
    }
    let decpt = exp10 + 1;
    let n = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let body = if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            format!(
                "0.{}{}",
                "0".repeat(usize::try_from(-decpt).unwrap_or(0)),
                digits
            )
        } else if decpt >= n {
            format!(
                "{}{}.0",
                digits,
                "0".repeat(usize::try_from(decpt - n).unwrap_or(0))
            )
        } else {
            let (a, b) = digits.split_at(usize::try_from(decpt).unwrap_or(0));
            format!("{a}.{b}")
        }
    } else {
        let e = decpt - 1;
        let sign_e = if e < 0 { '-' } else { '+' };
        let first: String = digits.chars().take(1).collect();
        let rest: String = digits.chars().skip(1).collect();
        let m = if rest.is_empty() {
            first
        } else {
            format!("{first}.{rest}")
        };
        format!("{m}e{sign_e}{:02}", e.abs())
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(v: &StateValue) -> String {
        render_checked(v, 1 << 20, false).unwrap()
    }

    #[test]
    fn float_vectors_from_the_spec() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-0.0, "-0.0"),
            (100.0, "100.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (2.5e-5, "2.5e-05"),
            (123456789.123, "123456789.123"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e21, "1e+21"),
            (1.5e300, "1.5e+300"),
            (12345678901234567890.0, "1.2345678901234567e+19"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (9999999999999998.0, "9999999999999998.0"),
            (0.0, "0.0"),
            (12.5, "12.5"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(*x), *want, "for {x:e}");
        }
    }

    #[test]
    fn spec_example_nested_state() {
        let v = StateValue::object([
            (
                "order",
                StateValue::object([
                    ("id", StateValue::int(1042)),
                    (
                        "charges",
                        StateValue::list([StateValue::Float(12.5), StateValue::Float(12.5)]),
                    ),
                    ("note", StateValue::Null),
                    ("paid", StateValue::Bool(true)),
                ]),
            ),
            ("tags", StateValue::list([])),
            (
                "history",
                StateValue::list([
                    StateValue::object([
                        ("at", StateValue::string("mon")),
                        ("ev", StateValue::string("shipped")),
                    ]),
                    StateValue::string("late"),
                ]),
            ),
        ]);
        assert_eq!(
            text(&v),
            "order:\n  id: 1042\n  charges:\n    - 12.5\n    - 12.5\n  note: \n  paid: True\ntags:\n\nhistory:\n  - at: mon\n    ev: shipped\n  - late"
        );
    }

    #[test]
    fn spec_example_list_lstrip() {
        let v = StateValue::list([
            StateValue::string("  a"),
            StateValue::string("\u{1c}\u{1d}b"),
            StateValue::string("\n\nc"),
            StateValue::string(""),
            StateValue::Null,
            StateValue::list([StateValue::int(1), StateValue::list([StateValue::int(2)])]),
        ]);
        assert_eq!(text(&v), "- a\n- b\n- c\n- \n- \n- - 1\n  - - 2");
    }

    #[test]
    fn lstrip_does_not_touch_zwsp_bom_or_root_strings() {
        let v = StateValue::list([
            StateValue::string("\u{200b}x"),
            StateValue::string("\u{feff}y"),
        ]);
        assert_eq!(text(&v), "- \u{200b}x\n- \u{feff}y");
        assert_eq!(text(&StateValue::string("  root  ")), "  root  ");
    }

    #[test]
    fn lstrip_removes_exactly_the_29_spaces() {
        let set: Vec<char> = (0u32..=0x3100)
            .filter_map(char::from_u32)
            .filter(|c| is_py_space(*c))
            .collect();
        assert_eq!(set.len(), 29);
        for c in set {
            let v = StateValue::list([StateValue::string(format!("{c}x"))]);
            assert_eq!(text(&v), "- x", "U+{:04X}", c as u32);
        }
    }

    #[test]
    fn first_key_of_object_in_list_is_stripped_other_keys_are_not() {
        let v = StateValue::list([StateValue::object([
            ("\u{a0}a", StateValue::string("1")),
            ("\u{a0}b", StateValue::string("2")),
        ])]);
        assert_eq!(text(&v), "- a: 1\n  \u{a0}b: 2");
    }

    #[test]
    fn object_values_are_not_stripped() {
        let v = StateValue::object([("k", StateValue::string("  v"))]);
        assert_eq!(text(&v), "k:   v");
    }

    #[test]
    fn empty_container_value_gives_blank_line() {
        let v = StateValue::object([
            ("a", StateValue::object::<String>([])),
            ("b", StateValue::Null),
        ]);
        assert_eq!(text(&v), "a:\n\nb: ");
    }

    #[test]
    fn multiline_strings_are_not_indented() {
        let v = StateValue::object([("k", StateValue::string("l1\nl2"))]);
        assert_eq!(text(&v), "k: l1\nl2");
        let l = StateValue::list([StateValue::string("l1\nl2")]);
        assert_eq!(text(&l), "- l1\nl2");
    }

    #[test]
    fn big_integers_and_bools() {
        let v = StateValue::list([
            StateValue::int_text("12345678901234567890123").unwrap(),
            StateValue::Bool(false),
        ]);
        assert_eq!(text(&v), "- 12345678901234567890123\n- False");
    }

    #[test]
    fn stops_at_the_limit() {
        let v = StateValue::string("x".repeat(1000));
        assert_eq!(render_checked(&v, 999, false), Err(Error::StateTooLarge));
        assert!(render_checked(&v, 1000, false).is_ok());
    }
}
