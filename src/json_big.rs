//! A strict JSON reader that keeps integers of any size (spec 017 §3.2).
//!
//! [`json_strict::parse`](crate::json_strict::parse) rests on `serde_json`, which turns an
//! integer beyond 64 bits into a float without saying so. A state may legitimately contain such
//! a number (an identifier, a counter), and silently rounding it would change the text the model
//! reads, so the dataset reader uses this parser instead: the same rules (RFC 8259, no BOM,
//! duplicate keys refused, nesting limited, finite numbers) and [`Json::BigInt`] for an integer
//! that does not fit `i128`. The manifests keep using `parse`.

use crate::json_strict::{Json, JsonProblem, MAX_BYTES, MAX_DEPTH};

/// Parse one JSON document.
///
/// # Errors
/// [`JsonProblem`]: invalid syntax (with the byte column), a byte-order mark, more than 8 MiB,
/// nesting deeper than 16 levels, a duplicate key, or a number that is not finite.
pub fn parse_with_big_ints(bytes: &[u8]) -> Result<Json, JsonProblem> {
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
    let text = std::str::from_utf8(bytes).map_err(|e| {
        JsonProblem::Syntax(format!(
            "invalid UTF-8 at column {}",
            e.valid_up_to().saturating_add(1)
        ))
    })?;
    let mut parser = Parser { text, i: 0 };
    parser.skip_space();
    let value = parser.value(0)?;
    parser.skip_space();
    if parser.i < text.len() {
        return Err(parser.fail("trailing characters after the value"));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    i: usize,
}

type Res<T> = Result<T, JsonProblem>;

impl Parser<'_> {
    fn fail(&self, what: &str) -> JsonProblem {
        JsonProblem::Syntax(format!(
            "invalid JSON: {what} at column {}",
            self.i.saturating_add(1)
        ))
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.i).copied()
    }

    fn bump(&mut self) {
        self.i += 1;
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.bump();
        }
    }

    fn expect(&mut self, byte: u8, what: &str) -> Res<()> {
        if self.peek() == Some(byte) {
            self.bump();
            Ok(())
        } else {
            Err(self.fail(what))
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Res<Json> {
        if self
            .text
            .get(self.i..)
            .is_some_and(|rest| rest.starts_with(word))
        {
            self.i += word.len();
            Ok(value)
        } else {
            Err(self.fail("expected a value"))
        }
    }

    fn value(&mut self, depth: usize) -> Res<Json> {
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.fail("expected a value")),
        }
    }

    fn enter(&self, depth: usize) -> Res<()> {
        if depth + 1 > MAX_DEPTH {
            Err(JsonProblem::Limit(format!(
                "nesting is deeper than {MAX_DEPTH} levels"
            )))
        } else {
            Ok(())
        }
    }

    fn array(&mut self, depth: usize) -> Res<Json> {
        self.enter(depth)?;
        self.bump();
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_space();
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.bump(),
                Some(b']') => {
                    self.bump();
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.fail("expected ',' or ']'")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Res<Json> {
        self.enter(depth)?;
        self.bump();
        let mut pairs: Vec<(String, Json)> = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(Json::Object(pairs));
        }
        loop {
            self.skip_space();
            if self.peek() != Some(b'"') {
                return Err(self.fail("expected a string key"));
            }
            let key = self.string()?;
            self.skip_space();
            self.expect(b':', "expected ':'")?;
            self.skip_space();
            let value = self.value(depth + 1)?;
            if pairs.iter().any(|(k, _)| *k == key) {
                return Err(JsonProblem::DuplicateKey(key));
            }
            pairs.push((key, value));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.bump(),
                Some(b'}') => {
                    self.bump();
                    return Ok(Json::Object(pairs));
                }
                _ => return Err(self.fail("expected ',' or '}'")),
            }
        }
    }

    fn hex4(&mut self) -> Res<u32> {
        let mut code = 0u32;
        for _ in 0..4 {
            let digit = self
                .peek()
                .and_then(|b| char::from(b).to_digit(16))
                .ok_or_else(|| self.fail("expected four hexadecimal digits after \\u"))?;
            code = code * 16 + digit;
            self.bump();
        }
        Ok(code)
    }

    fn string(&mut self) -> Res<String> {
        self.bump();
        let mut out = String::new();
        let mut run = self.i;
        loop {
            match self.peek() {
                None => return Err(self.fail("unterminated string")),
                Some(b'"') => {
                    out.push_str(self.text.get(run..self.i).unwrap_or(""));
                    self.bump();
                    return Ok(out);
                }
                Some(b'\\') => {
                    out.push_str(self.text.get(run..self.i).unwrap_or(""));
                    self.bump();
                    let escaped = self
                        .peek()
                        .ok_or_else(|| self.fail("unterminated string"))?;
                    self.bump();
                    match escaped {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        _ => {
                            self.i -= 1;
                            return Err(self.fail("invalid escape"));
                        }
                    }
                    run = self.i;
                }
                Some(b) if b < 0x20 => {
                    return Err(self.fail("control character in a string"));
                }
                Some(_) => self.bump(),
            }
        }
    }

    /// The character of `\uXXXX`, joining a surrogate pair `\uD83D\uDE00`.
    fn unicode_escape(&mut self) -> Res<char> {
        let first = self.hex4()?;
        match first {
            0xD800..=0xDBFF => {
                let has_pair = self
                    .text
                    .get(self.i..)
                    .is_some_and(|rest| rest.starts_with("\\u"));
                if !has_pair {
                    return Err(self.fail("lone leading surrogate in a \\u escape"));
                }
                self.i += 2;
                let second = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&second) {
                    return Err(self.fail("lone leading surrogate in a \\u escape"));
                }
                let code = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                char::from_u32(code).ok_or_else(|| self.fail("invalid \\u escape"))
            }
            0xDC00..=0xDFFF => Err(self.fail("lone trailing surrogate in a \\u escape")),
            _ => char::from_u32(first).ok_or_else(|| self.fail("invalid \\u escape")),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.bump();
        }
        self.i - start
    }

    fn number(&mut self) -> Res<Json> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        match self.peek() {
            Some(b'0') => self.bump(),
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.fail("expected a digit")),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.bump();
            if self.digits() == 0 {
                return Err(self.fail("expected a digit after the decimal point"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            if self.digits() == 0 {
                return Err(self.fail("expected a digit in the exponent"));
            }
        }
        let literal = self.text.get(start..self.i).unwrap_or("");
        if is_float {
            return match literal.parse::<f64>() {
                Ok(f) if f.is_finite() => Ok(Json::Float(f)),
                _ => Err(JsonProblem::Syntax(format!(
                    "invalid JSON: number out of range at column {}",
                    start + 1
                ))),
            };
        }
        Ok(match literal.parse::<i128>() {
            Ok(n) => Json::Int(n),
            Err(_) => Json::BigInt(literal.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn ok(text: &str) -> Json {
        parse_with_big_ints(text.as_bytes()).unwrap()
    }

    fn err(text: &str) -> JsonProblem {
        parse_with_big_ints(text.as_bytes()).unwrap_err()
    }

    #[test]
    fn integers_beyond_64_bits_are_kept() {
        assert_eq!(
            ok("18446744073709551616"),
            Json::Int(18_446_744_073_709_551_616)
        );
        assert_eq!(
            ok("-9223372036854775809"),
            Json::Int(-9_223_372_036_854_775_809)
        );
        assert_eq!(ok("0"), Json::Int(0));
        assert_eq!(ok("-0"), Json::Int(0));
    }

    #[test]
    fn beyond_i128_is_bigint_text() {
        let digits = "9".repeat(60);
        assert_eq!(ok(&digits), Json::BigInt(digits.clone()));
        let negative = format!("-{digits}");
        assert_eq!(ok(&negative), Json::BigInt(negative.clone()));
        let inside = ok(&format!("{{\"state\": {digits}}}"));
        assert_eq!(inside.get("state"), Some(&Json::BigInt(digits)));
    }

    #[test]
    fn the_basics() {
        assert_eq!(ok(" null "), Json::Null);
        assert_eq!(ok("[true,false,null,1,2.5,\"x\"]").kind(), "an array");
        assert_eq!(ok("1e2"), Json::Float(100.0));
        assert_eq!(ok("-1.5E-1"), Json::Float(-0.15));
        assert_eq!(
            ok("{\"b\":1,\"a\":[]}"),
            Json::Object(vec![
                ("b".to_string(), Json::Int(1)),
                ("a".to_string(), Json::Array(vec![]))
            ])
        );
    }

    #[test]
    fn nan_infinity_trailing_comma_duplicate_key_truncated() {
        for bad in [
            "NaN",
            "Infinity",
            "-Infinity",
            "[1,]",
            "{\"a\":1,}",
            "[1 2]",
            "{\"a\" 1}",
            "{\"a\":",
            "[",
            "{\"a\":1",
            "\"abc",
            "tru",
            "01",
            "1.",
            ".5",
            "1e",
            "+1",
            "--1",
            "",
            "  ",
            "1 2",
            "[1]]",
            "'a'",
            "{a:1}",
        ] {
            assert!(
                matches!(err(bad), JsonProblem::Syntax(_)),
                "{bad:?} gave {:?}",
                err(bad)
            );
        }
        assert_eq!(
            err("{\"a\":1,\"a\":2}"),
            JsonProblem::DuplicateKey("a".to_string())
        );
        assert_eq!(
            err("{\"x\":{\"a\":1,\"b\":{\"a\":1,\"a\":2}}}"),
            JsonProblem::DuplicateKey("a".to_string())
        );
        assert!(matches!(err("1e999"), JsonProblem::Syntax(m) if m.contains("out of range")));
    }

    #[test]
    fn columns_are_reported() {
        assert_eq!(
            err("[1, x]"),
            JsonProblem::Syntax("invalid JSON: expected a value at column 5".to_string())
        );
    }

    #[test]
    fn escapes_and_surrogates() {
        assert_eq!(
            ok("\"a\\n\\t\\\"\\\\\\/\\b\\f\\r\\u0041\""),
            Json::Str("a\n\t\"\\/\u{8}\u{c}\rA".to_string())
        );
        assert_eq!(ok("\"\\ud83d\\ude00\""), Json::Str("\u{1F600}".to_string()));
        assert_eq!(ok("\"é — 日本\""), Json::Str("é — 日本".to_string()));
        for bad in [
            "\"\\ud83d\"",
            "\"\\ude00\"",
            "\"\\ud83dx\"",
            "\"\\u12\"",
            "\"\\q\"",
            "\"a\nb\"",
        ] {
            assert!(matches!(err(bad), JsonProblem::Syntax(_)), "{bad:?}");
        }
    }

    #[test]
    fn limits() {
        let deep_ok = format!("{}{}", "[".repeat(16), "]".repeat(16));
        assert!(parse_with_big_ints(deep_ok.as_bytes()).is_ok());
        let too_deep = format!("{}{}", "[".repeat(17), "]".repeat(17));
        assert!(matches!(err(&too_deep), JsonProblem::Limit(_)));
        let mut bom = vec![0xEF, 0xBB, 0xBF];
        bom.extend_from_slice(b"{}");
        assert!(matches!(
            parse_with_big_ints(&bom),
            Err(JsonProblem::Limit(m)) if m.contains("byte order mark")
        ));
        assert!(matches!(
            parse_with_big_ints(&[b'"', 0xFF, b'"']),
            Err(JsonProblem::Syntax(m)) if m.contains("invalid UTF-8")
        ));
    }

    fn json_strategy() -> impl Strategy<Value = serde_json::Value> {
        let leaf = prop_oneof![
            Just(serde_json::Value::Null),
            any::<bool>().prop_map(serde_json::Value::Bool),
            any::<i64>().prop_map(serde_json::Value::from),
            any::<u64>().prop_map(serde_json::Value::from),
            (-1e12f64..1e12).prop_map(serde_json::Value::from),
            "\\PC{0,12}".prop_map(serde_json::Value::String),
        ];
        leaf.prop_recursive(4, 40, 5, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..5).prop_map(serde_json::Value::Array),
                prop::collection::btree_map("\\PC{0,6}", inner, 0..5)
                    .prop_map(|m| serde_json::Value::Object(m.into_iter().collect())),
            ]
        })
    }

    fn to_json(v: &serde_json::Value) -> Json {
        match v {
            serde_json::Value::Null => Json::Null,
            serde_json::Value::Bool(b) => Json::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Json::Int(i128::from(i))
                } else if let Some(u) = n.as_u64() {
                    Json::Int(i128::from(u))
                } else {
                    Json::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => Json::Str(s.clone()),
            serde_json::Value::Array(a) => Json::Array(a.iter().map(to_json).collect()),
            serde_json::Value::Object(o) => {
                Json::Object(o.iter().map(|(k, v)| (k.clone(), to_json(v))).collect())
            }
        }
    }

    proptest! {
        #[test]
        fn same_tree_as_serde_json(value in json_strategy()) {
            for text in [value.to_string(), serde_json::to_string_pretty(&value).unwrap()] {
                let expected = crate::json_strict::parse(text.as_bytes());
                let got = parse_with_big_ints(text.as_bytes());
                match (expected, got) {
                    (Ok(a), Ok(b)) => prop_assert!(same_up_to_float_rounding(&a, &b), "{a:?} vs {b:?}"),
                    (Err(_), Err(_)) => {}
                    (Ok(_), Err(e)) => prop_assert!(false, "serde accepts, we refuse: {e:?} for {text}"),
                    (Err(e), Ok(_)) => prop_assert!(false, "we accept, serde refuses: {e:?} for {text}"),
                }
                // and the tree is the one of the value that produced the text
                prop_assert_eq!(to_json_compare(&to_json(&value)), to_json_compare(&parse_with_big_ints(text.as_bytes()).unwrap()));
            }
        }

        #[test]
        fn refuses_what_serde_json_refuses(bytes in prop::collection::vec(prop::sample::select(
            b" \t\n\r{}[]\",:0123456789-+.eEtrufalsn\\u/ab".to_vec()), 0..40)) {
            let ours = parse_with_big_ints(&bytes);
            let theirs = crate::json_strict::parse(&bytes);
            // Big integers are the one thing we accept and they do not: compare only when the
            // text has no integer of more than 18 digits.
            let long_digits = bytes.windows(19).any(|w| w.iter().all(u8::is_ascii_digit));
            if !long_digits {
                prop_assert_eq!(ours.is_ok(), theirs.is_ok(), "{:?}", String::from_utf8_lossy(&bytes));
                if let (Ok(a), Ok(b)) = (ours, theirs) {
                    prop_assert!(same_up_to_float_rounding(&a, &b));
                }
            }
        }

        #[test]
        fn never_panics_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..200)) {
            let _ = parse_with_big_ints(&bytes);
        }
    }

    /// `serde_json` without `float_roundtrip` can be one unit in the last place off for some
    /// decimals; `str::parse::<f64>`, which this parser uses, is correctly rounded. Everything
    /// else must be identical.
    fn same_up_to_float_rounding(a: &Json, b: &Json) -> bool {
        match (a, b) {
            (Json::Float(x), Json::Float(y)) => (x - y).abs() <= 1e-15 * x.abs().max(y.abs()),
            (Json::Array(x), Json::Array(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y)
                        .all(|(p, q)| same_up_to_float_rounding(p, q))
            }
            (Json::Object(x), Json::Object(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y)
                        .all(|((k, p), (l, q))| k == l && same_up_to_float_rounding(p, q))
            }
            _ => a == b,
        }
    }

    fn to_json_compare(j: &Json) -> String {
        j.to_canonical_string()
    }
}
