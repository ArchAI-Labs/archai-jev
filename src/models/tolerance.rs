//! Tolerances of the self-check, per (family, dtype). They live in the library, never in a
//! manifest: an author must not be able to widen the check (spec 005, 7.3).
//!
//! Values come from D25 (Kev on llama.cpp, confirmed) and D29 (default model, confirmed with
//! spec 005). The self-check compares the prompt token ids **exactly**; these numbers apply to
//! what is left. Changing one is a decision and a CHANGELOG entry, never a way to make a test pass.

/// How far the engine may be from the expected numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Largest accepted `|delta logit|`; `None` where the family compares probabilities only.
    pub logit: Option<f64>,
    /// Largest accepted `|delta p|`.
    pub p: f64,
}

/// The tolerance for `family` with weights of type `dtype`, if one is defined.
pub fn lookup(family: &str, dtype: &str) -> Option<Tolerance> {
    let t = match (family, dtype) {
        ("qwen35-pointer", "f32" | "bf16") => Tolerance {
            logit: Some(6e-2),
            p: 1.5e-2,
        },
        ("qwen35-pointer", "q8_0") => Tolerance {
            logit: Some(0.25),
            p: 7e-2,
        },
        ("qwen2-letters", "f32" | "bf16") => Tolerance {
            logit: None,
            p: 1.5e-2,
        },
        ("qwen2-letters", "q8_0") => Tolerance {
            logit: None,
            p: 0.2,
        },
        #[cfg(feature = "testing")]
        ("test-fake" | "test-kev", "f32" | "q8_0") => Tolerance {
            logit: Some(1e-3),
            p: 1e-3,
        },
        _ => return None,
    };
    Some(t)
}

/// Every `(family, dtype)` that has a tolerance (used by tests that walk the whole table).
pub fn all() -> Vec<(&'static str, &'static str)> {
    #[allow(unused_mut)]
    let mut rows = vec![
        ("qwen35-pointer", "f32"),
        ("qwen35-pointer", "bf16"),
        ("qwen35-pointer", "q8_0"),
        ("qwen2-letters", "f32"),
        ("qwen2-letters", "bf16"),
        ("qwen2-letters", "q8_0"),
    ];
    #[cfg(feature = "testing")]
    rows.extend([
        ("test-fake", "f32"),
        ("test-fake", "q8_0"),
        ("test-kev", "f32"),
        ("test-kev", "q8_0"),
    ]);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_matches_the_decisions() {
        assert_eq!(
            lookup("qwen35-pointer", "bf16"),
            Some(Tolerance {
                logit: Some(0.06),
                p: 0.015
            })
        );
        assert_eq!(
            lookup("qwen2-letters", "q8_0"),
            Some(Tolerance {
                logit: None,
                p: 0.2
            })
        );
        assert_eq!(lookup("qwen2-letters", "q4_k_m"), None);
        assert_eq!(lookup("nope", "f32"), None);
    }

    #[test]
    fn every_listed_row_resolves() {
        for (f, d) in all() {
            assert!(lookup(f, d).is_some(), "{f} {d}");
        }
    }
}
