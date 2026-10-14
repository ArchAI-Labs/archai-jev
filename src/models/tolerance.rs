//! Tolerances of the self-check, per (family, dtype, reference). They live in the library, never
//! in a manifest: an author must not be able to widen the check (spec 005, 7.3).
//!
//! Values come from D25 (Kev on llama.cpp, confirmed) and D29 (default model, confirmed with
//! spec 005). The self-check compares the prompt token ids **exactly**; these numbers apply to
//! what is left. Changing one is a decision and a CHANGELOG entry, never a way to make a test pass.

/// Where the expected numbers of a variant's vectors come from (spec 009, section 4).
///
/// The manifest chooses it with the optional key `selfcheck.reference`; the tolerance it gets
/// is a row of the library's table, so choosing `Engine` never widens anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reference {
    /// The reference implementation at full precision (PyTorch fp32): the rule of spec 005.
    #[default]
    FullPrecision,
    /// Recorded by this engine on the exact file of the variant: a quantised variant is checked
    /// against itself with a tight tolerance; its distance from full precision is measured when
    /// the variant is registered, not at every load.
    Engine,
}

impl Reference {
    /// The name used in manifests and messages.
    pub fn name(self) -> &'static str {
        match self {
            Reference::FullPrecision => "full_precision",
            Reference::Engine => "engine",
        }
    }

    /// The reference called `name`, if there is one.
    pub fn parse(name: &str) -> Option<Reference> {
        match name {
            "full_precision" => Some(Reference::FullPrecision),
            "engine" => Some(Reference::Engine),
            _ => None,
        }
    }
}

/// How far the engine may be from the expected numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Largest accepted `|delta logit|`; `None` where the family compares probabilities only.
    pub logit: Option<f64>,
    /// Largest accepted `|delta p|`.
    pub p: f64,
}

/// The tolerance for `family` with weights of type `dtype` checked against `reference`, if one
/// is defined.
pub fn lookup(family: &str, dtype: &str, reference: Reference) -> Option<Tolerance> {
    #[cfg(feature = "testing")]
    use Reference::Engine;
    use Reference::FullPrecision;
    let t = match (family, dtype, reference) {
        ("qwen35-pointer", "f32" | "bf16", FullPrecision) => Tolerance {
            logit: Some(6e-2),
            p: 1.5e-2,
        },
        ("qwen35-pointer", "q8_0", FullPrecision) => Tolerance {
            logit: Some(0.25),
            p: 7e-2,
        },
        // spec 010: provisional, the same as the letters family until the spike S11 measures them
        ("qwen2-keys", "f32" | "bf16", FullPrecision) => Tolerance {
            logit: None,
            p: 1.5e-2,
        },
        ("qwen2-keys", "q8_0", FullPrecision) => Tolerance {
            logit: None,
            p: 0.2,
        },
        ("qwen2-letters", "f32" | "bf16", FullPrecision) => Tolerance {
            logit: None,
            p: 1.5e-2,
        },
        ("qwen2-letters", "q8_0", FullPrecision) => Tolerance {
            logit: None,
            p: 0.2,
        },
        #[cfg(feature = "testing")]
        ("test-fake" | "test-kev", "f32" | "q8_0", FullPrecision) => Tolerance {
            logit: Some(1e-3),
            p: 1e-3,
        },
        // study rows (spec 009): refuse nothing, so that a K-quant can be loaded and MEASURED;
        // never in a release build
        #[cfg(feature = "quant-study")]
        ("qwen2-letters" | "qwen2-keys", "q6_k" | "q5_k_m" | "q4_k_m", FullPrecision) => {
            Tolerance {
                logit: None,
                p: 1.0,
            }
        }
        // a tight row for the engine-recorded rule (spec 009 section 4), to test the mechanism
        #[cfg(feature = "testing")]
        ("test-fake", "f32", Engine) => Tolerance {
            logit: None,
            p: 1e-6,
        },
        _ => return None,
    };
    Some(t)
}

/// Every `(family, dtype, reference)` that has a tolerance (used by tests that walk the table).
pub fn all() -> Vec<(&'static str, &'static str, Reference)> {
    #[allow(unused_mut)]
    let mut rows = vec![
        ("qwen35-pointer", "f32", Reference::FullPrecision),
        ("qwen35-pointer", "bf16", Reference::FullPrecision),
        ("qwen35-pointer", "q8_0", Reference::FullPrecision),
        ("qwen2-letters", "f32", Reference::FullPrecision),
        ("qwen2-letters", "bf16", Reference::FullPrecision),
        ("qwen2-letters", "q8_0", Reference::FullPrecision),
        ("qwen2-keys", "f32", Reference::FullPrecision),
        ("qwen2-keys", "bf16", Reference::FullPrecision),
        ("qwen2-keys", "q8_0", Reference::FullPrecision),
    ];
    #[cfg(feature = "testing")]
    rows.extend([
        ("test-fake", "f32", Reference::FullPrecision),
        ("test-fake", "q8_0", Reference::FullPrecision),
        ("test-kev", "f32", Reference::FullPrecision),
        ("test-kev", "q8_0", Reference::FullPrecision),
        ("test-fake", "f32", Reference::Engine),
    ]);
    #[cfg(feature = "quant-study")]
    rows.extend([
        ("qwen2-letters", "q6_k", Reference::FullPrecision),
        ("qwen2-letters", "q5_k_m", Reference::FullPrecision),
        ("qwen2-letters", "q4_k_m", Reference::FullPrecision),
        ("qwen2-keys", "q6_k", Reference::FullPrecision),
        ("qwen2-keys", "q5_k_m", Reference::FullPrecision),
        ("qwen2-keys", "q4_k_m", Reference::FullPrecision),
    ]);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: Reference = Reference::FullPrecision;

    #[test]
    fn the_table_matches_the_decisions() {
        assert_eq!(
            lookup("qwen35-pointer", "bf16", FP),
            Some(Tolerance {
                logit: Some(0.06),
                p: 0.015
            })
        );
        assert_eq!(
            lookup("qwen2-letters", "q8_0", FP),
            Some(Tolerance {
                logit: None,
                p: 0.2
            })
        );
        // a K-quant has a row only behind the study feature, and then one that refuses nothing
        #[cfg(not(feature = "quant-study"))]
        assert_eq!(lookup("qwen2-letters", "q4_k_m", FP), None);
        #[cfg(feature = "quant-study")]
        assert_eq!(
            lookup("qwen2-letters", "q4_k_m", FP).map(|t| t.p),
            Some(1.0)
        );
        assert_eq!(lookup("qwen2-letters", "q2_k", FP), None);
        assert_eq!(lookup("nope", "f32", FP), None);
    }

    #[test]
    fn every_listed_row_resolves() {
        for (f, d, r) in all() {
            assert!(lookup(f, d, r).is_some(), "{f} {d} {}", r.name());
        }
    }

    #[test]
    fn a_reference_is_part_of_the_key() {
        // the same (family, dtype) has a loose row for the full-precision rule and a tight one
        // for the engine-recorded rule; neither answers for the other
        let loose = lookup("test-fake", "f32", FP).unwrap();
        let tight = lookup("test-fake", "f32", Reference::Engine).unwrap();
        assert!(tight.p < loose.p);
        assert_eq!(lookup("qwen2-letters", "q8_0", Reference::Engine), None);
        assert_eq!(lookup("qwen35-pointer", "bf16", Reference::Engine), None);
    }

    #[test]
    fn engine_rows_are_tight() {
        // spec 009 section 4: against its own recorded numbers a variant gets at most 0.05;
        // a variant that would need more is not admitted
        for (f, d, r) in all() {
            if r == Reference::Engine {
                let t = lookup(f, d, r).unwrap();
                assert!(t.p <= 0.05, "{f} {d}: p = {}", t.p);
            }
        }
    }

    #[test]
    fn reference_names_round_trip() {
        for r in [Reference::FullPrecision, Reference::Engine] {
            assert_eq!(Reference::parse(r.name()), Some(r));
        }
        assert_eq!(Reference::parse("Engine"), None);
        assert_eq!(Reference::parse(""), None);
        assert_eq!(Reference::default(), Reference::FullPrecision);
    }
}

#[cfg(test)]
mod name_tests {
    use super::*;
    use crate::models::recipes;

    #[test]
    fn every_name_has_a_row_and_every_real_row_has_a_name() {
        // spec 009 AC-1: the dtypes a release accepts and the dtypes it can verify are the
        // same set (for the real families; the test families are not names users can pick)
        for family in ["qwen2-letters", "qwen2-keys"] {
            for d in recipes::DTYPES {
                assert!(
                    lookup(family, d, Reference::FullPrecision).is_some(),
                    "{family} {d} has a name but no tolerance"
                );
            }
        }
        for (f, d, _) in all() {
            if f == "qwen2-letters" || f == "qwen2-keys" || f == "qwen35-pointer" {
                assert!(
                    recipes::DTYPES.contains(&d),
                    "{f} {d} has a row but no name"
                );
            }
        }
    }
}
