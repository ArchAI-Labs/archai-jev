//! Tolerances of the self-check, per (family, dtype, reference, backend). They live in the library,
//! never in a manifest: an author must not be able to widen the check (spec 005, 7.3).
//!
//! The backend is the fourth key (spec 011, section 3 rule 3): every row written before GPUs
//! existed is a `Cpu` row and did not change by a digit. A GPU row is added **by decision**, after
//! the fidelity and calibration of that backend were measured, and is never wider than the CPU row
//! of the same `(family, dtype, reference)`; a model with no row on a backend is not allowed to
//! answer there.
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

use crate::engine::device::BackendKind;

/// How far the engine may be from the expected numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Largest accepted `|delta logit|`; `None` where the family compares probabilities only.
    pub logit: Option<f64>,
    /// Largest accepted `|delta p|`.
    pub p: f64,
}

impl Tolerance {
    /// True if `self` accepts nothing that `other` would refuse: its `p` is not larger and, where
    /// `other` bounds the logits, so does `self` with a bound that is not larger. A row that
    /// leaves the logits free where the other bounds them is wider.
    pub fn is_no_wider_than(&self, other: &Tolerance) -> bool {
        let logit_ok = match (self.logit, other.logit) {
            (_, None) => true,
            (Some(mine), Some(theirs)) => mine <= theirs,
            (None, Some(_)) => false,
        };
        self.p <= other.p && logit_ok
    }
}

/// The tolerance for `family` with weights of type `dtype` checked against `reference` on
/// `backend`, if one is defined. `None` means the model is not verified there.
pub fn lookup(
    family: &str,
    dtype: &str,
    reference: Reference,
    backend: BackendKind,
) -> Option<Tolerance> {
    match backend {
        BackendKind::Cpu => cpu_row(family, dtype, reference),
        gpu => gpu_row(family, dtype, reference, gpu),
    }
}

/// The GPU rows. None exist yet in a release: they are added by decision after the measures of
/// spec 011 (task T12), never to make a test pass. Under `testing` there is one, to exercise the
/// machinery without a GPU.
#[allow(unused_variables)]
fn gpu_row(
    family: &str,
    dtype: &str,
    reference: Reference,
    backend: BackendKind,
) -> Option<Tolerance> {
    match (family, dtype, reference, backend) {
        #[cfg(feature = "testing")]
        ("test-fake", "f32", Reference::FullPrecision, BackendKind::Cuda) => Some(Tolerance {
            logit: Some(5e-4),
            p: 5e-4,
        }),
        _ => None,
    }
}

fn cpu_row(family: &str, dtype: &str, reference: Reference) -> Option<Tolerance> {
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
        // spec 010, spike S11 (D54): measured against the Decision Index kit in fp32 on CPU.
        // f16 weights 0.0027 (87 questions); the published Q8_0 0.0627 on 129 questions of 3 to
        // 2,400 tokens (it does not grow with the length). Q8_0 gets 1.6x the largest gap seen.
        ("qwen2-keys", "f32" | "bf16", FullPrecision) => Tolerance {
            logit: None,
            p: 1.5e-2,
        },
        ("qwen2-keys", "q8_0", FullPrecision) => Tolerance {
            logit: None,
            p: 0.1,
        },
        ("qwen2-letters", "f32" | "bf16", FullPrecision) => Tolerance {
            logit: None,
            p: 1.5e-2,
        },
        ("qwen2-letters", "q8_0", FullPrecision) => Tolerance {
            logit: None,
            p: 0.2,
        },
        // spec 010b, spike S13 (D62): the Q8_0 files on this engine against the reference runtime
        // at the same precision, on the path with cuts: largest gap 0.064 (1.7B, 108 questions)
        // and 0.048 (4B). 0.10 is 1.6x the largest, as for the keys head (D54). Only Q8_0 was
        // measured; no other precision has a row.
        ("spark2_5-letters", "q8_0", FullPrecision) => Tolerance {
            logit: None,
            p: 0.1,
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

/// Every `(family, dtype, reference, backend)` that has a tolerance (used by tests that walk the
/// table).
pub fn all() -> Vec<(&'static str, &'static str, Reference, BackendKind)> {
    #[allow(unused_mut)] // only the testing build adds a GPU row
    let mut rows: Vec<_> = cpu_rows()
        .into_iter()
        .map(|(f, d, r)| (f, d, r, BackendKind::Cpu))
        .collect();
    #[cfg(feature = "testing")]
    rows.push((
        "test-fake",
        "f32",
        Reference::FullPrecision,
        BackendKind::Cuda,
    ));
    rows
}

fn cpu_rows() -> Vec<(&'static str, &'static str, Reference)> {
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
        ("spark2_5-letters", "q8_0", Reference::FullPrecision),
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

    /// The CPU rows, which are the ones that existed before the backend became a key.
    fn lookup(family: &str, dtype: &str, reference: Reference) -> Option<Tolerance> {
        super::lookup(family, dtype, reference, BackendKind::Cpu)
    }

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
        assert_eq!(
            lookup("qwen2-keys", "q8_0", FP),
            Some(Tolerance {
                logit: None,
                p: 0.1
            })
        );
        assert_eq!(
            lookup("spark2_5-letters", "q8_0", FP),
            Some(Tolerance {
                logit: None,
                p: 0.1
            })
        );
        // only the precision that was measured has a row
        for other in ["f32", "bf16", "f16", "q4_k_m", "q6_k"] {
            assert_eq!(lookup("spark2_5-letters", other, FP), None, "{other}");
        }
        assert_eq!(lookup("spark2_5-letters", "q8_0", Reference::Engine), None);
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
        for (f, d, r, b) in all() {
            assert!(
                super::lookup(f, d, r, b).is_some(),
                "{f} {d} {} {b}",
                r.name()
            );
        }
    }

    #[test]
    fn existing_rows_are_cpu_and_unchanged() {
        // spec 011 AC-6: every row that existed before GPUs is a CPU row, with the same numbers.
        // These are the values of D25, D29, D54 and D62; the test fails if one of them moves.
        let expected: [(&str, &str, Option<f64>, f64); 10] = [
            ("qwen35-pointer", "f32", Some(0.06), 0.015),
            ("qwen35-pointer", "bf16", Some(0.06), 0.015),
            ("qwen35-pointer", "q8_0", Some(0.25), 0.07),
            ("qwen2-letters", "f32", None, 0.015),
            ("qwen2-letters", "bf16", None, 0.015),
            ("qwen2-letters", "q8_0", None, 0.2),
            ("qwen2-keys", "f32", None, 0.015),
            ("qwen2-keys", "bf16", None, 0.015),
            ("qwen2-keys", "q8_0", None, 0.1),
            ("spark2_5-letters", "q8_0", None, 0.1),
        ];
        for (family, dtype, logit, p) in expected {
            assert_eq!(
                super::lookup(family, dtype, FP, BackendKind::Cpu),
                Some(Tolerance { logit, p }),
                "{family} {dtype}"
            );
            // and none of them answers on a GPU backend: no row there, no answer there
            for gpu in BackendKind::GPU_BY_PREFERENCE {
                assert_eq!(
                    super::lookup(family, dtype, FP, gpu),
                    None,
                    "{family} {dtype} {gpu}"
                );
            }
        }
        // the real families have no GPU row at all until a decision adds one (T12)
        for (f, _, _, b) in all() {
            if b != BackendKind::Cpu {
                assert!(
                    f.starts_with("test-"),
                    "{f} has a GPU row without a decision"
                );
            }
        }
    }

    #[test]
    fn gpu_rows_are_never_wider_than_the_cpu_row() {
        let rows = all();
        assert!(
            rows.iter().any(|r| r.3 != BackendKind::Cpu),
            "the testing build carries a GPU row, so this test is not empty"
        );
        for (family, dtype, reference, backend) in rows {
            if backend == BackendKind::Cpu {
                continue;
            }
            let gpu = super::lookup(family, dtype, reference, backend).unwrap();
            let cpu = super::lookup(family, dtype, reference, BackendKind::Cpu)
                .unwrap_or_else(|| panic!("{family} {dtype} has a {backend} row but no CPU row"));
            assert!(
                gpu.is_no_wider_than(&cpu),
                "{family} {dtype} {backend}: {gpu:?} is wider than the CPU row {cpu:?}"
            );
        }
    }

    #[test]
    fn what_wider_means() {
        let t = |logit, p| Tolerance { logit, p };
        let cpu = t(Some(1e-3), 1e-3);
        assert!(
            t(Some(1e-3), 1e-3).is_no_wider_than(&cpu),
            "equal is allowed"
        );
        assert!(t(Some(5e-4), 5e-4).is_no_wider_than(&cpu));
        assert!(!t(Some(1e-3), 1.1e-3).is_no_wider_than(&cpu), "p larger");
        assert!(
            !t(Some(1.1e-3), 1e-3).is_no_wider_than(&cpu),
            "logit larger"
        );
        assert!(
            !t(None, 1e-3).is_no_wider_than(&cpu),
            "free logits where the CPU bounds them"
        );
        // where the CPU row does not bound the logits, neither choice of the GPU row is wider
        let cpu_p_only = t(None, 0.1);
        assert!(t(None, 0.1).is_no_wider_than(&cpu_p_only));
        assert!(t(Some(1.0), 0.1).is_no_wider_than(&cpu_p_only));
        assert!(!t(None, 0.100001).is_no_wider_than(&cpu_p_only));
    }

    #[test]
    fn a_backend_is_part_of_the_key() {
        // the machinery works with a GPU row: it answers on its own backend only
        let on_cuda = super::lookup("test-fake", "f32", FP, BackendKind::Cuda);
        assert!(on_cuda.is_some());
        for other in [BackendKind::Vulkan, BackendKind::Metal] {
            assert_eq!(
                super::lookup("test-fake", "f32", FP, other),
                None,
                "{other}"
            );
        }
        // a variant with no GPU row is not admitted there, even though it has a CPU row
        assert!(lookup("test-fake", "q8_0", FP).is_some());
        assert_eq!(
            super::lookup("test-fake", "q8_0", FP, BackendKind::Cuda),
            None
        );
        // and the CPU row is the loose one
        let cpu = lookup("test-fake", "f32", FP).unwrap();
        assert!(on_cuda.unwrap().p < cpu.p);
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
        for (f, d, r, b) in all() {
            if r == Reference::Engine {
                let t = super::lookup(f, d, r, b).unwrap();
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

    fn lookup(family: &str, dtype: &str, reference: Reference) -> Option<Tolerance> {
        super::lookup(family, dtype, reference, BackendKind::Cpu)
    }

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
        for (f, d, _, _) in all() {
            if f == "qwen2-letters" || f == "qwen2-keys" || f == "qwen35-pointer" {
                assert!(
                    recipes::DTYPES.contains(&d),
                    "{f} {d} has a row but no name"
                );
            }
        }
    }
}
