//! The CPU the engine needs (spec 006b section 7).
//!
//! The wheel is built for x86_64 with AVX2, FMA, F16C and BMI2 (`packaging.md`, measured by the
//! `llama-probe` workflow). On a CPU without them the process would die with "illegal
//! instruction"; this check turns that into an error that names the missing feature. The detector
//! is injectable so the test does not need such a CPU.

use crate::error::{Error, Result};

/// The features the wheel assumes on x86_64.
pub const REQUIRED: [&str; 4] = ["avx2", "fma", "f16c", "bmi2"];

/// Answers whether the CPU has a feature.
pub trait CpuFeatures {
    /// True if `name` (lowercase, as in [`REQUIRED`]) is available.
    fn has(&self, name: &str) -> bool;
}

/// The CPU this process runs on.
pub struct HostCpu;

impl CpuFeatures for HostCpu {
    #[cfg(target_arch = "x86_64")]
    fn has(&self, name: &str) -> bool {
        match name {
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            "f16c" => std::arch::is_x86_feature_detected!("f16c"),
            "bmi2" => std::arch::is_x86_feature_detected!("bmi2"),
            _ => true,
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn has(&self, _name: &str) -> bool {
        // ARM builds use their own baseline (NEON is always there on aarch64).
        true
    }
}

/// Refuse to start the engine on a CPU that lacks a required feature.
///
/// # Errors
/// [`Error::Inference`] naming the first missing feature.
pub fn check(cpu: &dyn CpuFeatures) -> Result<()> {
    for feature in REQUIRED {
        if !cpu.has(feature) {
            return Err(Error::Inference {
                detail: format!(
                    "this CPU does not have {}, which the inference engine needs (it needs AVX2, FMA, F16C and BMI2, i.e. an Intel or AMD CPU from about 2013-2015 or later); the engine was not started",
                    feature.to_uppercase()
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Without(&'static str);
    impl CpuFeatures for Without {
        fn has(&self, name: &str) -> bool {
            name != self.0
        }
    }

    #[test]
    fn each_missing_feature_is_named() {
        for f in REQUIRED {
            let err = check(&Without(f)).unwrap_err();
            assert!(err.to_string().contains(&f.to_uppercase()), "{err}");
        }
        assert!(check(&Without("none")).is_ok());
    }

    #[test]
    fn this_machine_passes() {
        // The development and CI machines all have the baseline.
        assert!(check(&HostCpu).is_ok());
    }
}
