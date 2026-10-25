//! Cooperative control of a running request (spec 008): the work asks "may I go on?" between
//! rows, never in the middle of one. Pure Rust: the Python binding supplies an implementation
//! that also looks at `Ctrl-C`.

use crate::error::{Error, Result};
use crate::hub::events::Cancel;

/// Asked between the rows of a request and while waiting for a slot of the pool.
///
/// An implementation may take its time (the Python one briefly takes the GIL), so it must not be
/// called with a lock of this crate in hand.
pub trait Control: Send + Sync {
    /// `Ok(())` to go on, `Err(Error::Cancelled)` to stop with no partial result.
    ///
    /// # Errors
    /// [`Error::Cancelled`] when the request must stop.
    fn checkpoint(&self) -> Result<()>;
}

/// Never stops: the behaviour of every call that does not ask for control.
pub struct NoControl;

impl Control for NoControl {
    fn checkpoint(&self) -> Result<()> {
        Ok(())
    }
}

/// Stops when a [`Cancel`] (a flag another thread sets) says so.
pub struct CancelControl<'a>(pub &'a dyn Cancel);

impl Control for CancelControl<'_> {
    fn checkpoint(&self) -> Result<()> {
        if self.0.cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::events::CancelFlag;

    #[test]
    fn no_control_never_stops() {
        assert!(NoControl.checkpoint().is_ok());
    }

    #[test]
    fn cancel_control_follows_the_flag() {
        let flag = CancelFlag::default();
        let control = CancelControl(&flag);
        assert!(control.checkpoint().is_ok());
        flag.cancel();
        assert!(matches!(control.checkpoint(), Err(Error::Cancelled)));
    }

    /// The core is pure Rust (principle 1): only the binding in `src/python/` may name PyO3.
    #[test]
    fn the_core_does_not_name_pyo3() {
        fn walk(dir: &std::path::Path, hits: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|n| n == "python") {
                        continue;
                    }
                    walk(&path, hits);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    for needle in [
                        "use pyo3",
                        "pyo3::",
                        "extern crate pyo3",
                        "#[pyclass",
                        "#[pyfunction",
                        "#[pymethods",
                    ] {
                        // this very test names the needles
                        if path.ends_with("control.rs") {
                            continue;
                        }
                        if text.contains(needle) {
                            hits.push(format!("{} ({needle})", path.display()));
                        }
                    }
                }
            }
        }
        let mut hits = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut hits,
        );
        assert!(hits.is_empty(), "PyO3 outside src/python: {hits:?}");
    }
}
