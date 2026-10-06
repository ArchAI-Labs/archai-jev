//! Stage 6 for pointer heads: the head weights file (read by the converter module, 018).

use std::path::Path;

use super::incompat::Incompat;
use super::manifest::HeadSpec;

/// What a head-weights reader returns: the four tensors and the metadata of the file.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadTensors {
    /// `(name, shape, f32 values)` per tensor.
    pub tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
    /// The `temperature` stored in the file, if any.
    pub temperature: Option<f64>,
    /// The hidden size stored in the file, if any.
    pub d_model: Option<u64>,
}

/// Reads a head-weights file without running any code in it (018 implements it).
pub trait HeadReader: Send + Sync {
    /// Read the file at `path`.
    ///
    /// # Errors
    /// [`Incompat::HeadWeights`] (or another reason) if the file is not acceptable.
    fn read(&self, path: &Path) -> Result<HeadTensors, Incompat>;
}

fn bad(detail: String) -> Incompat {
    Incompat::HeadWeights { detail }
}

/// Check a pointer head: exactly `q.weight`, `q.bias`, `k.weight`, `k.bias` with the right
/// shapes, all finite, and a file whose metadata agrees with the manifest.
///
/// `declared_temperature` is the variant's declared temperature, if any.
///
/// # Errors
/// [`Incompat::HeadWeights`] naming the tensor or metadata that is wrong.
pub fn check_pointer(
    head: &HeadSpec,
    reader: &dyn HeadReader,
    path: &Path,
    declared_temperature: Option<f64>,
) -> Result<(), Incompat> {
    let HeadSpec::Pointer {
        d_model, proj_dim, ..
    } = head
    else {
        return Ok(());
    };
    let file = reader.read(path)?;
    let expected: [(&str, Vec<u64>); 4] = [
        ("q.weight", vec![*proj_dim, *d_model]),
        ("q.bias", vec![*proj_dim]),
        ("k.weight", vec![*proj_dim, *d_model]),
        ("k.bias", vec![*proj_dim]),
    ];
    for (name, shape) in &expected {
        let Some((_, got_shape, values)) = file.tensors.iter().find(|(n, _, _)| n == name) else {
            return Err(bad(format!("tensor '{name}' is missing")));
        };
        if got_shape != shape {
            return Err(bad(format!(
                "tensor '{name}' has shape {got_shape:?}, expected {shape:?}"
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(bad(format!("tensor '{name}' has non-finite values")));
        }
    }
    if let Some((n, _, _)) = file
        .tensors
        .iter()
        .find(|(n, _, _)| !expected.iter().any(|(e, _)| e == n))
    {
        return Err(bad(format!("tensor '{n}' is not expected")));
    }
    if let (Some(file_t), Some(declared)) = (file.temperature, declared_temperature)
        && (file_t - declared).abs() > 1e-9
    {
        return Err(bad(format!(
            "the file stores temperature {file_t}, the manifest declares {declared}"
        )));
    }
    if let Some(d) = file.d_model
        && d != *d_model
    {
        return Err(bad(format!(
            "the file stores d_model {d}, the manifest declares {d_model}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::files::FileEntry;

    struct Fixed(HeadTensors);
    impl HeadReader for Fixed {
        fn read(&self, _: &Path) -> Result<HeadTensors, Incompat> {
            Ok(self.0.clone())
        }
    }

    fn head() -> HeadSpec {
        HeadSpec::Pointer {
            d_model: 4,
            proj_dim: 2,
            weights: FileEntry {
                path: "head.pt".into(),
                size: 1,
                sha256: "0".repeat(64),
                origin: None,
            },
        }
    }

    fn good() -> HeadTensors {
        HeadTensors {
            tensors: vec![
                ("q.weight".into(), vec![2, 4], vec![0.5; 8]),
                ("q.bias".into(), vec![2], vec![0.0; 2]),
                ("k.weight".into(), vec![2, 4], vec![0.5; 8]),
                ("k.bias".into(), vec![2], vec![0.0; 2]),
            ],
            temperature: Some(2.351),
            d_model: Some(4),
        }
    }

    fn run(t: HeadTensors, declared: Option<f64>) -> Result<(), Incompat> {
        check_pointer(&head(), &Fixed(t), Path::new("x"), declared)
    }

    #[test]
    fn a_good_head_passes_with_the_same_temperature() {
        assert_eq!(run(good(), Some(2.351)), Ok(()));
        assert_eq!(run(good(), None), Ok(()));
    }

    #[test]
    fn each_defect_is_a_head_weights_error() {
        let mut missing = good();
        missing.tensors.remove(1);
        let mut shape = good();
        shape.tensors[0].1 = vec![4, 2];
        let mut nan = good();
        nan.tensors[2].2[3] = f32::NAN;
        let mut extra = good();
        extra.tensors.push(("evil".into(), vec![1], vec![0.0]));
        let mut wrong_d = good();
        wrong_d.d_model = Some(8);
        for (case, t, declared) in [
            ("missing", missing, None),
            ("shape", shape, None),
            ("non-finite", nan, None),
            ("extra", extra, None),
            ("d_model", wrong_d, None),
            ("temperature", good(), Some(1.0)),
        ] {
            assert!(
                matches!(run(t, declared), Err(Incompat::HeadWeights { .. })),
                "{case}"
            );
        }
    }
}
