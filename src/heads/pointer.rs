//! Kev's pointer head (spec 006b section 8, spike S2 section 4):
//!
//! ```text
//! z_j = ( W_k·h_j + b_k ) · ( W_q·h_dec + b_q ) / sqrt(proj_dim)
//! ```
//!
//! `h_dec` is the hidden state at the `decide` token and `h_j` the one at the closing token of
//! option `j`; both are the last layer after the final norm, in f32. Weights are f32; the products
//! are accumulated in f64.

use super::Head;
use crate::engine::{Output, OutputSpec};
use crate::error::{Error, Result};
use crate::models::head::HeadTensors;
use crate::prompt::{Branch, QuestionSupport, Unsupported};
use crate::schema::Question;

fn inference(detail: impl Into<String>) -> Error {
    Error::Inference {
        detail: detail.into(),
    }
}

/// The pointer head.
#[derive(Debug, Clone)]
pub struct PointerHead {
    d: usize,
    p: usize,
    q_weight: Vec<f32>,
    q_bias: Vec<f32>,
    k_weight: Vec<f32>,
    k_bias: Vec<f32>,
    scale: f64,
    use_q_bias: bool,
    use_k_bias: bool,
}

fn tensor<'a>(t: &'a HeadTensors, name: &str, shape: &[u64]) -> Result<&'a [f32]> {
    let (_, got, values) = t
        .tensors
        .iter()
        .find(|(n, _, _)| n == name)
        .ok_or_else(|| inference(format!("the head has no tensor '{name}'")))?;
    if got != shape {
        return Err(inference(format!(
            "head tensor '{name}' has shape {got:?}, expected {shape:?}"
        )));
    }
    Ok(values)
}

impl PointerHead {
    /// Build from the four tensors of a head file (`q.weight`, `q.bias`, `k.weight`, `k.bias`).
    ///
    /// # Errors
    /// [`Error::Inference`] if a tensor is missing or has another shape.
    pub fn from_tensors(t: &HeadTensors, d_model: u64, proj_dim: u64) -> Result<Self> {
        let wshape = [proj_dim, d_model];
        PointerHead::new(
            usize::try_from(d_model).map_err(|_| inference("d_model is too large"))?,
            usize::try_from(proj_dim).map_err(|_| inference("proj_dim is too large"))?,
            tensor(t, "q.weight", &wshape)?.to_vec(),
            tensor(t, "q.bias", &[proj_dim])?.to_vec(),
            tensor(t, "k.weight", &wshape)?.to_vec(),
            tensor(t, "k.bias", &[proj_dim])?.to_vec(),
        )
    }

    /// Build from the weights directly (`*_weight` are `proj_dim` rows of `d` values).
    ///
    /// # Errors
    /// [`Error::Inference`] if the lengths do not match `d` and `p`.
    pub fn new(
        d: usize,
        p: usize,
        q_weight: Vec<f32>,
        q_bias: Vec<f32>,
        k_weight: Vec<f32>,
        k_bias: Vec<f32>,
    ) -> Result<Self> {
        if p == 0 || d == 0 {
            return Err(inference("the head has no dimensions"));
        }
        if q_weight.len() != p * d
            || k_weight.len() != p * d
            || q_bias.len() != p
            || k_bias.len() != p
        {
            return Err(inference(
                "the head tensors do not match d_model and proj_dim",
            ));
        }
        Ok(PointerHead {
            d,
            p,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            scale: 1.0 / (p as f64).sqrt(),
            use_q_bias: true,
            use_k_bias: true,
        })
    }

    /// Hidden size the head reads.
    pub fn d_model(&self) -> usize {
        self.d
    }

    /// Size of the projection (256 for Kev).
    pub fn proj_dim(&self) -> usize {
        self.p
    }

    fn project(&self, weight: &[f32], bias: &[f32], use_bias: bool, h: &[f32]) -> Vec<f64> {
        weight
            .chunks_exact(self.d)
            .zip(bias)
            .map(|(row, b)| {
                let dot: f64 = row
                    .iter()
                    .zip(h)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum();
                if use_bias { dot + f64::from(*b) } else { dot }
            })
            .collect()
    }

    /// The raw logits of the options whose closing hidden states are `h_opts`.
    ///
    /// # Errors
    /// [`Error::Inference`] if a hidden state does not have `d_model` values.
    pub fn logits(&self, h_dec: &[f32], h_opts: &[&[f32]]) -> Result<Vec<f64>> {
        if h_dec.len() != self.d || h_opts.iter().any(|h| h.len() != self.d) {
            return Err(inference(format!(
                "a hidden state does not have the {} values the head reads",
                self.d
            )));
        }
        let q = self.project(&self.q_weight, &self.q_bias, self.use_q_bias, h_dec);
        Ok(h_opts
            .iter()
            .map(|h| {
                let k = self.project(&self.k_weight, &self.k_bias, self.use_k_bias, h);
                k.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() * self.scale
            })
            .collect())
    }

    /// A deliberately wrong head, to prove the tests would notice (spec 006b AC on mutants).
    #[cfg(any(test, feature = "testing"))]
    pub fn mutant(mut self, q_bias: bool, k_bias: bool, scale: Option<f64>) -> Self {
        self.use_q_bias = q_bias;
        self.use_k_bias = k_bias;
        if let Some(s) = scale {
            self.scale = s;
        }
        self
    }
}

impl QuestionSupport for PointerHead {
    fn check(&self, _: &str, _: &str, _: &Question) -> std::result::Result<(), Unsupported> {
        // Any question with 1..=255 options, which the domain types already guarantee.
        Ok(())
    }
}

impl Head for PointerHead {
    fn kind(&self) -> &'static str {
        "pointer"
    }

    fn outputs(&self, _question: &Question, branch: &Branch) -> Vec<OutputSpec> {
        let mut specs = Vec::with_capacity(1 + branch.option_ends().len());
        specs.push(OutputSpec::Hidden {
            position: branch.decide(),
        });
        specs.extend(
            branch
                .option_ends()
                .iter()
                .map(|&position| OutputSpec::Hidden { position }),
        );
        specs
    }

    fn score(&self, question: &Question, outputs: &[Output]) -> Result<Vec<f64>> {
        let mut hidden: Vec<&[f32]> = Vec::with_capacity(outputs.len());
        for o in outputs {
            match o {
                Output::Hidden(h) => hidden.push(h),
                #[allow(unreachable_patterns)]
                _ => return Err(inference("the pointer head reads hidden states only")),
            }
        }
        let (dec, opts) = hidden
            .split_first()
            .ok_or_else(|| inference("the pointer head got no hidden states"))?;
        if opts.len() != question.option_count() {
            return Err(inference(format!(
                "the pointer head got {} option states for {} options",
                opts.len(),
                question.option_count()
            )));
        }
        self.logits(dec, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// d_model 4, proj_dim 4 (scale 0.5): identity projections, one bias, checked by hand.
    fn tiny() -> PointerHead {
        let eye: Vec<f32> = (0..16)
            .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
            .collect();
        PointerHead::new(
            4,
            4,
            eye.clone(),
            vec![0.0; 4],
            eye,
            vec![1.0, 0.0, 0.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn tiny_head_by_hand() {
        let h = tiny();
        // q = h_dec = [1,2,3,4]; k = h_opt + b_k = [1,1,0,0]; dot = 3; scale 0.5 -> 1.5.
        let z = h
            .logits(
                &[1.0, 2.0, 3.0, 4.0],
                &[&[0.0, 1.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]],
            )
            .unwrap();
        // second option: k = [1,0,0,0]; dot = 1; -> 0.5
        assert_eq!(z, vec![1.5, 0.5]);
    }

    #[test]
    fn scale_is_one_over_sqrt_proj_dim_not_a_constant() {
        // proj_dim 1 (scale 1), d 1: z = (w_k*h_o + b_k)*(w_q*h_d + b_q).
        let h = PointerHead::new(1, 1, vec![2.0], vec![1.0], vec![3.0], vec![0.5]).unwrap();
        let z = h.logits(&[1.0], &[&[2.0]]).unwrap();
        assert_eq!(z, vec![(3.0 * 2.0 + 0.5) * (2.0 * 1.0 + 1.0)]);
    }

    #[test]
    fn wrong_sizes_are_errors_not_panics() {
        assert!(
            PointerHead::new(
                4,
                4,
                vec![0.0; 3],
                vec![0.0; 4],
                vec![0.0; 16],
                vec![0.0; 4]
            )
            .is_err()
        );
        assert!(tiny().logits(&[1.0; 3], &[]).is_err());
        assert!(tiny().logits(&[1.0; 4], &[&[0.0; 5]]).is_err());
        assert!(PointerHead::new(0, 0, vec![], vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn from_tensors_checks_names_and_shapes() {
        let t = HeadTensors {
            tensors: vec![
                ("q.weight".into(), vec![2, 3], vec![0.0; 6]),
                ("q.bias".into(), vec![2], vec![0.0; 2]),
                ("k.weight".into(), vec![2, 3], vec![0.0; 6]),
                ("k.bias".into(), vec![2], vec![0.0; 2]),
            ],
            ..HeadTensors::default()
        };
        assert!(PointerHead::from_tensors(&t, 3, 2).is_ok());
        assert!(PointerHead::from_tensors(&t, 4, 2).is_err());
        let mut missing = t.clone();
        missing.tensors.remove(0);
        assert!(PointerHead::from_tensors(&missing, 3, 2).is_err());
    }
}
