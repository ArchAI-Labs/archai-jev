//! The converter of Kev-style checkpoints: a Qwen3.5 base model plus a LoRA adapter, written as
//! a GGUF (spec 018, recipe `qwen35-lora/1`).

use std::path::Path;
use std::time::{Duration, Instant};

use super::config::Hparams;
use super::gguf_out::{self, Out, TensorDesc, Value};
use super::layout::Dims;
use super::lora::LoraConfig;
use super::plan::{self, Plan};
use super::safetensors::SafeTensors;
use super::stream::{Scratch, write_tensor};
use super::vocab::Vocab;
use crate::error::{Error, Result};
use crate::hub::events::{Cancel, Event, Observer};
use crate::models::convert::{ConvertJob, Converted, Converter, SourceFiles};
use crate::models::families::ArchParams;
use crate::models::incompat::Incompat;

/// The version of the recipe: it changes **only** when the output can change, so a library
/// update does not reconvert gigabytes for nothing.
pub const RECIPE: &str = "qwen35-lora/1";

/// How often progress is reported at most.
const PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// Name of the GGUF inside the output folder.
pub const MODEL_FILE: &str = "model.gguf";

/// The converter of `hf-lora` sources of the family `qwen35-pointer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen35Converter;

fn file_type(dtype: &str) -> u32 {
    // the values of llama.cpp's LlamaFileType: ALL_F32 = 0, MOSTLY_BF16 = 32
    if dtype == "bf16" { 32 } else { 0 }
}

/// The metadata of the GGUF, as the official script writes it (without the descriptive keys and
/// the chat template).
///
/// # Errors
/// `ConversionFailed` when a number of the config does not fit the 32 bits of a GGUF integer:
/// it is never written saturated.
pub fn metadata(h: &Hparams, vocab: &Vocab, dtype: &str) -> Result<Vec<(String, Value)>> {
    let overflow = std::cell::RefCell::new(None::<String>);
    let note = |what: &str, v: u64| {
        let mut slot = overflow.borrow_mut();
        if slot.is_none() {
            *slot = Some(format!("{what} is {v}"));
        }
    };
    let u = |k: &str, v: u64| {
        let n = u32::try_from(v).unwrap_or_else(|_| {
            note(k, v);
            0
        });
        (k.to_string(), Value::U32(n))
    };
    let mut sections: Vec<i32> = h
        .mrope_section
        .iter()
        .map(|s| {
            i32::try_from(*s).unwrap_or_else(|_| {
                note("an mrope section", *s);
                0
            })
        })
        .collect();
    sections.resize(4, 0);
    let mut recurrent = h.linear_layers.clone();
    recurrent.extend(std::iter::repeat_n(false, index(h.n_mtp, &note)));
    let meta = vec![
        (
            "general.architecture".to_string(),
            Value::Str("qwen35".to_string()),
        ),
        u("qwen35.block_count", h.n_layers + h.n_mtp),
        u("qwen35.context_length", h.max_position),
        u("qwen35.embedding_length", h.hidden),
        u("qwen35.feed_forward_length", h.ff),
        u("qwen35.attention.head_count", h.heads),
        u("qwen35.attention.head_count_kv", h.kv_heads),
        (
            "qwen35.rope.dimension_sections".to_string(),
            Value::I32s(sections),
        ),
        (
            "qwen35.rope.freq_base".to_string(),
            Value::F32(h.rope_theta as f32),
        ),
        (
            "qwen35.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(h.rms_eps as f32),
        ),
        u("qwen35.attention.key_length", h.head_dim),
        u("qwen35.attention.value_length", h.head_dim),
        u("general.file_type", u64::from(file_type(dtype))),
        u("qwen35.nextn_predict_layers", h.n_mtp),
        u("qwen35.ssm.conv_kernel", h.conv_kernel),
        u("qwen35.ssm.state_size", h.key_head_dim),
        u("qwen35.ssm.group_count", h.key_heads),
        u("qwen35.ssm.time_step_rank", h.value_heads),
        u("qwen35.ssm.inner_size", h.value_head_dim * h.value_heads),
        (
            "qwen35.attention.recurrent_layers".to_string(),
            Value::Bools(recurrent),
        ),
        u("qwen35.full_attention_interval", h.full_attention_interval),
        u("qwen35.rope.dimension_count", h.rope_dim()),
        u("general.quantization_version", 2),
        (
            "tokenizer.ggml.model".to_string(),
            Value::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::Str("qwen35".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Strs(vocab.tokens.clone()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::I32s(vocab.types.clone()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Value::Strs(vocab.merges.clone()),
        ),
        u("tokenizer.ggml.eos_token_id", u64::from(vocab.eos)),
        u("tokenizer.ggml.padding_token_id", u64::from(vocab.pad)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Value::Bool(vocab.add_bos),
        ),
        (
            "tokenizer.ggml.add_eos_token".to_string(),
            Value::Bool(vocab.add_eos),
        ),
    ];
    if let Some(detail) = overflow.take() {
        return Err(Error::IncompatibleModel(Incompat::ConversionFailed {
            detail: format!("{detail}, which does not fit the 32 bits of a GGUF integer"),
        }));
    }
    Ok(meta)
}

fn index(v: u64, note: &dyn Fn(&str, u64)) -> usize {
    usize::try_from(v).unwrap_or_else(|_| {
        note("a count", v);
        0
    })
}

fn io(path: &Path, e: &std::io::Error) -> Error {
    super::safetensors::io_error(path, e)
}

impl Converter for Qwen35Converter {
    fn kinds(&self) -> Vec<String> {
        vec!["hf-lora".to_string()]
    }

    fn version(&self) -> String {
        RECIPE.to_string()
    }

    fn convert(&self, job: &ConvertJob<'_>, out_dir: &Path) -> Result<Converted> {
        convert_files_to(
            &job.files,
            job.params,
            &job.source.base.repo,
            job.dtype,
            out_dir,
            Some(&job.destination),
            job.observer,
            job.cancel,
        )
    }
}

/// [`convert_files_to`] when the output folder is also the final one.
///
/// # Errors
/// See [`convert_files_to`].
pub fn convert_files(
    files: &SourceFiles,
    params: &ArchParams,
    base_repo: &str,
    dtype: &str,
    out_dir: &Path,
    observer: &dyn Observer,
    cancel: &dyn Cancel,
) -> Result<Converted> {
    convert_files_to(
        files, params, base_repo, dtype, out_dir, None, observer, cancel,
    )
}

/// Convert the files of a checkpoint into `out_dir/model.gguf`; the SHA-256 of the file, computed
/// while it was written, is in the result.
///
/// # Errors
/// `IncompatibleModelError` for anything wrong with the files or with the manifest's
/// description of them, an I/O error, `Cancelled`.
#[allow(clippy::too_many_arguments)]
pub fn convert_files_to(
    files: &SourceFiles,
    params: &ArchParams,
    base_repo: &str,
    dtype: &str,
    out_dir: &Path,
    destination: Option<&Path>,
    observer: &dyn Observer,
    cancel: &dyn Cancel,
) -> Result<Converted> {
    let started = Instant::now();
    plan::matrix_type(dtype)?;

    // 1. what the files say, against what the manifest says
    let hparams = Hparams::read(&files.base_config)?;
    hparams
        .check_against(params)
        .map_err(Error::IncompatibleModel)?;
    let lora = LoraConfig::read(&files.adapter_config, base_repo)?;

    // 2. the headers of the weights, and the plan
    let mut base = SafeTensors::open(&files.base_weights, "base weights")?;
    let mut adapter = SafeTensors::open(&files.adapter_weights, "adapter weights")?;
    let dims = Dims::from_hparams(&hparams);
    let plan: Plan = plan::build(&dims, dtype, &base, &adapter, &lora)?;

    // 3. the vocabulary and the header of the GGUF
    let vocab = Vocab::read(
        &files.base_tokenizer,
        &files.base_tokenizer_config,
        hparams.vocab,
    )?;
    let meta = metadata(&hparams, &vocab, dtype)?;
    let descs: Vec<TensorDesc> = plan
        .tensors
        .iter()
        .map(|p| TensorDesc {
            name: p.entry.gguf.clone(),
            dims: p.entry.dims.clone(),
            ty: p.ty,
            bytes: p.bytes,
        })
        .collect();
    let header = gguf_out::header(&meta, &descs);
    drop(vocab);

    // 4. the data, tensor by tensor
    let part = out_dir.join(format!("{MODEL_FILE}.part"));
    let final_path = out_dir.join(MODEL_FILE);
    observer.on_event(&Event::ConvertStart {
        tensors: plan.tensors.len() as u64,
        bytes: header.total_len,
        destination: destination
            .unwrap_or(out_dir)
            .join(MODEL_FILE)
            .display()
            .to_string(),
        dtype: dtype.to_string(),
    });
    let mut out = Out::create(&part)?;
    let written = (|| -> Result<(String, u64)> {
        out.write(&header.bytes)?;
        let data_start = header.bytes.len() as u64;
        let mut scratch = Scratch::default();
        let mut last_report = Instant::now();
        let total = plan.tensors.len() as u64;
        for (i, (p, offset)) in plan.tensors.iter().zip(&header.offsets).enumerate() {
            out.pad_to(data_start + offset)?;
            write_tensor(
                p,
                &mut base,
                &mut adapter,
                &lora,
                &mut out,
                cancel,
                &mut scratch,
            )?;
            if last_report.elapsed() >= PROGRESS_EVERY {
                observer.on_event(&Event::ConvertProgress {
                    done: i as u64 + 1,
                    total,
                });
                last_report = Instant::now();
            }
        }
        out.pad_to(header.total_len)?;
        out.finish_ref()
    })();
    let (sha256, size) = match written {
        Ok(done) => done,
        Err(e) => {
            // never leave a half-written file behind
            let _ = std::fs::remove_file(&part);
            return Err(e);
        }
    };
    if size != header.total_len {
        let _ = std::fs::remove_file(&part);
        return Err(Error::IncompatibleModel(Incompat::ConversionFailed {
            detail: format!("wrote {size} bytes, the header says {}", header.total_len),
        }));
    }
    std::fs::rename(&part, &final_path).map_err(|e| io(&part, &e))?;
    observer.on_event(&Event::ConvertDone {
        seconds: started.elapsed().as_secs_f64(),
        bytes: size,
    });
    Ok(Converted {
        model: MODEL_FILE.into(),
        head: None,
        sha256: Some(sha256),
        size: Some(size),
    })
}
