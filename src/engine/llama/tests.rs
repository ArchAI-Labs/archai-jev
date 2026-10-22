//! Tests of the settings of the engine that need no model: the parameters of a context, the
//! identity of the engine with its device, and the snapshots of another engine (spec 011).

use super::*;
use crate::engine::device::Layers;
use crate::prefix::{Fingerprint, FingerprintParts, KeyHasher};

fn config(plan: OffloadPlan) -> LlamaConfig {
    LlamaConfig {
        threads: 4,
        max_context: 512,
        hidden_states: false,
        plan,
    }
}

fn cpu() -> OffloadPlan {
    cpu_plan().unwrap()
}

fn gpu() -> OffloadPlan {
    OffloadPlan {
        layers: Layers::All,
        device: cpu().device,
        op_offload: true,
    }
}

#[test]
fn a_context_takes_op_offload_from_the_plan() {
    for plan in [cpu(), gpu()] {
        let params = context_params_for(&config(plan), 0, 2, 4).unwrap();
        assert_eq!(params.op_offload(), plan.op_offload, "{plan:?}");
    }
    // the default of the library is `true`: a CPU plan must not inherit it
    assert!(
        !context_params_for(&config(cpu()), 0, 2, 4)
            .unwrap()
            .op_offload()
    );
}

#[test]
fn fingerprint_changes_with_the_device_and_the_plan() {
    let text = |plan: OffloadPlan, label: &str| fingerprint_text(&config(plan), 0, 2, label);
    let base = text(cpu(), "CPU/CPU/the processor");
    assert_eq!(base, text(cpu(), "CPU/CPU/the processor"), "deterministic");
    // another device, and another GPU in the same slot
    assert_ne!(base, text(cpu(), "CUDA/CUDA0/NVIDIA GeForce RTX 5070"));
    assert_ne!(
        text(cpu(), "CUDA/CUDA0/NVIDIA GeForce RTX 5070"),
        text(cpu(), "CUDA/CUDA0/NVIDIA GeForce RTX 4090")
    );
    // the offload plan itself
    assert_ne!(base, text(gpu(), "CPU/CPU/the processor"));
    let no_offload = OffloadPlan {
        op_offload: false,
        ..gpu()
    };
    assert_ne!(text(gpu(), "x"), text(no_offload, "x"));
}

#[test]
fn a_prefix_saved_on_one_device_has_another_key_on_another() {
    let key = |label: &str| {
        let engine = fingerprint_text(&config(cpu()), 0, 2, label);
        let fingerprint = Fingerprint::new(&FingerprintParts {
            model: "m",
            gguf_sha256: "g",
            family: "f",
            dtype: "q8_0",
            template_id: "t",
            template_version: 1,
            tokenizer_sha256: "k",
            engine: &engine,
        });
        let mut hasher = KeyHasher::new(&fingerprint);
        hasher.update(&[1, 2, 3]);
        hasher.key()
    };
    assert_eq!(key("CPU/CPU/p"), key("CPU/CPU/p"));
    assert_ne!(
        key("CPU/CPU/p"),
        key("CUDA/CUDA0/g"),
        "the cache never serves another device"
    );
}

fn snapshot_of<T: std::any::Any + Send + Sync>(engine: &str, state: T) -> Snapshot {
    Snapshot::new(
        3,
        12,
        Saved {
            engine: Arc::from(engine),
            state,
        },
    )
}

#[test]
fn a_snapshot_of_another_engine_configuration_is_refused() {
    let mine = fingerprint_text(&config(cpu()), 0, 2, "CPU/CPU/p");
    let on_gpu = fingerprint_text(&config(gpu()), 0, 2, "CUDA/CUDA0/g");
    let made_here = snapshot_of(&mine, 7_u32);
    assert_eq!(*open::<u32>(&made_here, &mine).unwrap(), 7);
    // made on a GPU, restored on the CPU (and the other way round): the bytes are another device's
    let err = open::<u32>(&made_here, &on_gpu).err().unwrap();
    assert!(matches!(err, Error::Inference { .. }), "{err:?}");
    assert!(
        err.to_string().contains("another engine configuration"),
        "{err}"
    );
    let made_on_gpu = snapshot_of(&on_gpu, 7_u32);
    assert!(open::<u32>(&made_on_gpu, &mine).is_err());
    // a snapshot of another kind of engine is refused too, as before
    let err = open::<u64>(&made_here, &mine).err().unwrap();
    assert!(err.to_string().contains("not made by this engine"), "{err}");
}
