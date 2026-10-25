//! Build script: gives every build an identifier that changes with the source code, so that
//! cached self-check results never outlive the code that produced them (spec 005, 7.2).
//! Standard library only.

use std::fs;
use std::path::Path;

// The guard of the `metal` feature is shared with the library, which tests it.
#[path = "src/engine/build_guard.rs"]
mod build_guard;

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs" || e == "json") {
            out.push(p);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GGML_METAL");
    refuse_a_metal_build_without_metal();
    let mut files = Vec::new();
    walk(Path::new("src"), &mut files);
    files.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in files {
        for b in f.to_string_lossy().replace('\\', "/").bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        if let Ok(bytes) = fs::read(&f) {
            for b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    println!("cargo:rustc-env=ARCHAI_JEV_BUILD_ID={h:016x}");
    link_clang_runtime_on_macos();
}

/// The `metal` feature is accepted by Cargo on every platform and `.cargo/config.toml` switches
/// Metal off: stop the build rather than produce a binary that says `metal` and contains none.
fn refuse_a_metal_build_without_metal() {
    let feature = std::env::var_os("CARGO_FEATURE_METAL").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let ggml_metal = std::env::var("GGML_METAL").ok();
    if let Err(message) = build_guard::check(feature, &target_os, ggml_metal.as_deref()) {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

/// macOS only. ggml's C++ uses `__builtin_available`, which clang lowers to a call into compiler-rt
/// (`___isPlatformVersionAtLeast`); rustc links the extension without clang's default libraries and
/// maturin adds `-undefined dynamic_lookup`, so without this the wheel installs but `import` fails
/// ("symbol not found in flat namespace"). Seen on macos-arm64 in the `llama-probe` workflow.
fn link_clang_runtime_on_macos() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = std::process::Command::new("cc")
        .arg("--print-file-name=libclang_rt.osx.a")
        .output();
    let path = out
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    if Path::new(&path).is_absolute() && Path::new(&path).exists() {
        println!("cargo:rustc-link-arg-cdylib={path}");
    } else {
        println!(
            "cargo:warning=libclang_rt.osx.a not found (`cc --print-file-name` gave {path:?}); the extension may fail to import on macOS"
        );
    }
}
