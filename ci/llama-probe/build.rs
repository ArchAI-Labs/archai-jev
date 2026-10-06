//! macOS only. ggml's C++ code uses `__builtin_available`, which clang lowers to a call to compiler-rt's
//! `___isPlatformVersionAtLeast`. rustc links the extension without clang's default libraries and maturin adds
//! `-undefined dynamic_lookup`, so the symbol stays unresolved at link time and `import` fails at run time with
//! "symbol not found in flat namespace '___isPlatformVersionAtLeast'" (seen on macos-arm64 in the llama-probe workflow).
//! Linking clang's runtime archive explicitly fixes it. The real extension needs the same build script.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    // `cc --print-file-name` prints the absolute path when the file exists in clang's resource dir, else just the name.
    let out = Command::new("cc")
        .arg("--print-file-name=libclang_rt.osx.a")
        .output()
        .expect("could not run `cc`");
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        Path::new(&path).is_absolute() && Path::new(&path).exists(),
        "libclang_rt.osx.a not found: `cc --print-file-name=libclang_rt.osx.a` returned {path:?}"
    );
    println!("cargo:rustc-link-arg-cdylib={path}");
}
