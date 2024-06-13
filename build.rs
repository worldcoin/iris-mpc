//! Rust replaces LD_LIBRARY_PATH rather than prepending to it, which makes it
//! difficult to install the CUDA libraries in non-standard directories.
//!
//! This build script is a workaround until the bug is fixed in `cargo` or the
//! `cudarc` build scripts:
//! <https://github.com/rust-lang/cargo/issues/4895#issuecomment-2159726116>
//!
//! Usage:
//! ```sh
//! export PRE_CARGO_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
//! cargo run/test/bench
//! ```

fn main() {
    // TODO: get actual local search paths from cudarc, or fix cudarc to use
    // rustc-link-search?
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    let build_path = std::env::var("LD_LIBRARY_PATH");
    let sys_path = std::env::var("PRE_CARGO_LD_LIBRARY_PATH");

    match (build_path, sys_path) {
        (Ok(build), Ok(sys)) => println!("cargo:rustc-env=LD_LIBRARY_PATH={build}:{sys}"),
        (Err(_build), Ok(sys)) => println!("cargo:rustc-env=LD_LIBRARY_PATH={sys}"),
        (_, Err(_sys)) => {
            // We don't have anything to do, so just leave the default build
            // path alone
        }
    }

    // Prints all dynamic loading as it happens
    // println!("cargo:rustc-env=LD_DEBUG=all");

    // Prints shared objects loaded at launch for debugging,
    // not useful for post-launch dlopen() loads
    // println!("cargo:rustc-env=LD_TRACE_LOADED_OBJECTS=");
}
