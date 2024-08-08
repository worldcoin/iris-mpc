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
}
