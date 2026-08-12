//! Emits `IRIS_MPC_SOFTWARE_VERSION` (`<crate version>+<commit>`) so the binary
//! carries the commit it was built from. It is read by
//! `iris_mpc_common::config::SOFTWARE_VERSION`, which is hashed into
//! `CommonConfig` and therefore compared across all three MPC parties during
//! startup synchronization.
//!
//! The full string is composed here rather than in the crate because `concat!`
//! only joins literals, so the crate cannot stitch together a version and an
//! optional hash at compile time.
//!
//! Commit resolution order:
//!
//! 1. `IRIS_MPC_GIT_HASH` from the build environment. Docker builds need this:
//!    `.dockerignore` excludes `.git`, so `git` inside the image build has
//!    nothing to read. Pass it as a build arg (see `Dockerfile.hawk`).
//! 2. `git rev-parse` in the source tree, for local/CI cargo builds.
//! 3. `"unknown"`, so a build without any git metadata still compiles. Note
//!    that this degrades the cross-party check to a crate-version comparison
//!    rather than failing the build.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=IRIS_MPC_GIT_HASH");

    let hash = std::env::var("IRIS_MPC_GIT_HASH")
        .ok()
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
        .or_else(git_hash)
        .unwrap_or_else(|| "unknown".to_owned());

    // Always set by cargo for build scripts.
    let crate_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();

    println!("cargo:rustc-env=IRIS_MPC_SOFTWARE_VERSION={crate_version}+{hash}");
}

/// `<short sha>` for a clean tree, `<short sha>-dirty` when there are
/// uncommitted changes. `None` if `git` is unavailable or this is not a
/// checkout.
fn git_hash() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;

    // Rebuild when HEAD moves. `--git-dir` resolves through worktrees, where
    // the naive `<repo>/.git` guess is a file rather than a directory.
    //
    // Deliberately NOT watching `.git/index`: git rewrites it whenever it
    // refreshes cached stat info (`git status`, `git diff`, IDE git polling),
    // which would invalidate this build script — and most of the workspace
    // depends on this crate, so that is a full rebuild cascade for nothing.
    // The trade-off is that the `-dirty` suffix below is only as fresh as the
    // last time HEAD moved. It is a human-facing hint, not part of the
    // cross-party equality contract: all parties in a given deployment are
    // built from one artifact, so they agree regardless.
    if let Some(git_dir) = git(&manifest_dir, &["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
    }

    let short_sha = git(&manifest_dir, &["rev-parse", "--short=12", "HEAD"])?;

    // `--quiet` makes the exit code the whole answer: non-zero means the
    // working tree differs from HEAD.
    let dirty = Command::new("git")
        .current_dir(&manifest_dir)
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .map(|status| !status.success())
        .unwrap_or(false);

    Some(if dirty {
        format!("{short_sha}-dirty")
    } else {
        short_sha
    })
}

/// Run `git` in `dir`, returning trimmed stdout on success.
fn git(dir: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}
