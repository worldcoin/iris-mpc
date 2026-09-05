use std::{env, fs, path::PathBuf};

fn main() {
    let path = PathBuf::from("../../src/protocol/ops.rs");
    let iris_path = PathBuf::from("../../src/protocol/shared_iris.rs");
    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo:rerun-if-changed={}", iris_path.display());
    let source = fs::read_to_string(path).unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let function = source.find("fn accumulate_component_tiled_6x4<").unwrap();
    let start = source[..function].rfind("#[cfg(target_arch").unwrap();
    let end = source.find("pub fn non_existent_distance()").unwrap();
    fs::write(out.join("baseline.rs"), &source[start..end]).unwrap();

    // Compile the PR's complete mixed-plane module, including query caches,
    // packed-pair dispatch and tail handling. Executable code is unchanged;
    // expose the module and render its service-only doc link as plain code.
    let function = source.find("mod mixed_scan {").unwrap();
    let start = source[..function].rfind("#[cfg(target_arch").unwrap();
    let end = source[function..].find("#[cfg(test)]").unwrap() + function;
    fs::write(
        out.join("mixed_scan.rs"),
        source[start..end]
            .replacen("mod mixed_scan {", "pub mod mixed_scan {", 1)
            .replace(
                "[`super::rotation_aware_pairwise_distance_rowmajor`]",
                "`rotation_aware_pairwise_distance_rowmajor`",
            ),
    )
    .unwrap();
    // Keep the production rotation constants, rather than duplicate their math.
    let start = source
        .find("    /// Rotation amounts for each rotation.")
        .unwrap();
    let end = source[start..].find("    /// Rotate row directly").unwrap() + start;
    let constants = format!(
        "#[cfg(target_arch = \"aarch64\")]\npub struct PrerotatedQueryRowMajorView<const ROTATIONS: usize>;\n\
         #[cfg(target_arch = \"aarch64\")]\nimpl<const ROTATIONS: usize> PrerotatedQueryRowMajorView<ROTATIONS> {{\n{}\n}}\n",
        &source[start..end]
    );
    fs::write(out.join("rotation_constants.rs"), constants).unwrap();

    // Resident layout and its conversion are also taken verbatim from the PR.
    let source = fs::read_to_string(iris_path).unwrap();
    let function = source.find("pub struct MixedPlaneIris {").unwrap();
    let start = source[..function].rfind("#[derive(").unwrap();
    let end = source.find("/// Resident layout of a worker pool").unwrap();
    fs::write(out.join("mixed_iris.rs"), &source[start..end]).unwrap();
}
