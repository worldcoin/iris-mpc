fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    tonic_build::configure()
        .out_dir("src/proto/")
        .compile_protos(
            &["reshare.proto"], // Files in the path
            &["protos"],        // The include path to search
        )
        .unwrap();
}
