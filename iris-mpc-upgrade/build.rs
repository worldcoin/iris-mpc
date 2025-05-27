fn main() {
    println!("cargo:rerun-if-changed=protos/reshare.proto");
    tonic_build::configure()
        .out_dir("src/proto/")
        .emit_rerun_if_changed(false) // https://github.com/hyperium/tonic/issues/1070#issuecomment-1729075588
        .compile_protos(
            &["reshare.proto"], // Files in the path
            &["protos"],        // The include path to search
        )
        .unwrap();
}
