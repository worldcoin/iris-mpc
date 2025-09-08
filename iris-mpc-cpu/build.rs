fn main() {
    println!("cargo:rerun-if-changed=src/proto/party_node.proto");
    tonic_build::configure()
        .out_dir("src/proto_generated")
        .emit_rerun_if_changed(false) // https://github.com/hyperium/tonic/issues/1070#issuecomment-1729075588
        .compile_protos(&["src/proto/party_node.proto"], &["src/proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
