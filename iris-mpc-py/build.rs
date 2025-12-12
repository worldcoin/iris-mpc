fn main() {
    // When building the Python extension module on macOS, allow unresolved Python C-API
    // symbols at link time (they are resolved by the Python interpreter at runtime).
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
