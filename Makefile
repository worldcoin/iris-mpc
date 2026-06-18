# This allows one to test the same commands that get run by the .github/workflows.
# Eliminates errors resulting from running a less restrictive check.

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings --no-deps

fmt:
	cargo fmt --all -- --check

unit_tests:
	cargo test --release

build_tests:
	cargo build --release --all-features --tests

build_all:
	cargo build --release --all-features --workspace --lib --bins --benches --examples

build_docs:
	cargo doc --all-features --no-deps --document-private-items


all:
	@$(MAKE) fmt
	@$(MAKE) clippy
	@$(MAKE) build_docs
	@$(MAKE) build_tests
	@$(MAKE) build_all
	@$(MAKE) unit_tests
