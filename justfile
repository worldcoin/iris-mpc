default:
  just --list
dev-pg-up:
  docker run --name gpu-iris-dev-db -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
dev-pg-down:
  docker stop gpu-iris-dev-db && docker rm gpu-iris-dev-db
lint:
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features -q -- -D warnings
  RUSTDOCFLAGS='-D warnings' cargo doc --workspace -q --no-deps
