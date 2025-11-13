# Repository Guidelines

## Project Structure & Module Organization
- Root `Cargo.toml` wires `iris-mpc/` (service entrypoints), `iris-mpc-common/` (shared types), `iris-mpc-cpu/` (HNSW stack), `iris-mpc-gpu/` (CUDA stack), and `iris-mpc-store/` (persistence).
- Tooling lives in `iris-mpc-utils/` and `iris-mpc-upgrade*/`; bindings in `iris-mpc-py/`; ops assets land in `scripts/`, `deploy/`, `migrations/`, `certs/`.
- Each crate keeps integration coverage in `tests/` beside the code under test.

## Build, Test, and Development Commands
- `cargo build --workspace --all-targets` compiles everything; add `-p <crate>` for focused builds.
- `cargo test --workspace` runs the full suite; `cargo test -p iris-mpc-cpu tests::` targets one crate.
- `just lint` runs `cargo fmt`, `cargo clippy`, and `cargo doc`; keep a clean slate before pushing.
- `just dev-pg-up` / `just dev-pg-down` control the Postgres instance used by HNSW.

## MPC Implementations: GPU vs HNSW
- GPU path (`iris-mpc/bin/server.rs`, `iris-mpc-gpu/src/server/actor.rs`) preloads shares into CUDA `ShareDB` slices, scans distances via cuBLAS GEMM, coordinates multi-GPU batches with NCCL, and shapes device inputs through `PreprocessedBatchQuery`.
- HNSW path (`iris-mpc/src/server/mod.rs`, `iris-mpc-cpu/src/execution/hawk_main.rs`) steers ABY3 sessions with `HawkActor` and drives `HnswSearcher` lookups backed by the Postgres `GraphPg` store.
- Matching differs: GPU thresholds every candidate chunk with `DistanceComparator`; HNSW narrows neighbors (`search_to_insert`, `match_count`) then rechecks via ABY3 distance MPC in `is_match_batch.rs`.
- State updates differ: GPU reloads device slices from `load_iris_db` and streams OR-policy masks, while HNSW derives insertion layers from a shared PRF and persists edges through `ConnectPlanV`.

## Coding Style & Naming Conventions
- Format with `cargo fmt`; `rustfmt.toml` targets Rust 2021 with 4-space indents and field init shorthand.
- Use `snake_case` for functions/modules, `CamelCase` for types/traits, and `SCREAMING_SNAKE_CASE` for constants; align enums and structs for readability.
- Run `cargo clippy --workspace --all-targets --all-features -D warnings`; fix lints instead of suppressing.

## Testing Guidelines
- Co-locate unit tests with implementation and keep scenario tests under `*/tests/`, named after their domain (e.g., `store_connection.rs`).
- Use deterministic RNG seeds in property tests and gate CUDA-sensitive cases behind opt-in features.
- Preserve CPU/GPU parity for protocol changes; run `cargo test --features gpu` for CUDA paths.

## Commit & Pull Request Guidelines
- Follow git history: imperative, present-tense subjects, optional ticket tags (e.g., `[POP-2993]`), and mention PR numbers when merging.
- Squash WIP before opening a PR; document the change, rollout plan, and linked issues.
- Attach screenshots or sample outputs for user-facing shifts and list the validation commands you ran.
