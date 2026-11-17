# GPU implementation retirement

CPU linear scan (`iris-mpc-linear-scan`) is the maintained matching service.
The retired implementation remains available in Git history before this change.

## Removed

- `iris-mpc-gpu/`: CUDA kernels, cuBLAS/NCCL communication, device memory and
  preprocessing, GPU actor, GPU-specific benchmarks and tests.
- `iris-mpc-gpu` and `nccl` binary targets, `gpu-reference` and `gpu-parity`
  features, and parity-only distance-capture instrumentation in the CPU actor.
- GPU-only startup ChaCha seed derivation and the CUDA library-path build script.
- `cudarc` and unused GPU-only lockfile dependencies. Other dependency versions
  and the pinned `ampc-common` revision are unchanged.
- GPU execution in `scripts/run-native-ground-truth.sh`. Its CPU run and JSON
  comparison modes remain available for CPU regression checks.

## Build and CI audit

The GPU test/image workflows and GPU Dockerfiles had already been retired.
The remaining workflows build CPU services, clients, genesis, database utilities,
resharing, retention and anonymous statistics. Their job names and image tags
remain unchanged, including the legacy `nocuda` helper-image interface.
`Dockerfile.nocuda` still builds CPU utilities; it is not a GPU image.

The all-targets, Clippy, documentation, unit-test and CPU integration workflows
remain in place. Workspace builds with all features no longer select CUDA or
NCCL. CPU Dockerfiles and Compose configurations retain both CPU linear-scan and
HNSW entrypoints. Internal build paths in the helper, encoding and resharing
Dockerfiles use `/src/iris-mpc`.

## Retained compatibility contracts

| Surface | Why it remains |
| --- | --- |
| `SMPC__DATABASE`, `SMPC__GPU_SCHEMA_NAME_SUFFIX`, `get_gpu_db_schema()` | CPU linear scan and candidate/reconciliation paths use the existing shares database and schema. Renaming would change deployed configuration. |
| `CommonConfig` fields and `SyncState::dh_nonce` | Shared startup serialization and configuration agreement remain compatible. Removing an unused field can change the agreement hash or serialized shape. |
| Result layouts, matching rules and CPU tests mentioning GPU semantics | These describe the behavior the CPU replacement must preserve, including partial matches, rotations, mirror handling, reset, recovery and reauthentication. |
| Anonymous-statistics GPU origins and non-lifted jobs | CPU-only processing of historical GPU distance rows. Retaining the iteration schedule also avoids changing the statistics service's inter-party protocol. Current CPU writers use HNSW origins and lifted distances. |
| Database migrations, S3 snapshot decoding, reapers and deployment values | Historical shares and journals remain readable; retention and maintenance jobs do not require a GPU. Database/resource deletion is a separate operational change. |
| Genesis, DB sanity checks and parked HNSW algorithms | CPU tooling still reads the source shares database, historically named the GPU database. |
| Existing resource names, including SNS topics and the `just` development DB | Names identify existing CPU-compatible resources; renaming is not required to remove GPU execution. |

No matching, ingestion, persistence, wire-format, database-schema or live resource
change is required for this source removal. Roll out CPU binaries and images
through their existing release paths. Rolling back this change restores source
and build targets; it does not require a data migration.
