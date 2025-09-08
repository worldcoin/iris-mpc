Profiling helper scripts
========================

Prereqs
- Run hawk services with the profiling build and extra caps:
  - docker compose -f docker-compose.test.yaml -f docker-compose.profiling.override.yaml up --build
- Ensure host allows perf events if you intend to use perf: sysctl kernel.perf_event_paranoid<=2

Quick start (3 nodes)
- ./scripts/profiling/quick_all.sh 30
  - Collects pprof flamegraphs (SVG) and protobuf profiles from ports 3000/3001/3002.
  - Captures NUMA and TCP snapshots per container.

Individual scripts
- collect_pprof.sh <host> <port> [seconds=30] [frequency=99] [out_prefix]
  - Saves <prefix>.flame.svg and <prefix>.pprof
- collect_perf.sh <container_name> [seconds=60] [frequency=99] [out_dir=profiles]
  - Best-effort perf record + perf.script inside the container (requires perf installed in image)
- collect_offcpu.sh <container_name> [seconds=60] [out_dir=profiles]
  - Best-effort off-CPU folded stack capture using offcputime-bpfcc if available
- snapshot_node.sh <container_name> [out_dir=profiles]
  - Captures numastat, numa_maps, and ss -ti

Notes
- Prefer pprof endpoints for flamegraphs in environments where installing perf/bcc is difficult.
- To render off-CPU folded stacks into SVG, use FlameGraph's flamegraph.pl or inferno offline.
- For NUMA experiments, run hawk containers with cpuset constraints or launch processes under numactl.

Continuous S3 collector
- Embedded (preferred): built into `iris-mpc-hawk` as an optional background task.
  - Enable via env: set `SMPC__ENABLE_PPROF_COLLECTOR=true` in the hawk pod.
  - Per-batch capture: set `SMPC__ENABLE_PPROF_PER_BATCH=true` to auto-profile exactly while each batch runs (produces `per-batch/` artifacts).
  - Optional env overrides:
    - `SMPC__PPROF_S3_BUCKET` (default: `wf-smpcv2-stage-hnsw-performance-reports`)
    - `SMPC__PPROF_PREFIX` (default: `hnsw/pprof`)
    - `SMPC__PPROF_RUN_ID` (default: `run-<UTC>`)
    - `SMPC__PPROF_SECONDS` (default: `30`)
    - `SMPC__PPROF_FREQUENCY` (default: `99`)
    - `SMPC__PPROF_IDLE_INTERVAL_SEC` (default: `5`)
    - `SMPC__PPROF_FLAME_ONLY` or `SMPC__PPROF_PROFILE_ONLY` (default: false)
  - Targets: hits its own `/pprof/*` endpoints on `http://localhost:$SMPC__HAWK_SERVER_HEALTHCHECK_PORT`.
  - S3 key format (same as standalone):
    - `<prefix>/<run_id>/<target>/<UTC-ISO>_<seconds>s_<frequency>Hz.flame.svg`
    - `<prefix>/<run_id>/<target>/<UTC-ISO>_<seconds>s_<frequency>Hz.profile.pprof`
    - Example: `stage/hnsw/loadtest-2025-08-27/party0/2025-08-27T14-12-00Z_30s_99Hz.flame.svg`
  - Per-batch keys:
    - `<prefix>/<run_id>/<target>/per-batch/<UTC-ISO>_batch-<hash4>_dur<sec>s_freq<Hz>Hz.flame.svg`
    - `<prefix>/<run_id>/<target>/per-batch/<UTC-ISO>_batch-<hash4>_dur<sec>s_freq<Hz>Hz.profile.pprof`

- Standalone: `pprof-collector` (under `iris-mpc-common`, uses AWS SDK + reqwest)
  - Purpose: polls `/pprof/flame` and `/pprof/profile` on one or more targets, uploads artifacts to S3 with timestamped keys for later analysis.
- Example:
  - `pprof-collector --bucket wf-smpcv2-stage-hnsw-performance-reports \
      --prefix stage/hnsw --run-id loadtest-2025-08-27 \
      --target coordinator=http://hawk-coordinator:3000 \
      --target party0=http://hawk-participant-0:3001 \
      --target party1=http://hawk-participant-1:3002 \
      --seconds 30 --frequency 99 --idle-interval-sec 5`
- S3 key format:
  - `<prefix>/<run_id>/<target>/<UTC-ISO>_<seconds>s_<frequency>Hz.flame.svg`
  - `<prefix>/<run_id>/<target>/<UTC-ISO>_<seconds>s_<frequency>Hz.profile.pprof`
  - Example: `stage/hnsw/loadtest-2025-08-27/party0/2025-08-27T14-12-00Z_30s_99Hz.flame.svg`
- Env vars supported:
  - `PPROF_S3_BUCKET`, `PPROF_S3_PREFIX`, `PPROF_RUN_ID`, `PPROF_TARGETS`, `PPROF_SECONDS`, `PPROF_FREQUENCY`, `PPROF_IDLE_INTERVAL_SEC`
- Run placement:
  - Embedded: just set the env vars on the hawk deployment.
  - Helm values overlay:
    - Use `deploy/snippets/pprof-collector.values.yaml` and apply alongside your party values.
    - Example: `helm upgrade <release> <chart> -f deploy/stage/ampc-hnsw-0-stage/values-ampc-hnsw.yaml -f deploy/snippets/pprof-collector.values.yaml --namespace <ns>`
    - Update `podAnnotations.loadtest/run-id` to uniquely mark each test run.
  - Sidecar (optional alternative):
    - env:
      - name: PPROF_S3_BUCKET
        value: wf-smpcv2-stage-hnsw-performance-reports
      - name: PPROF_S3_PREFIX
        value: stage/hnsw
      - name: PPROF_RUN_ID
        valueFrom:
          fieldRef:
            fieldPath: metadata.annotations['loadtest/run-id']
      - name: PPROF_TARGETS
        value: self=http://localhost:$(SMPC__HAWK_SERVER_HEALTHCHECK_PORT)
      - name: PPROF_SECONDS
        value: "30"
      - name: PPROF_FREQUENCY
        value: "99"
      - name: PPROF_IDLE_INTERVAL_SEC
        value: "5"
    - image: your-built-image-containing-pprof-collector
      name: pprof-collector
      command: ["/app/pprof-collector"]
