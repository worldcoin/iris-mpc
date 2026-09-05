# Exact CPU NTT linear scan

The opt-in implementation evaluates the dense anonymous-statistics predicate
directly in the prime field. On three r8g.24xlarge hosts it reaches 10.80 million
comp/s through the full service, versus 5.45 million for the original scan.
This is a 1.98x end-to-end improvement; the larger local CPU gain does not
translate into a 3x service speedup.

`IRIS_MPC_CPU_NTT=1` selects the integrated NTT implementation in
`iris-mpc-linear-scan`. The default is `0`, which uses the existing PR #2348
mixed-plane implementation. Set the same value on all three parties. Both
implementations use the production scheduler, pinned CPU workers, 31 rotations,
normal/mirror matching, anonymous statistics, second-eye policy, persistence,
and result publication. The switch is rejected for HNSW. The selected full-scan
eye must remain fixed, as required by the existing CPU linear-scan server.
For the measured Graviton4 allocation, also set
`SMPC__SEPARATE_TOKIO_CORES_PER_NODE=32`; enabling NTT alone retains the existing
core allocation and does not achieve the best measured throughput.

## Representation and score computation

The representation is lossless: 64 code sequences and 32 mask sequences, each
200 angular columns long. The 200-point NTT uses the prime 52,201 and root 43,061.
All frequencies and all original dimensions participate. Products are summed
across channels before the selected inverse; it produces all 31 accepted
rotations for the combined anonymous-statistics score `g=2*C-m`, where the
full mask overlap is `M=2m`. Code and mask spectra are combined before a single
inverse per orientation. There is no search index or approximation.
See [the algebra](exhaustive-spectral-matching.md) for the correlation identity,
paired positive/negative inverse, and SIMD accumulation bounds.

The resident database stores each field coefficient in two signed byte planes.
The payload is 38,400 bytes per record, eye, and party, excluding containers and
allocator overhead. Only the configured exhaustive eye is resident in this form.
Original shares for both eyes remain in Postgres, with bounded raw LUC/LFU caches
for sparse comparisons and identity operations. There is no additional complete
resident copy of the original database. At 1,048,576 records the spectral payload
is 37.5 GiB; 18 million records would require approximately 643.7 GiB before
runtime and cache overhead. The latter is a capacity estimate, not a benchmark.

On AArch64 with `i8mm`, the score engine batches eight database records and two
orientations through SMMLA. Rust controls loads, loop scheduling, and registers;
only the SMMLA instruction has an assembly wrapper because its intrinsic is
unstable on Rust 1.95. Other CPUs use an exact portable implementation. The
selected inverse shares sums and differences between positive and negative
rotations and uses widening NEON on AArch64.

## Private conversion of existing shares

A local `% 52201` operation cannot convert deployed Galois-ring shares. The
migration in `protocol::ntt::conversion` instead performs these steps jointly:

1. Apply the Galois-ring reconstruction weights and undo the basis transform,
   obtaining additive contributions to the original coordinates in Z/(2^16).
2. Privately refresh and reshare these contributions into replicated shares.
   A valid code coordinate is ternary, `v = LSB(v) - 2*MSB(v)`; mask coordinates
   are bits. Extract the sign with the existing binary MPC protocol.
3. Inject the XOR-shared bits into F_52201. Two masked field multiplications
   evaluate the XOR of the three components. Pairwise PRF masks are sampled
   uniformly by rejection sampling, rather than biased reduction of random u16s.
4. Convert replicated field components to degree-one Shamir evaluations at
   points 1, 2, 3 using local linear maps, then apply and pack the local NTT.

No iris coordinate is opened. This uses the existing semi-honest three-party
security model and assumes correctly encoded ternary codes and binary masks.
It does not prove client input validity or add active security.

For replicated additive component x_i, known to parties i and i+1, let z_m be
the evaluation point of the party missing that component. Assign the polynomial
`x_i * (z_m - z) / z_m`. It has constant x_i and is zero at the missing party.
Each party can therefore compute its evaluation locally. Summing the three
polynomials gives a degree-one sharing of the secret coordinate.

The query follows the same private conversion once per request, before the
concurrent normal/mirror scans. Query preprocessing folds in inverse
normalization, code/mask weights `(2,-1)`, and the degree-two reconstruction weights
`[3, -3, 1]`. Mirroring uses the original coordinate permutation and imaginary
component sign change.

## Prime-field anonymous-statistics predicate

The dense scan keeps scores in the prime field. The existing CPU predicate is
`4*C-M >= 0`, equivalent to `g=2*C-m >= 0`. Its valid integer range is
`[-32000,19200]`, so `Y=g+32000` lies in `[0,51200]` and fits F_52201.
The predicate is the canonical comparison `Y>=32000`, including equality and
zero overlap, preserving the existing fixed-point behavior.

`protocol::ntt::threshold` refreshes the local field contributions into Rep3.
An adaptation of ABY3's two-way split then binary-shares `u=(a+b) mod p` and
`v=c`: the pair-owning party reduces its local sum before XOR-masking it.
All split chunks are sent before receiving their independent neighbor chunks.
Neither the score nor the pair sum is opened. The integer `W=u+v` fits 17 bits,
and the exact predicate is

```text
[Y >= 32000] = [W >= 32000] XOR [W >= 52201] XOR [W >= 84201].
```

The addition and three public-threshold comparisons process bits together,
sharing equal low-bit prefixes and simplifying the final comparison to a mux,
using 46 secure AND gates in 17 layers. The original ring predicate uses 66
AND gates in 17 layers. The new split adds one communication phase, while
refreshing only one score reduces ideal batched payload from 12.375 to 8.542
bytes sent per party per rotation. These are circuit/payload counts, not
measured throughput. Uniform field masks use batched 32-bit PRG candidates,
rejecting values above an exact multiple of the prime to avoid modulo bias.
Only ordinary pairwise PRF masks are needed; there is no edaBit preprocessing
or recurring correlated-mask inventory.

The caller opens only the existing anonymous-statistics bits. Records with any
accepted rotation, together with forced reauthentication targets, are recovered
through the original ring worker path. All 31 rotations of each selected record
are included. The existing strict predicate, second-eye policy, and retained
code/mask distances then run unchanged. No full score conversion occurs in the
dense scan. Candidate recovery cost depends on the accepted-record count.

## Persistence, startup, and rollout

The SQL migration adds `cpu_spectral_irises`; original `irises` remain
authoritative. A derived row includes serial ID, version, eye, party, format
version, a public generation identifier, packed payload, and checksum. Format 2
fixes the prime, root, coordinate order, and packed layout. A representation
change requires a new format version.

At startup all parties compare ordered record/version manifests. A batch is
reused only if all parties have complete valid payloads from the same generation.
If one party has a missing, stale, corrupt, or differently generated cache,
all three privately reconvert the batch and write a fresh generation. Local
batch writes are transactional. A crash between parties' commits is repaired
by the next startup; old and fresh Shamir evaluations are never mixed.
Inconsistent authoritative versions fail startup instead of silently mixing
records. The service remains unready until all batches and the final three-party
barrier complete. Migration uses 64-record batches across at most 48 sessions. Each session runs
as a separate task so CPU-heavy conversion can use the runtime worker cores;
per-session record order remains deterministic across parties.

New insertions and identity updates persist the transformed query at the new
record version before publishing it to the in-memory scan. Deletion installs
the same public dummy iris as the existing implementation. A restart reconciles
these derived rows with authoritative raw persistence. Disabling the switch
restores the original CPU representation and leaves the derived table unused.
The migration is additive; rollback does not require reconstructing raw shares.

## Native-prime server performance

Measured on 2026-09-05 on three existing r8g.24xlarge hosts, using the same
service binary with the NTT switch enabled or disabled. Each run queued 128
independent requests and excluded the first 16 from the steady completion
interval. All 384 party responses in each run agreed and reported the expected
nonmatch. The database grew from approximately 1.049 million records through
successful insertions; the analyzer uses the exact count for every scan.
Both orientations and all 31 rotations are included. Timed runs did not overlap
compilation, profiling, or local kernel benchmarks.

| Path | Dot/runtime cores | Matching cascade comp/s | Queued service comp/s, after 16 warm-ups | Entire burst comp/s, from publish |
| --- | ---: | ---: | ---: | ---: |
| Original PR #2348 CPU path | 85/11 | 6,330,178 | 5,449,554 | 5,436,148 |
| Native-prime NTT | 85/11 | 7,213,134 | 5,780,650 | 5,671,347 |
| Native-prime NTT | 64/32 | 13,753,149 | 10,803,513 | 10,203,351 |
| Native-prime NTT | 48/48 | 12,903,935 | 10,252,428 | 9,625,695 |

The best tested allocation is 64 dot workers and 32 runtime/auxiliary cores.
It improves the matching cascade by 2.17x and queued service throughput by
1.98x. Including the entire burst from initial publish gives 1.88x. Including
client preparation as well gives 9,498,072 versus 5,226,705 comp/s (1.82x).
The queued intervals are 21.763 seconds for NTT and 43.139 seconds for the
baseline, each covering 112 completed requests. These are finite synthetic
bursts using Moto for AWS messaging, not a sustained production capacity claim.

At 85/11 the faster local scan exposes MPC runtime work on the critical path.
Reallocating cores matters despite the reduction from 66 to 46 AND gates:
the split, transposition, PRGs, copies, and communication still consume CPU.
Median server compute time falls from 334 ms to 169 ms with the 64/32 split;
request reception remains about 23 ms and persistence about 2 ms. The baseline
also had occasional cross-party start skew, up to 0.868 seconds; the 64/32 NTT
run's maximum was 0.0114 seconds. Earlier short diagnostic bursts showed
variable ingestion delays, motivating the equally warmed 128-request runs.

With the service stopped, a separate 85-worker local rerun on 1,048,576 distinct
resident payloads measured 24,344,294 comp/s for `linear_scan_ntt_cpu` and
6,249,701 comp/s for `linear_scan_dot_cpu`'s fused pair (nine measured passes
after one warm-up). The former times the production score kernel through Rayon;
the latter also includes the existing worker dispatch and collection of separate
code/mask outputs. Their respective medians were 86.146 ms and 335.560 ms for
both orientations. This confirms the local gain but is not a service comparison:
neither includes MPC, candidate recovery, query conversion, or persistence.

The initial private conversion and persistence of 1,048,737 records into
F_52201 took 644.57 seconds across the three parties. The subsequent 64/32
run loaded its cached spectral database in 12.68 seconds on party 0. These
timings cover the spectral startup phase; all records were resident before
request timing. The derived table used a temporary 100-GiB gp3 volume per party.
The benchmark measures approximately one million records, not 18 million.

## Historical conversion-heavy server performance

The following results measured the earlier F_25601 implementation, which
converted every score back to u16 before thresholding. They do not measure the
new F_52201 predicate. Measured on 2026-09-05 with three r8g.24xlarge servers and the full service path.
Each run queued 16 independent requests, with the first excluded from the steady
completion interval. The initial database had 1,048,609 records; successful
requests append records, and the analyzer uses the exact count for each scan.
Fresh deterministic query seeds avoid replay matches. The small size drift is
below 0.02%. Both normal and mirrored orientations scan all records.

| Path | Dot/runtime cores | Cascade comp/s | Queued comp/s, after warm-up | Entire burst comp/s, from publish |
| --- | ---: | ---: | ---: | ---: |
| PR #2348 baseline, A | 85/11 | 6,434,899 | 6,051,896 | 3,876,358 |
| PR #2348 baseline, B | 85/11 | 6,328,061 | 3,798,860 | 3,024,357 |
| PR #2348 baseline, C | 85/11 | 6,352,352 | 5,886,417 | 2,826,253 |
| NTT, original conversion | 85/11 | 862,723 | 848,023 | 793,004 |
| NTT, cheaper conversion | 85/11 | 1,259,620 | 1,227,709 | 1,096,104 |
| NTT, cheaper conversion | 48/48 | 2,404,885 | 2,297,530 | 1,928,395 |
| NTT, cheaper conversion | 32/64 | 2,408,085 | 2,300,488 | 1,996,798 |

The two-comparison converter improves throughput by about 45% at the unchanged
85/11 core allocation. More runtime cores raise it to about 2.30 million comp/s,
but the baseline's observed 3.80–6.05 million comp/s remains 1.65–2.63 times
faster. That earlier implementation did not justify changing the default.
The 23.3 million comp/s kernel result excludes the
private field-to-ring bridge and must not be used as server throughput.

Entire-burst timings include the first response delay; the steady interval starts
at warm-up completion. Including client preparation and its one-second S3 wait,
entire-burst rates were 2.58–3.43 million comp/s for the baseline and
1.81–1.87 million comp/s for the two rebalanced NTT runs. Queued per-request
latency grows with queue position, so dividing work by each request's latency
would not measure steady throughput.

The baseline's median server compute time was stable at 329–335 ms, while
completion timing varied. Run B included three requests with cross-party
cascade-start skew of 0.306, 0.601, and 0.893 seconds. Run C reused B's running
servers and returned to 5.89 million comp/s after warm-up, but had a longer
initial response delay. All three runs are included above; the cascade median
alone would hide these end-to-end delays. Runs B and C used the final binary
with the NTT switch disabled. These short synthetic bursts characterize this
test setup, not a production latency distribution or sustained capacity test.

An independent CPU profile of the original converter found 29.8% of user-cycle
samples in local scoring, with substantial remaining samples in bit extraction,
binary AND, transposition, copies, and TLS. These are CPU sample fractions, not
wall-time fractions across the differently sized worker pools. With 48 runtime
cores, measured NIC traffic reached approximately 34–35 Gb/s per direction; AWS
reports a 40-Gigabit network for this instance type. Increasing runtime cores to
64 did not improve throughput materially.

The initial private conversion and persistence of 1,048,625 records took
643.1 seconds on all three parties. Cached load phases took 16.3–19.4 seconds.
These are the spectral startup phase, excluding other server initialization.
Raw tables used the existing root volumes; the new derived table used a separate
100-GiB gp3 volume per party. Root WAL writes reached approximately 123 MB/s
against the 125 MB/s volume limit, so the one-time result includes storage costs.
All transforms were resident before request timing. This is a million-record
benchmark, not a measured 18-million-record deployment.

## Validation and reproduction

Focused checks:

```sh
cargo test -p iris-mpc-cpu --lib protocol::ntt --locked --no-default-features
cargo test -p iris-mpc-cpu --lib spectral_thresholds_and_retained_distances --locked --no-default-features
SMPC__DATABASE__URL=postgres://postgres@localhost/iris_test \
  cargo test -p iris-mpc-cpu --test spectral_migration --features db_dependent --locked
```

The protocol tests reconstruct all coordinate shares against real encoded iris
inputs, cover every field value with randomized three-party sums, and compare every rotation with original
Galois-ring dot products. The full pipeline tests include mirrored inputs,
identical irises, empty masks, tile tails, strict/anonymous thresholds, and
retained distances. The database test checks reuse, partial-generation recovery,
and rejection of mismatched authoritative manifests using three local parties. The widened inverse also has boundary-value SIMD tests.
A distributed production run of the final native-prime implementation completed
100 deterministic requests with no expected-result validation failures and 300
agreeing party responses: 42 uniqueness/mirror requests, 16 reauthentications,
8 deletions, 8 reset checks, 9 reset updates, 9 recovery checks, and 8 recovery
updates. This correctness run used a separate small database and is not included
in the million-record throughput measurements. After restarting all three
servers, four original queries again returned their original matched IDs on
every party; all three loaded the persisted spectral cache successfully.

The distributed benchmark uses three existing hosts with native PostgreSQL and
Moto providing S3/SNS/SQS/Secrets Manager. CPU settings match the production
allocation on r8g.24xlarge, with the runtime/dot split varied explicitly:
server batch size 1, request parallelism 48, connection parallelism 16, 4,096-record scan
chunks, 256-record worker tasks, LUC lookback 500, and LFU capacity 12,288.

```sh
LINEAR_SCAN_BENCH_NTT=1 \
LINEAR_SCAN_BENCH_TOKIO_CORES=32 \
LINEAR_SCAN_BENCH_DATABASE_SIZE=1048576 \
LINEAR_SCAN_BENCH_REQUEST_COUNT=128 \
LINEAR_SCAN_BENCH_WARMUP_REQUESTS=16 \
LINEAR_SCAN_BENCH_PIPELINED_REQUESTS=1 \
LINEAR_SCAN_BENCH_SSH_KEY=/path/to/key \
  scripts/run-distributed-linear-scan-benchmark.sh host0 host1 host2 /tmp/ntt-run
```

Use `LINEAR_SCAN_BENCH_NTT=0` and `LINEAR_SCAN_BENCH_TOKIO_CORES=11` with the
same binary and remaining settings for the original baseline allocation.
`LINEAR_SCAN_BENCH_REUSE_DB=1` preserves an already seeded benchmark database;
choose a fresh deterministic client seed for unrelated requests when reusing it.
Benchmark artifacts should remain outside the repository. The local NTT kernel
can be measured independently on Graviton4 with:

```sh
RUSTFLAGS='--cfg aes_armv8 -C force-frame-pointers=yes -Ctarget-cpu=neoverse-v2 -Ctarget-feature=+lse' \
IRIS_MPC_DOT_BENCH_DB_SIZE=1048576 IRIS_MPC_DOT_BENCH_RUNS=9 \
IRIS_MPC_NTT_BENCH_WORKERS=85 IRIS_MPC_NTT_BENCH_FIRST_CORE=11 \
  cargo bench -p iris-mpc-cpu --bench linear_scan_ntt_cpu --locked
```

One comparison means one record, one eye, one orientation, including all 31
rotations and all code/mask dimensions. Count the normal and mirror work, but do not
multiply by the three parties or by 31. The analyzer reports the search cascade
separately from client SNS-publish-to-all-three-responses latency. With queued
requests it reports completed comparisons over the interval between warm-up
completion and final completion, including persistence and result delivery.
Individual queued request latency is not itself a throughput measurement. The
analyzer also reports the entire burst, including its first response delay,
from both publish start and query preparation start. The latter includes the
client's fixed one-second S3 propagation wait.
