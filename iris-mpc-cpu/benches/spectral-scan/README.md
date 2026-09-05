# Exhaustive spectral matching on CPU

This standalone benchmark is based on [PR #2348](https://github.com/worldcoin/iris-mpc/pull/2348),
commit `42cdb2ad0e41fea8c0e35d933ac02f54b065619b`, on branch
`codex/cpu-spectral-ntt-smmla`. It compares the PR's actual fused UMMLA score kernel
with precomputed f64 FFT and exact NTT representations. It also evaluates the
proposed paired positive/negative rotation inverse, including deferred modular
reductions and SIMD tail handling, and a packed signed-byte SMMLA spectral kernel.

On the target r8g.24xlarge, the initial packed SMMLA NTT score stage took **90.2 ms**
versus **277.7 ms** for the PR kernel (**3.08x faster**) at 1,048,576
records and both orientations. It cuts another **22.6%** from the previous
paired NTT's 116.5 ms, with the same database payload size and exact field scores.

See the [PR description](https://github.com/worldcoin/iris-mpc/pull/2386) for
the performance measurements, including comparisons per second and methodology.
The paired-inverse experiment found only 1.5% improvement over direct inversion
despite its lower multiplication count. Reproduction commands are below.

## Compiler-managed SMMLA loop

The current packed kernel uses Rust loops and NEON load/store intrinsics. Only
the SMMLA instruction uses a small inline-assembly wrapper, because Rust 1.95's
`vmmlaq_s32` intrinsic is unstable. The compiler controls register allocation,
loads, loop unrolling, and scheduling. The handwritten kernel loop was removed
after a controlled comparison on the same r8g.24xlarge.

Both candidates used identical packing, eight-record/two-orientation tiles,
query preparation, inverse code, worker tasks, and database allocations in one
test binary. Three rounds rotated method order and varied four query inputs,
with two warmups per round. The million-record run used 31 measured samples per
round; the smaller runs used 15. The table pools the measured samples across
rounds. Each comparison includes all 31 rotations and both code/mask scores
for one database record and one query orientation.

| Records | Workers | Handwritten (ms) | Rust (ms) | Handwritten (comp/s) | Rust (comp/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | 85 | 90.229 | 90.088 | 23,242,467 | 23,278,855 |
| 65,536 | 1 | 265.915 | 249.874 | 492,909 | 524,552 |
| 65,536 | 85 | 7.679 | 7.616 | 17,068,440 | 17,210,755 |

The million-record result is effectively tied (0.16% lower latency with Rust);
the one-worker result is 6.0% faster by latency. The PR #2348 baseline in the
same million-record run took 277.707 ms, or 7,551,677 comp/s. The compiler
unrolled the eight-target loop and kept all 16 accumulators in registers without
spills in the matrix loop. Both candidates passed the same 15,998 additional
scalar-oracle outputs and produced equal checksums for every timed sample.
This supports removing handwritten scheduling while retaining the packed
representation and matrix instructions. The experiment did not test whether
scalar Rust alone would automatically select SMMLA.

## Workload and timing boundary

Every method scans every database record, all 12,800 code coefficients, all
6,400 stored mask coefficients, and all 31 accepted rotations. There is no
index, candidate selection, dimensionality reduction, or approximate score.
Database representations are precomputed and resident before timing. Query
preparation, spectral products, inverse evaluation, local output allocation,
and checksum consumption are inside the timing boundary.

The target-machine experiment uses one query eye, normal and mirrored
orientations together, 85 workers pinned to CPUs 11–95, main thread on CPU 0,
and 256-record worker tasks. Production's 4,096-record MPC batches are split
into those worker tasks; the harness benchmarks the worker computation.

These are **local per-party score-stage times**, not complete MPC requests.
Actor dispatch, persistence, network communication, rerandomization, resharing,
field conversion, threshold comparison, match reduction, and two-eye policy
are not measured. Common query ingestion/Lagrange processing and geometric
mirror construction are also outside the timing boundary.

## Compared implementations

| CSV method | Implementation |
| --- | --- |
| `legacy_neon` | The older row-major u16 NEON kernel retained in the PR; two independent orientation traversals. |
| `pr2348_ummla_pair` | The PR's complete mixed-plane module: doubled queries, per-worker query caches, packed pair UMMLA, fused normal/mirror traversal, mask scaling, and tail fallback. |
| `fft_f64` | RustFFT 6.4.1, 101 complex f64 bins per real stream; spectral multiplication, two full inverse transforms, rounding, and reduction modulo 65,536. |
| `ntt_direct` | Exact field p=25,601, 200 frequency bins, independent selected inverse for each of the 31 rotations. |
| `ntt_paired` | Pair +r/-r using 99 spectrum sums/differences, with centered inputs and intermediate reductions. |
| `ntt_paired_lazy` | The same pairing with unreduced signed sums/differences and only final-output modular reductions. |
| `ntt_paired_simd` | Deferred reductions, plus a safely padded 104-term dot that eliminates the three scalar tail products. |
| `ntt_smmla` | Signed-byte spectral products for eight targets and both orientations together, using SMMLA and the paired SIMD inverse. |

`build.rs` extracts the production u16 kernels, the entire mixed-scan module,
the rotation constants, and `MixedPlaneIris` plus its conversion directly from
the checked-out PR source. It exposes the mixed module and renders one
service-only documentation link as plain code; executable code is unchanged.
`src/protocol.rs` supplies minimal share containers so the benchmark can compile
without the service dependency graph. The production assembly and packing
loops are not rewritten. This comparison requires AArch64 with `i8mm`.

The first four NTT variants use the same frequency-major database and traversal:
for each target, compute normal and mirror on the same worker before advancing.
This reuses the resident target across orientations, following the PR's fused
traversal approach. It differs from the earlier NTT harness, which traversed
the database separately for each orientation. Comparing the NTT variants in
the same run isolates the inverse choice from that traversal change.

`ntt_smmla` repacks the same residues into signed low/high byte planes and
processes eight targets and both orientations per frequency. It shares query
loads across eight records, uses 16 independent matrix accumulators, and folds
frequency reversal, inverse normalization, and mask doubling into query preparation.

NTT stores centered i16 residues, using the same 38,400-byte payload per
record/eye/party as the raw and mixed-plane layouts. f64 FFT needs 155,136
bytes, or 4.04x as much memory. Both NTT inverses and SIMD accumulation bounds
are described in the [algebra note](../../../docs/exhaustive-spectral-matching.md).
The paired inverse folds `200^-1` into query preparation. It preserves all
scores exactly over the prime field; it does not change accepted rotations.

## Validation

`--validate-only` checks:

- 37,324 arbitrary inverse outputs for each of the three paired variants,
  including every spectrum impulse position, DC/Nyquist entries, signed extrema,
  and deterministic random spectra.
- 4,774 full-width scores against independent scalar FFT/ring and NTT/field
  oracles, plus 8,122 FFT scores against tiled NEON.
- 53,196 score outputs for the PR kernel against legacy NEON, and for the first
  four NTT variants against direct field dots, including normal/mirror input arrays,
  singleton cases, full groups, and target/rotation tails.
- 132 mixed-plane storage roundtrips, 40 signed SIMD accumulation boundary
  cases, and every signed sum in [-25,600,25,600] for centering.
- 15,998 additional SMMLA field outputs against independent scalar dots,
  including signed extrema, both channel widths, both orientations, and tile
  tails; signed-byte roundtrip checks for all 25,601 field representatives.
- Every measured repetition checks equality of legacy/PR/FFT checksums and
  equality across all NTT variants. Checksums consume every score element.

The largest observed FFT error before rounding was 0.000076294. This is an
empirical check, not an all-input rounding proof. NTT uses exact arithmetic.

The benchmark generates deterministic synthetic local coefficients. Applying
`% 25601` to those values is test-data generation, **not a secure conversion
of deployed Galois-ring shares**. NTT requires correctly encoded prime-field
shares and a secure conversion/comparison protocol before deployment. Its
checksums consequently differ from the current ring's checksums.

## Build and run

From the repository root on the target CPU:

```sh
CARGO_TARGET_DIR=/tmp/iris-spectral-target RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --locked \
  --manifest-path iris-mpc-cpu/benches/spectral-scan/Cargo.toml

/tmp/iris-spectral-target/release/iris-spectral-cpu-bench --validate-only

/tmp/iris-spectral-target/release/iris-spectral-cpu-bench \
  --sizes 1048576 --threads 85 --chunk 256 --orientations 2 \
  --cpu-list 11-95 --main-cpu 0 --preprocess-threads 85 --reps 15
```

The harness retains raw, mixed-plane, FFT, i16 NTT, and packed NTT forms together:
301.5 GiB of payload at 1,048,576 records, plus allocation overhead. Each timed
method scans only its own representation. The first four NTT variants share one
database; SMMLA uses a second, equally sized byte-plane representation.
Use a smaller `--sizes` value on machines without sufficient RAM. Explicit
CPU affinity is implemented for Linux; omit those flags for local macOS runs.

Two warmups precede each set of samples. Method order rotates, queries vary
among four deterministic inputs, and persistent Rayon pools supply the worker
budget. Every sample, including warmups, is logged to stderr. CSV medians,
minima, maxima, throughputs, and checksums go to stdout. Offline transform time
and isolated query-preparation time for the earlier methods are recorded separately.
SMMLA query preparation is included in its total score time; it has no separate
query-preparation microbenchmark in this run.
