# Exact exhaustive iris matching through circular correlation

Research note, 2026-09-05. This work is now based on
[PR #2348](https://github.com/worldcoin/iris-mpc/pull/2348), commit
`42cdb2ad0e41fea8c0e35d933ac02f54b065619b`, on branch `codex/cpu-spectral-ntt-smmla`.
The original measurements used the older stack baseline
`94e4a13a8c8a35d946989b603369da27c1dc8c87`.

**Latest CPU measurements on the updated stack:** On the target AWS r8g.24xlarge,
the packed SMMLA NTT score stage takes **90.1 ms** versus **277.7 ms** for
PR #2348's fused UMMLA kernel (**3.08x faster**), scanning 1,048,576 records,
both orientations, with 85 workers pinned to CPUs 11–95. The initial packing
and batching experiment cut **22.6%** from the previous paired NTT's 116.5 ms
to 90.2 ms in the same run. That run measured direct NTT at 118.1 ms and
precomputed f64 FFT at 660.3 ms. See the
[PR description](https://github.com/worldcoin/iris-mpc/pull/2386) for the full
performance table and methodology, and the
[benchmark README](../iris-mpc-cpu/benches/spectral-scan/README.md) for reproduction.
The earlier paired-inverse experiment found only 1.5% improvement from pairing itself.
These are local score times with production CPU allocation and 256-record
worker tasks; secure field conversion/comparison and service overhead are
excluded.
The GPU operation counts below describe the original research model, not CPU
measurements.

The strongest candidate found is to change the sharing field and store each
iris in a number-theoretic transform (NTT) representation. Compute the circular
correlation of query and database codes, and of their masks, to obtain all
rotation scores together. Every database record and every original dimension
participates. There is no index, candidate selection, dimensionality reduction,
or approximate match decision.

The arithmetic model gives **about 14 times fewer INT8 multiply-accumulates** for
the score stage, including a simple selected inverse transform. This is an
operation count, **not a measured GPU or end-to-end speedup**. The experiment
proves arithmetic equivalence with simulated three-party shares. A GPU kernel
and secure field conversion remain to be implemented and benchmarked.

## What the current implementation already does

- `iris-mpc-common/src/lib.rs`: 12,800 code coordinates, 6,400 mask coordinates,
  and 31 rotations.
- `iris-mpc-common/src/galois_engine.rs`, `encode_iris_code`: masked ternary
  encoding `m * (1 - 2*c)`, then degree-four Galois-ring sharing and basis changes.
  Its coordinate mapping exposes the original `(16, 200, 2, 2)` layout.
- `iris-mpc-gpu/src/dot/share_db.rs`, `ShareDB::dot`: three INT8 GEMMs for a
  dot product modulo 2^16. The high-high byte product vanishes in that ring.
- `iris-mpc-gpu/src/helpers/query_processor.rs`: separate code and mask dots;
  double the mask dot because real/imaginary mask values are duplicated.
- `iris-mpc-gpu/src/threshold_ring/protocol.rs`,
  `compare_threshold_masked_many`: lift the mask result, construct the threshold
  expression, and extract its sign bit. An OR-reduction implementation also
  already exists here.
- `iris-mpc-gpu/src/server/actor.rs`: the current full scan first runs one eye
  with the anonymous-statistics threshold, then processes selected records.
  The proposal replaces the exhaustive score engine; savings must be compared
  against this actual path, not an assumed two-eye full scan.

Replacing XOR/popcount with a dot product or adding an OR tree would therefore
repeat existing work. This proposal instead removes repeated work across the
31 rotations.

## Exact matching algebra

Let `c` be a code bit, `m` its validity bit, and `x = m(1-2c)`.
For one query/database alignment define:

\[
C = \sum_i x_{q,i}x_{d,i},\qquad
M = \sum_i m_{q,i}m_{d,i},\qquad
H = \sum_i (c_{q,i}\mathbin{\mathrm{xor}}c_{d,i})m_{q,i}m_{d,i}.
\]

Then `C = M - 2H`. The mathematical rule `H/M < 0.345` is, for positive overlap,

\[
100C > 31M.
\]

For zero overlap the fractional distance is undefined; the plaintext helper
does not report a match. A production replacement must preserve the selected
implementation's behavior rather than silently change its threshold semantics.

In particular, the current GPU kernel computes the sign of

\[
A M - 65536 C - 1,\quad A=20316
\]

for the match threshold, and uses `A=16384` for anonymous statistics. The `-1`
comes from `shared_lift_mul_sub` in the CUDA threshold kernel. This differs from
the strict rational rule at some boundaries. For example, `M=12800, H=4416`
is exactly 0.345, which the strict rational rule rejects and the GPU expression
accepts. The standalone GPU expression also accepts `C=M=0`; that observation
does not establish whether request validation allows such an input through the
whole service. This experiment preserves both score values and checks each
threshold rule separately. It does not change either rule.

## Represent rotation as circular correlation

Permute the original coordinates into 64 code sequences of length 200:
one sequence for each radial row, wavelength, and real/imaginary component.
There are 32 distinct mask sequences of length 200. Each mask applies to its two
corresponding code sequences. This is a lossless reordering of the input.

For angular offset `r`, the scores are

\[
C_r=\sum_{a=0}^{63}\sum_{t=0}^{199} x_q[a,t]x_d[a,t+r],
\]

\[
M_r=2\sum_{a=0}^{31}\sum_{t=0}^{199}m_q[a,t]m_d[a,t+r],
\]

where column indices wrap modulo 200. Correlation computes these sums for all
offsets together. Only offsets `-15..15` feed the existing match/eye policy;
accepting any of the other 169 offsets would change the matcher.

Ordinary floating-point FFTs on secret shares require a separate numerical error
argument. Instead use exact arithmetic in the prime field

\[
\mathbb F_{25601},\qquad 25601-1=128\cdot200.
\]

The validation script checks primality. `3` is a primitive root and
`omega = 3^128 mod 25601 = 9217` has order exactly 200. Thus an invertible
200-point NTT exists, using radix factors `2*2*2*5*5`.

With forward transform `X[k] = sum_t x[t] omega^(kt)`, compute

\[
\widehat C[k]=\sum_a X_q[a,-k]X_d[a,k],\qquad
\widehat U[k]=\sum_a F(m_q)[a,-k]F(m_d)[a,k].
\]

The inverse transforms give `C_r mod 25601` and `U_r=M_r/2 mod 25601`.
All 200 frequency coefficients are retained.

The channel sum happens **before the inverse transform**. For each code channel
let `S_a[k] = X_q[a,-k] X_d[a,k]`. Linearity allows the finite sums to exchange:

\[
C_r = \sum_{a=0}^{63}\left(200^{-1}\sum_{k=0}^{199}
S_a[k]\omega^{-kr}\right)
= 200^{-1}\sum_{k=0}^{199}\left(\sum_{a=0}^{63}S_a[k]\right)\omega^{-kr}.
\]

Thus one aggregate code spectrum yields every code rotation score, and one
aggregate mask spectrum yields every overlap score. A complete inverse returns
200 outputs; the selected inverse evaluates only the 31 accepted offsets.
Multiplication must precede channel aggregation: multiplying the separately
summed query and database spectra would introduce unwanted cross-channel terms.

This reduces 64 code inverses and 32 mask inverses to two inverses per target
and orientation. Spectral products remain 19,200 field multiplications. With a
direct 200-by-31 selected inverse, total multiplication counts fall from
`19200 + 96*200*31 = 614400` to `19200 + 2*200*31 = 31600`. These counts compare
the same selected-inverse algorithm; full mixed-radix inverses have different
costs, and neither comparison establishes a CPU latency ratio.

This is exact as an integer computation:

- `-12800 <= C_r <= 12800`; the field has precisely 25,601 elements, enough to
  represent every integer in that interval uniquely. Decode residues above
  12,800 as negative.
- `0 <= U_r <= 6400`; recover it as a nonnegative integer, then double it.
- Modular wraparound during transforms is intentional and harmless. No rounding
  or statistical distribution assumption is involved.
- Do not form `100C-31M` in this small field and interpret its residue's sign.
  That expression exceeds the field's unique signed range. Decode/convert the
  separate scores securely before evaluating the threshold.

The NTT/convolution identity is standard; the field choice and cost model here
are derived for this repository. See the
[NTT survey by Liang and Zhao](https://arxiv.org/abs/2211.13546)
for background on exact convolution through finite-field transforms.

## Compute positive and negative rotations together on CPU

Normalize the query spectrum by `200^-1` during query preparation. Consequently,
the locally accumulated spectrum `T[k]` already includes inverse normalization.
This is still a linear query transformation, and the stored database is unchanged.

For `k=1..99`, form the signed integer sums and differences

\[
A_k=T[k]+T[200-k],\qquad B_k=T[k]-T[200-k].
\]

For each `r=1..15`, precompute public field weights

\[
a_{r,k}=(\omega^{-rk}+\omega^{rk})/2,\qquad
b_{r,k}=(\omega^{-rk}-\omega^{rk})/2.
\]

Then evaluate both rotations using the same two dot products:

\[
E_r=T[0]+(-1)^r T[100]+\sum_{k=1}^{99} A_k a_{r,k},\qquad
O_r=\sum_{k=1}^{99} B_k b_{r,k},
\]

\[
C_r=E_r+O_r\pmod p,\qquad C_{-r}=E_r-O_r\pmod p.
\]

Rotation zero only needs `T[0]+T[100]+sum(A)`. Apply the same computation to
the mask spectrum, retaining its final doubling. All 200 spectrum entries and
all 31 allowed rotations remain represented exactly.

Inverse multiplication counts fall from `31*200=6200` to `15*2*99=2970`
per component. Including the code/mask spectral products, the score stage falls
from **31,600 to 25,140 multiplications**, a **20.44% reduction**. This excludes
one-time query normalization and does not predict wall-clock speedup.

The benchmark implements three paired variants. The simple one centers `A` and `B`
and reduces each intermediate dot before combining it. The version with deferred
reductions keeps these signed sums in `[-25600,25600]`, accumulates the two dots
as exact integers, and reduces only the final rotation outputs. This avoids
198 intermediate centering operations and 30 intermediate dot reductions
per component.

The deferred version is safe with the existing widening NEON schedule. A
99-term dot puts 96 terms through SIMD, six products in each of 16 independent
signed 32-bit accumulator lanes:

```text
6 * 25600 * 12800 = 1,966,080,000 < i32::MAX
```

The final three products and horizontal sums use signed 64-bit arithmetic.
Combining the two unreduced dots also fits in i64. Naively padding to 104 terms
would invalidate this 32-bit bound. The SIMD variant instead centers just the
three live tail terms, pads with five zeros, and uses this tighter bound:

```text
6 * 25600 * 12800 + 12800^2 = 2,129,920,000 < i32::MAX
```

It computes the tail in SIMD with no scalar products. Padding increases the
executed inverse multiplications to `15*2*104=3120` per component, including
zero products, or **25,440 total** for the score stage (19.49% below 31,600).
The mathematical nonzero work is unchanged. All three paired variants are
benchmarked separately.

Validation covers 37,324 arbitrary inverse outputs per paired variant, including
all 200 impulse positions, random spectra, signed extrema, DC, and the Nyquist
entry. Additional full-width direct-dot checks cover normal/mirror input arrays,
mask scaling, and target tails. These are arithmetic kernel checks; secure share
conversion, actual query ingestion, and the production match policy remain
outside the experiment.

## Pack the spectral products for SMMLA on CPU

Represent each centered field residue exactly as `v = lo + 256*hi`, where
`lo` is the signed low byte and `hi = (v-lo)/256`. Over `[-12800,12800]`,
`|lo| <= 128` and `|hi| <= 50`. This keeps two bytes per residue and the same
38,400-byte database payload per record, eye, and party. Each group of eight
coefficients is stored as eight low bytes followed by eight high bytes.

The new kernel computes eight database targets and both query orientations
together. Sixteen independent SMMLA accumulators compute all four byte products
for each target/orientation dot. Reconstruct each result in i64 as

```text
dot(v,w) = dot(lo_v,lo_w)
         + 256 * (dot(hi_v,lo_w) + dot(lo_v,hi_w))
         + 65536 * dot(hi_v,hi_w).
```

All four terms are necessary modulo 25,601. Each signed i32 partial over 64
channels has magnitude at most `64*128^2 = 1,048,576`; reconstruction and final
modular reduction use i64. Query preprocessing folds in frequency reversal,
inverse normalization, and mask doubling. The paired SIMD inverse is unchanged.

For eight targets, two orientations, and 64 channels, the spectral dot loop
uses 128 SMMLA instructions in place of 256 widening multiply-accumulate
instructions. It also shares target loads across orientations and query loads
across eight records. These changes are benchmarked together, so the measured
gain cannot be attributed solely to the instruction count.

The initial target-machine run gave 90.17 ms versus 116.46 ms for the previous paired
SIMD implementation. On 65,536 records, the new kernel takes 266.73 ms versus
491.69 ms on one worker, and 7.76 ms versus 10.10 ms on 85 workers. The smaller
gain at high parallelism motivates testing memory layout as well as arithmetic.
The byte representation roundtrips every field value; 15,998 additional scalar
oracle outputs validate the matrix kernel, full scores, and partial target tiles.

A controlled follow-up compared the handwritten SMMLA loop against Rust loops
with a single-instruction assembly wrapper. On 1,048,576 records and 85 workers,
the pooled medians over three rounds of 31 samples were 90.229 ms handwritten
and 90.088 ms Rust, effectively equal. On 65,536 records and one worker, three
rounds of 15 samples gave 265.915 ms handwritten and 249.874 ms Rust (6.0% lower
latency). Both versions used identical packing, query preparation, inverse code,
and input allocations. The compiler unrolled the eight-record loop, kept all
16 accumulators in registers, and introduced no matrix-loop spills. The current
implementation therefore removes handwritten scheduling and retains only the
SMMLA wrapper required by the supported stable Rust toolchain. These results
do not establish that scalar Rust would automatically select matrix instructions.

Two further experiments remain unmeasured: use matrix products across records
for the inverse's public weights, and store spectral tiles contiguously with
process-local huge-page advice. Neither changes the exhaustive matching rule.

## Three-party computation

Use independent degree-one Shamir shares over the new prime field for each
input coordinate, with evaluation points 1, 2, and 3. A single party's shares
remain independent of the secret, even though the field is small. Security here
comes from the sharing threshold, not treating the field size as a cryptographic
security parameter. Active-security checks require their own protocol design.

Each party can transform its own shares locally because the NTT is linear.
The input transformation is invertible, so it preserves the privacy of those
shares. Fresh independent sharing randomness must be used across coordinates
and records.

Local query/database products have degree two in the Shamir variable. Weight
the parties' products by the degree-two interpolation coefficients `(3,-3,1)`.
Their sum is the desired frequency-domain product. Apply the inverse transform
**locally before exchanging anything**: linearity means each party now holds
an additive field share of each rotation score. Retain only the 31 allowed
offsets, rerandomize, and then perform any required redistribution/conversion.
There is no need to communicate the 200 frequency results.

The current comparator expects replicated shares over powers of two. Prime-field
shares cannot be fed into it or cast to `u16` shares. Required options are:

1. Convert only the final code/mask scores securely to the existing ring, then
   reuse the existing exact threshold circuit.
2. Implement a comparator for additive prime-field shares with the same integer
   predicate. A correctness-first circuit can privately add the three canonical
   residues, reduce modulo the prime, decode the signed code value, and evaluate
   the threshold. Whether this is cheaper requires circuit and network measurements.

Raw local products must not be redistributed without rerandomization: that can
leak information beyond the final score. The script reconstructs scores only in
its synthetic test oracle; it is not a complete secure protocol implementation.

Existing Galois-ring database shares also cannot be converted by applying `%
25601` locally. Their modulus and basis encoding differ. Deployment requires a
secure migration or newly encoded shares, plus query/update/serialization parity.
Mirroring and the two-eye AND/OR policies must continue to use the same coordinate
permutations, sign changes, and record identities as today. These service paths
are outside the arithmetic experiment's validation scope.

## GPU cost and layout

Precompute each database record's transform once at insertion, and transform
each query once. Store data frequency-major within scan tiles. For each frequency
use batched GEMM to compare every database record with every query: the reduction
dimension is 64 for code and 32 for masks. The batching dimension covers 200
frequencies. Query rotations are no longer expanded into 31 separate columns.

The prime-field residues still fit in two bytes, so stored iris shares can retain
the same 38,400-byte payload per eye, record, and party. Keep transformed storage
instead of a second complete copy. Extra caches and migration overlap would add
memory beyond this payload calculation.

For an exact INT8 implementation, center residues into `[-12800,12800]`, then
split `v = lo + 256*hi`, with signed `lo` in `[-128,127]` and `hi` in `[-50,50]`.
Unlike the existing power-of-two ring, this field needs **all four byte products**.
Each partial dot fits in INT32 for reduction lengths 32, 64, and 200, but the
weighted recombination can exceed INT32. Combine in INT64 and reduce modulo
25,601. Reusing the current overflowing INT32 accumulation with scaled GEMM
alphas would be incorrect in this field.

A simple selected inverse uses a fixed public `200 x 31` matrix whose entries
are `200^-1 * omega^(-kr)`. Use the same four-product decomposition for this
matrix multiplication, or benchmark a mixed-radix inverse instead. Standard
INT8 inputs with INT32 accumulation are documented in
[NVIDIA's cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex).

Ideal unpadded counts per query/database eye pair, per party:

| Score-stage work | INT8 multiply-accumulates |
| --- | ---: |
| Current: `3 * 31 * (12800 + 6400)` | 1,785,600 |
| Spectral products: `4 * 200 * (64 + 32)` | 76,800 |
| Two selected inverses: `4 * 2 * 200 * 31` | 49,600 |
| Spectral total | 126,400 |

This is a 14.13x reduction in counted INT8 MACs. Padding the inverse's 31 outputs
to 32 gives 128,000 MACs and a 13.95x reduction, before other layout padding.
The multiplication of two private inputs alone drops by a factor of 31 in
field-operation count, but prime arithmetic and inverse transforms consume
part of that gain.

Reasons the wall-clock improvement will be smaller, or could disappear:

- The existing wide GEMM has query width `31*batch_size`; the frequency GEMMs
  have width `batch_size` and much shorter reduction dimensions. Tensor-core
  utilization and launch overhead need measurement with batched/grouped kernels.
- Database bytes read per scan do not automatically decrease. A transfer-bound
  workload gets little benefit from fewer MACs.
- Two spectra require 400 intermediate field elements per pair, versus 62 final
  scores, about 6.45x as many. Tile the computation and inverse rather than
  allocating spectra for the whole database. Layout changes and transposes cost
  bandwidth.
- Prime reduction, INT64 recombination, query transforms, and score conversion
  are additional work. The current efficient power-of-two threshold lifting
  relies on arithmetic properties that do not carry over automatically.
- There are still 31 threshold decisions per pair. Communication for comparison
  does not disappear merely because the score engine is faster.

Measure `T_new = T_frequency + T_inverse + T_conversion + T_threshold + T_IO`
against the actual existing pipeline. The MAC ratio must not be substituted for
an end-to-end speedup.

## Validation and next engineering step

Run the dependency-free correctness experiment:

```sh
python3 scripts/verify_spectral_iris_scan.py
```

It verifies primality/root order, random transform round trips, all residue byte
decompositions, random and extreme four-product dots, and 2,400 exact code/mask
score pairs using simulated three-party Shamir shares. The 12 synthetic cases
cover unrelated and identical codes, both signed endpoints, zero overlap, one
visible mask pair, circular wraparound, rotations at +/-15 and outside the
allowed window, and values at/adjacent to the 0.345 boundary. Both GPU threshold
expressions are checked against independently calculated plaintext scores.

The script is deliberately not a timing comparison between Python loops. No CUDA
benchmark, secure score-conversion protocol, end-to-end privacy proof, or
CPU/GPU service-parity test has been completed. Production Rust/CUDA code is
unchanged.

The next experiment should benchmark the frequency GEMMs, tiled inverse,
modular reduction, and final-score conversion at the actual batch/chunk sizes.
Profile code and mask GEMM time, transfers, and threshold time separately first.
Only pursue the database migration if the complete measured path beats the
current scan.

## Other approaches considered

Two small prime fields, for example 251 and 241, can represent ordinary dot
products via CRT with two INT8 GEMMs instead of three: their product 60,491
exceeds the full signed score range. This is an exact fallback with a smaller
potential gain (1.5x for dot arithmetic), but secure CRT reconstruction may
consume the savings. It does not share work across rotations.

Dropping transform coefficients, learning a compact embedding, or taking only
Fourier magnitudes would change the matching rule. In particular, magnitudes
discard phase needed for the exact correlation. Packing more bits into a scalar
also does not let ordinary integer carries replace secret per-bit operations.

Aggregating scores before thresholding is not generally equivalent to asking
whether any row matches: positive and negative margins can cancel. A recent
[threshold-FHE iris matching paper](https://arxiv.org/html/2601.17561v1)
investigates polynomial folding to reduce later work, but uses a gap around the
threshold and a distribution/uniqueness assumption for folding. It is useful
related research, not evidence of an exact replacement for this matcher.
