#!/usr/bin/env python3
"""Exact arithmetic experiment for docs/exhaustive-spectral-matching.md.

Uses synthetic plaintext and simulated three-party Shamir shares, with no third
party dependencies. This is a correctness oracle, not production MPC or a GPU
benchmark. Reconstructing scores here is for testing only.
"""

from functools import lru_cache
from math import isqrt
from random import Random


P = 25_601
COLS = 200
CODE_ROWS = 64
MASK_ROWS = 32
ROTATIONS = range(-15, 16)
LAGRANGE = (3, -3, 1)  # Degree-two interpolation at zero, points 1, 2, 3.
THRESHOLD_A = int((1.0 - 2.0 * 0.345) * (1 << 16))


def primitive_root():
    # P - 1 = 2**10 * 5**2.
    return next(g for g in range(2, P) if all(
        pow(g, (P - 1) // factor, P) != 1 for factor in (2, 5)
    ))


ROOT = pow(primitive_root(), (P - 1) // COLS, P)


@lru_cache(maxsize=None)
def twiddles(n, root, radix):
    return tuple(tuple(pow(root, j * k, P) for j in range(radix))
                 for k in range(n))


def ntt(values, root=ROOT):
    """Mixed radix 2/5 NTT, forward convention sum_t x[t] * root**(k*t)."""
    n = len(values)
    if n == 1:
        return [values[0] % P]
    radix = 2 if n % 2 == 0 else 5
    assert n % radix == 0
    size = n // radix
    parts = [ntt(values[j::radix], pow(root, radix, P)) for j in range(radix)]
    return [sum(parts[j][k % size] * weights[j] for j in range(radix)) % P
            for k, weights in enumerate(twiddles(n, root, radix))]


def inverse_ntt(values):
    scale = pow(COLS, -1, P)
    return [v * scale % P for v in ntt(values, pow(ROOT, -1, P))]


def signed_limbs(value):
    value = value % P
    if value > P // 2:
        value -= P
    low = (value + 128) % 256 - 128
    return low, (value - low) // 256


def limb_dot(left, right):
    """Four signed INT8 dots, then WIDE recombination and prime reduction."""
    a = [signed_limbs(v) for v in left]
    b = [signed_limbs(v) for v in right]
    partials = [sum(x[i] * y[j] for x, y in zip(a, b))
                for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    assert all(-(1 << 31) <= v < (1 << 31) for v in partials)
    ll, lh, hl, hh = partials
    return (ll + 256 * (lh + hl) + 65536 * hh) % P


def share_and_transform(rows, rng):
    """Independent degree-one Shamir sharing per input coordinate."""
    out = [[], [], []]
    for row in rows:
        slopes = [rng.randrange(P) for _ in row]
        for party in range(3):
            out[party].append(ntt([
                (v + (party + 1) * slope) % P
                for v, slope in zip(row, slopes)
            ]))
    return out


def shared_correlation(query_shares, database_shares):
    """Each party does its own frequency products and inverse transform.

    Only then are final score shares reconstructed in this test harness.
    Production must rerandomize before redistribution and keep scores secret.
    """
    partials = []
    for party, weight in enumerate(LAGRANGE):
        spectrum = [weight * sum(q[-k % COLS] * d[k] for q, d in zip(
            query_shares[party], database_shares[party]
        )) % P for k in range(COLS)]
        partials.append(inverse_ntt(spectrum))
    return [sum(part[r] for part in partials) % P for r in range(COLS)]


def encode(bits, masks):
    # Pairing real/imaginary rows is a permutation of the original layout.
    return [[masks[row // 2][col] * (1 - 2 * bits[row][col])
             for col in range(COLS)] for row in range(CODE_ROWS)]


def direct(bits_q, masks_q, bits_d, masks_d, rotation):
    overlap = mismatches = 0
    for row in range(MASK_ROWS):
        for col in range(COLS):
            other_col = (col + rotation) % COLS
            if masks_q[row][col] and masks_d[row][other_col]:
                overlap += 2
                mismatches += bits_q[2 * row][col] ^ bits_d[2 * row][other_col]
                mismatches += bits_q[2 * row + 1][col] ^ bits_d[2 * row + 1][other_col]
    return overlap - 2 * mismatches, overlap


def gpu_match(code, mask, a=THRESHOLD_A):
    # Mirrors the integer expression in shared_lift_mul_sub, including -1.
    return a * mask - (1 << 16) * code - 1 < 0


def rational_match(code, mask):
    return mask > 0 and 100 * code > 31 * mask


def rotate(rows, amount):
    return [row[-amount % COLS:] + row[:-amount % COLS] for row in rows]


def run_case(name, query, database, rng):
    bits_q, masks_q = query
    bits_d, masks_d = database
    code_q = share_and_transform(encode(bits_q, masks_q), rng)
    code_d = share_and_transform(encode(bits_d, masks_d), rng)
    mask_q = share_and_transform(masks_q, rng)
    mask_d = share_and_transform(masks_d, rng)
    codes = shared_correlation(code_q, code_d)
    masks = shared_correlation(mask_q, mask_d)
    actual, expected = [], []
    for r in range(COLS):
        c = codes[r] if codes[r] <= P // 2 else codes[r] - P
        m = 2 * masks[r]
        ref_c, ref_m = direct(bits_q, masks_q, bits_d, masks_d, r)
        assert (c, m) == (ref_c, ref_m), (name, r, c, m, ref_c, ref_m)
        assert rational_match(c, m) == rational_match(ref_c, ref_m)
        for a in (THRESHOLD_A, 16_384):  # Match and anonymous-stats thresholds.
            assert gpu_match(c, m, a) == gpu_match(ref_c, ref_m, a)
        actual.append(gpu_match(c, m))
        expected.append(gpu_match(ref_c, ref_m))
    assert any(actual[r % COLS] for r in ROTATIONS) == any(
        expected[r % COLS] for r in ROTATIONS
    )
    print(f"PASS {name}: all 200 score pairs and both GPU thresholds", flush=True)
    return actual


def main():
    rng = Random(20260905)
    assert all(P % d for d in range(2, isqrt(P) + 1))
    assert pow(ROOT, COLS, P) == 1
    assert pow(ROOT, COLS // 2, P) != 1
    assert pow(ROOT, COLS // 5, P) != 1
    for _ in range(20):
        values = [rng.randrange(P) for _ in range(COLS)]
        assert inverse_ntt(ntt(values)) == values
    for v in range(P):
        lo, hi = signed_limbs(v)
        assert -128 <= lo <= 127 and -128 <= hi <= 127
        assert (lo + 256 * hi) % P == v
    for size in (32, 64, 200):
        for _ in range(100):
            left = [rng.randrange(P) for _ in range(size)]
            right = [rng.randrange(P) for _ in range(size)]
            assert limb_dot(left, right) == sum(x * y for x, y in zip(left, right)) % P
        endpoint = [P // 2] * size
        assert limb_dot(endpoint, endpoint) == sum(x * x for x in endpoint) % P
    print(f"P={P}, primitive root={primitive_root()}, order-200 root={ROOT}", flush=True)

    zeros = [[0] * COLS for _ in range(CODE_ROWS)]
    ones = [[1] * COLS for _ in range(CODE_ROWS)]
    full_mask = [[1] * COLS for _ in range(MASK_ROWS)]
    empty_mask = [[0] * COLS for _ in range(MASK_ROWS)]

    def random_iris():
        return ([[rng.randrange(2) for _ in range(COLS)] for _ in range(CODE_ROWS)],
                [[int(rng.random() < 0.9) for _ in range(COLS)] for _ in range(MASK_ROWS)])

    query = random_iris()
    cases = [
        ("unrelated", query, random_iris()),
        ("identical", query, query),
        ("positive endpoint +12800", (zeros, full_mask), (zeros, full_mask)),
        ("negative endpoint -12800", (zeros, full_mask), (ones, full_mask)),
        ("empty mask", (zeros, empty_mask), (ones, full_mask)),
        ("rotation +15", query, tuple(rotate(rows, 15) for rows in query)),
        ("rotation -15", query, tuple(rotate(rows, -15) for rows in query)),
        ("rotation +16 outside accepted window", query,
         tuple(rotate(rows, 16) for rows in query)),
    ]
    # Exactly 0.345 Hamming distance and its nearest neighbors at full overlap.
    for mismatch_count in (4415, 4416, 4417):
        bits = [[int(row * COLS + col < mismatch_count) for col in range(COLS)]
                for row in range(CODE_ROWS)]
        cases.append((f"threshold H={mismatch_count}, M=12800",
                      (zeros, full_mask), (bits, full_mask)))
    sparse_mask = [[0] * COLS for _ in range(MASK_ROWS)]
    sparse_mask[0][199] = 1
    cases.append(("one visible pair, wraparound", (zeros, sparse_mask),
                  (ones, rotate(sparse_mask, 1))))

    for name, q, d in cases:
        matches = run_case(name, q, d, rng)
        if name.startswith("rotation"):
            rotation = int(name.split()[1])
            assert matches[rotation % COLS]
            assert any(matches[r % COLS] for r in ROTATIONS) == (abs(rotation) <= 15)

    baseline_macs = 3 * 31 * (12_800 + 6_400)
    spectral_macs = 4 * COLS * (CODE_ROWS + MASK_ROWS)
    inverse_terms = 2 * len(ROTATIONS) * COLS
    print(f"Verified {len(cases) * COLS} exact score pairs with three-party shares.")
    print(f"Current scan INT8 MACs/pair/party: {baseline_macs:,}")
    print(f"Spectral frequency INT8 MACs/pair/party: {spectral_macs:,}")
    print(f"Frequency-product reduction: {baseline_macs / spectral_macs:.2f}x")
    print(f"Additional selected-inverse field terms/pair/party: {inverse_terms:,}")
    total_macs = spectral_macs + 4 * inverse_terms
    print(f"Including four-limb selected inverse: {total_macs:,} INT8 MACs "
          f"({baseline_macs / total_macs:.2f}x reduction before padding/overheads)")
    print("Operation counts only; no GPU speedup or secure conversion benchmark.")
    c, m = 12800 - 2 * 4416, 12800
    print(f"Boundary: rational strict match={rational_match(c, m)}; "
          f"current GPU match={gpu_match(c, m)}")
    print(f"Zero overlap: rational match={rational_match(0, 0)}; "
          f"current GPU comparator={gpu_match(0, 0)}")


if __name__ == "__main__":
    main()
