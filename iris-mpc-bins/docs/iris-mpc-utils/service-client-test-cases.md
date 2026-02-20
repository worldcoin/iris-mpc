# Service Client Complex Test Cases

Test configurations for `iris-mpc-utils/assets/service-client/`. All use `SharesGeneratorOptions::FromFile` with an iris shares NDJSON file containing at least 128 entries.

## Conventions

- Batches are 0-indexed in this document; TOML arrays are sequential.
- Label format: `{sort_prefix}-{TestName}` (e.g. `00-U-Enroll`). Sort prefix controls display order.
- Iris pair indexes reference 1-based rows in the NDJSON file. Each pair is `[left, right]`.
- "Parent" always references a Uniqueness request label from a **strictly earlier** batch.
- Notation: **U** = Uniqueness, **D** = IdentityDeletion, **RA** = Reauthorisation, **RC** = ResetCheck, **RU** = ResetUpdate.

---

## TC-1: Heterogeneous 32-Request Batches (Happy Path)

**Goal:** Exercise all 5 request types across 3 batches, 32 requests each, with cross-batch parent resolution and intra-batch child dispatch.

### Batch 0 (32 Uniqueness enrollments)

All Uniqueness requests, each with a unique iris pair. These serve as parents for batches 1 and 2.

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1 | `00-U` | U | [1, 2] | -- |
| 2 | `01-U` | U | [3, 4] | -- |
| 3 | `02-U` | U | [5, 6] | -- |
| 4 | `03-U` | U | [7, 8] | -- |
| 5 | `04-U` | U | [9, 10] | -- |
| 6 | `05-U` | U | [11, 12] | -- |
| 7 | `06-U` | U | [13, 14] | -- |
| 8 | `07-U` | U | [15, 16] | -- |
| 9 | `08-U` | U | [17, 18] | -- |
| 10 | `09-U` | U | [19, 20] | -- |
| 11 | `10-U` | U | [21, 22] | -- |
| 12 | `11-U` | U | [23, 24] | -- |
| 13 | `12-U` | U | [25, 26] | -- |
| 14 | `13-U` | U | [27, 28] | -- |
| 15 | `14-U` | U | [29, 30] | -- |
| 16 | `15-U` | U | [31, 32] | -- |
| 17 | `16-U` | U | [33, 34] | -- |
| 18 | `17-U` | U | [35, 36] | -- |
| 19 | `18-U` | U | [37, 38] | -- |
| 20 | `19-U` | U | [39, 40] | -- |
| 21 | `20-U` | U | [41, 42] | -- |
| 22 | `21-U` | U | [43, 44] | -- |
| 23 | `22-U` | U | [45, 46] | -- |
| 24 | `23-U` | U | [47, 48] | -- |
| 25 | `24-U` | U | [49, 50] | -- |
| 26 | `25-U` | U | [51, 52] | -- |
| 27 | `26-U` | U | [53, 54] | -- |
| 28 | `27-U` | U | [55, 56] | -- |
| 29 | `28-U` | U | [57, 58] | -- |
| 30 | `29-U` | U | [59, 60] | -- |
| 31 | `30-U` | U | [61, 62] | -- |
| 32 | `31-U` | U | [63, 64] | -- |

### Batch 1 (32 mixed child + independent requests)

Heterogeneous mix: 8 D, 8 RA, 8 RC, 8 RU. Children reference batch 0 parents. RC has no parent.

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1 | `00-D` | D | -- | `00-U` |
| 2 | `01-D` | D | -- | `01-U` |
| 3 | `02-D` | D | -- | `02-U` |
| 4 | `03-D` | D | -- | `03-U` |
| 5 | `04-D` | D | -- | `04-U` |
| 6 | `05-D` | D | -- | `05-U` |
| 7 | `06-D` | D | -- | `06-U` |
| 8 | `07-D` | D | -- | `07-U` |
| 9 | `08-RA` | RA | [65, 66] | `08-U` |
| 10 | `09-RA` | RA | [67, 68] | `09-U` |
| 11 | `10-RA` | RA | [69, 70] | `10-U` |
| 12 | `11-RA` | RA | [71, 72] | `11-U` |
| 13 | `12-RA` | RA | [73, 74] | `12-U` |
| 14 | `13-RA` | RA | [75, 76] | `13-U` |
| 15 | `14-RA` | RA | [77, 78] | `14-U` |
| 16 | `15-RA` | RA | [79, 80] | `15-U` |
| 17 | `16-RC` | RC | [81, 82] | -- |
| 18 | `17-RC` | RC | [83, 84] | -- |
| 19 | `18-RC` | RC | [85, 86] | -- |
| 20 | `19-RC` | RC | [87, 88] | -- |
| 21 | `20-RC` | RC | [89, 90] | -- |
| 22 | `21-RC` | RC | [91, 92] | -- |
| 23 | `22-RC` | RC | [93, 94] | -- |
| 24 | `23-RC` | RC | [95, 96] | -- |
| 25 | `24-RU` | RU | [97, 98] | `16-U` |
| 26 | `25-RU` | RU | [99, 100] | `17-U` |
| 27 | `26-RU` | RU | [101, 102] | `18-U` |
| 28 | `27-RU` | RU | [103, 104] | `19-U` |
| 29 | `28-RU` | RU | [105, 106] | `20-U` |
| 30 | `29-RU` | RU | [107, 108] | `21-U` |
| 31 | `30-RU` | RU | [109, 110] | `22-U` |
| 32 | `31-RU` | RU | [111, 112] | `23-U` |

### Batch 2 (32 mixed: new enrollments + operations on batch 0 remainders)

Uses remaining batch 0 parents (`24-U` through `31-U`) plus new enrollments.

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1-16 | `00-U2` .. `15-U2` | U | [113,114] .. [143,144] | -- |
| 17-20 | `16-D2` .. `19-D2` | D | -- | `24-U` .. `27-U` |
| 21-24 | `20-RA2` .. `23-RA2` | RA | [145,146] .. [151,152] | `28-U` .. `31-U` |
| 25-28 | `24-RC2` .. `27-RC2` | RC | [153,154] .. [159,160] | -- |
| 29-32 | `28-RU2` .. `31-RU2` | RU | [161,162] .. [167,168] | `24-U` .. `27-U` |

**Validates:**
- All 5 request types processed correctly across multiple batches.
- Cross-batch parent resolution (`SignupId` -> `IrisSerialId`) for D, RA, RU.
- Intra-batch child dispatch after parent correlation.
- RC requests (parentless) enqueue immediately after shares upload.
- 32-request batches exercising parallel shares upload and SNS publish.

---

## TC-2: Mirroring Attack Detection

**Goal:** Submit uniqueness requests with swapped L/R iris pairs to exercise mirror attack detection. The system flag `full_face_mirror_attacks_detection_enabled` is set to `true` in the enqueuer.

### Batch 0 (32 requests: 16 normal + 16 mirrored)

| # | Label | Type | Iris Pair | Notes |
|---|-------|------|-----------|-------|
| 1 | `00-U-Normal` | U | [1, 2] | Original enrollment |
| 2 | `01-U-Mirror` | U | [2, 1] | **Swapped** L/R of entry 1 |
| 3 | `02-U-Normal` | U | [3, 4] | Original enrollment |
| 4 | `03-U-Mirror` | U | [4, 3] | **Swapped** L/R of entry 3 |
| ... | ... | ... | ... | Repeat pattern for pairs [5,6]..[31,32] |
| 31 | `30-U-Normal` | U | [31, 32] | Original enrollment |
| 32 | `31-U-Mirror` | U | [32, 31] | **Swapped** L/R of entry 31 |

**Validates:**
- Swapped iris pairs pass TOML validation (`validate_iris_pairs` normalizes pairs before comparison, so `[1,2]` and `[2,1]` are treated as the same pair).
- System-level mirror attack detection flags the swapped enrollments.
- Correlation still works for all 32 responses (both normal and mirrored).
- The `matched_serial_ids` field in `UniquenessResult` reflects matches between normal and mirrored entries.

---

## TC-3: Duplicate Enrollment Stress Test

**Goal:** Submit identical iris pairs across multiple requests to test duplicate detection and the system's handling of repeat enrollments.

### Batch 0 (32 requests: 4 unique pairs, each enrolled 8 times)

| # | Label | Type | Iris Pair | Notes |
|---|-------|------|-----------|-------|
| 1-8 | `00-U-Dup` .. `07-U-Dup` | U | [1, 2] | Same pair, 8 enrollments |
| 9-16 | `08-U-Dup` .. `15-U-Dup` | U | [3, 4] | Same pair, 8 enrollments |
| 17-24 | `16-U-Dup` .. `23-U-Dup` | U | [5, 6] | Same pair, 8 enrollments |
| 25-32 | `24-U-Dup` .. `31-U-Dup` | U | [7, 8] | Same pair, 8 enrollments |

### Batch 1 (32 requests: deletions of all batch 0 enrollments)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1-8 | `00-D-Dup` .. `07-D-Dup` | D | -- | `00-U-Dup` .. `07-U-Dup` |
| 9-16 | `08-D-Dup` .. `15-D-Dup` | D | -- | `08-U-Dup` .. `15-U-Dup` |
| 17-24 | `16-D-Dup` .. `23-D-Dup` | D | -- | `16-U-Dup` .. `23-U-Dup` |
| 25-32 | `24-D-Dup` .. `31-D-Dup` | D | -- | `24-U-Dup` .. `31-U-Dup` |

**Validates:**
- Duplicate iris pairs pass TOML validation (allowed by `validate_iris_pairs`).
- System correctly handles multiple enrollments of the same biometric.
- Each uniqueness response returns a `serial_id` or `matched_serial_ids`; subsequent duplicates should reflect matches.
- Each deletion resolves to the correct `IrisSerialId` via cross-batch parent resolution.
- Response correlation correctly maps each of the 8 duplicate deletion responses to the right request (by `serial_id`).

---

## TC-4: Mirroring + Duplicate Combined Attack

**Goal:** Combined mirror and duplicate attack vectors in a single batch to test interaction between detection mechanisms.

### Batch 0 (32 requests)

| # | Label | Type | Iris Pair | Notes |
|---|-------|------|-----------|-------|
| 1 | `00-U-Orig` | U | [1, 2] | First enrollment |
| 2 | `01-U-Dup` | U | [1, 2] | Exact duplicate |
| 3 | `02-U-Mirror` | U | [2, 1] | Mirror of [1,2] |
| 4 | `03-U-MirrorDup` | U | [2, 1] | Duplicate mirror |
| 5 | `04-U-Orig` | U | [3, 4] | First enrollment |
| 6 | `05-U-Dup` | U | [3, 4] | Exact duplicate |
| 7 | `06-U-Mirror` | U | [4, 3] | Mirror of [3,4] |
| 8 | `07-U-MirrorDup` | U | [4, 3] | Duplicate mirror |
| ... | ... | ... | ... | Repeat pattern for pairs [5,6]..[15,16] |
| 29-32 | | U | [15,16] / [16,15] | Last group |

### Batch 1 (32 requests: lifecycle ops on batch 0)

Mix of D, RA, RC, RU referencing batch 0 parents:

| # | Label | Type | Iris Pair | Parent | Notes |
|---|-------|------|-----------|--------|-------|
| 1-8 | `00-D` .. `07-D` | D | -- | `00-U-Orig` .. `07-U-MirrorDup` | Delete both originals and mirrors |
| 9-16 | `08-RA` .. `15-RA` | RA | [17,18]..[24,25] | `04-U-Orig` .. various | Reauth against dup+mirror parents |
| 17-24 | `16-RC` .. `23-RC` | RC | [25,26]..[32,33] | -- | Independent reset checks |
| 25-32 | `24-RU` .. `31-RU` | RU | [33,34]..[40,41] | Various batch 0 parents | Reset updates against dup/mirror parents |

**Validates:**
- Interaction between duplicate detection and mirror attack detection.
- Child operations (D, RA, RU) correctly resolve parents when the parent was a duplicate or mirror.
- System returns appropriate `matched_serial_ids` for both duplicate and mirror scenarios.
- All 64 requests across 2 batches produce correct correlation.

---

## TC-5: Many Children, One Parent

**Goal:** Stress the intra-batch child dispatch and cross-batch resolution by having many child requests all depend on a single parent.

### Batch 0 (1 Uniqueness enrollment)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1 | `00-U-SingleParent` | U | [1, 2] | -- |

### Batch 1 (32 requests: all children of the single parent)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1-11 | `00-D` .. `10-D` | D | -- | `00-U-SingleParent` |
| 12-21 | `11-RA` .. `20-RA` | RA | [3,4]..[21,22] | `00-U-SingleParent` |
| 22-32 | `21-RU` .. `31-RU` | RU | [23,24]..[43,44] | `00-U-SingleParent` |

**Validates:**
- All 32 children resolve the same parent's `IrisSerialId` via `resolve_cross_batch_parents`.
- 11 IdentityDeletion requests all use the same `serial_id` (the system should handle multiple deletions of the same identity).
- 10 Reauthorisation and 11 ResetUpdate requests all reference the same parent.
- Enqueue order: all 32 become enqueueable simultaneously after cross-batch resolution.

---

## TC-6: Deep Dependency Chain (Multi-Hop)

**Goal:** Test sequential batch processing where each batch's children become the parents for the next batch's operations. Exercises the `cross_batch_resolutions` HashMap accumulating over many batches.

### Batch 0 (32 Uniqueness)

| # | Label | Type | Iris Pair |
|---|-------|------|-----------|
| 1-32 | `00-U-Gen0` .. `31-U-Gen0` | U | [1,2]..[63,64] |

### Batch 1 (32 requests: delete batch 0, re-enroll new)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1-16 | `00-D-Gen0` .. `15-D-Gen0` | D | -- | `00-U-Gen0` .. `15-U-Gen0` |
| 17-32 | `16-U-Gen1` .. `31-U-Gen1` | U | [65,66]..[95,96] | -- |

### Batch 2 (32 requests: ops on Gen1 + new enrollments)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1-8 | `00-RA-Gen1` .. `07-RA-Gen1` | RA | [97,98]..[111,112] | `16-U-Gen1` .. `23-U-Gen1` |
| 9-16 | `08-RU-Gen1` .. `15-RU-Gen1` | RU | [113,114]..[127,128] | `24-U-Gen1` .. `31-U-Gen1` |
| 17-32 | `16-U-Gen2` .. `31-U-Gen2` | U | [129,130]..[159,160] | -- |

### Batch 3 (32 requests: ops on Gen0 remainders + Gen2)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1-16 | `00-D-Gen0b` .. `15-D-Gen0b` | D | -- | `16-U-Gen0` .. `31-U-Gen0` |
| 17-32 | `16-RU-Gen2` .. `31-RU-Gen2` | RU | [161,162]..[191,192] | `16-U-Gen2` .. `31-U-Gen2` |

**Validates:**
- `cross_batch_resolutions` HashMap correctly accumulates across 4 batches.
- Later batches reference parents from non-adjacent earlier batches (batch 3 references batch 0).
- Mix of deletion + new enrollment in the same batch.
- No stale resolution data: each `SignupId -> IrisSerialId` mapping persists correctly.

---

## TC-7: Orphaned Children (No Parent Signup)
out of scope. not going to do it. the system was designed to prevent this.

---

## TC-8: Intra-Batch Parent-Child (Same-Batch Dependency)

not relevant, was for an old version of service client

---

## TC-9: Maximum Batch Stress Test

**Goal:** Push the limits with maximum-size batches to stress parallel shares upload, SNS publish, and SQS correlation.

### Batch 0 (32 Uniqueness)

Standard 32 uniqueness enrollments with pairs [1,2]..[63,64].

### Batch 1 (32 mixed, all child types, referencing all 32 parents)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1 | `00-D` | D | -- | `00-U` |
| 2 | `01-RA` | RA | [65, 66] | `01-U` |
| 3 | `02-RC` | RC | [67, 68] | -- |
| 4 | `03-RU` | RU | [69, 70] | `03-U` |
| 5 | `04-D` | D | -- | `04-U` |
| 6 | `05-RA` | RA | [71, 72] | `05-U` |
| 7 | `06-RC` | RC | [73, 74] | -- |
| 8 | `07-RU` | RU | [75, 76] | `07-U` |
| ... | ... | ... | ... | Repeating D/RA/RC/RU pattern for all 32 |

### Batch 2 (32 Uniqueness: re-enrollment after deletions)

Re-enroll using the **same iris pairs** as the entries deleted in batch 1 (pairs from `00-U`, `04-U`, `08-U`, etc. that were deleted).

| # | Label | Type | Iris Pair | Notes |
|---|-------|------|-----------|-------|
| 1-8 | `00-U-ReEnroll` .. `07-U-ReEnroll` | U | [1,2],[9,10],[17,18]... | Same pairs as deleted entries |
| 9-32 | `08-U-Fresh` .. `31-U-Fresh` | U | [77,78]..[123,124] | New enrollments |

**Validates:**
- Full lifecycle: enroll -> delete -> re-enroll with same biometric data.
- Re-enrollment after deletion produces new `serial_id` (not the old one).
- Interleaved request types in a single batch all correlate correctly.
- 3 full 32-request batches processed sequentially.

---

## TC-10: Reauth and Reset Full Lifecycle

**Goal:** Exercise the complete reauth and reset flows with multiple rounds, ensuring `reauth_id` and `reset_id` UUID uniqueness across requests.

### Batch 0 (8 Uniqueness)

| # | Label | Type | Iris Pair |
|---|-------|------|-----------|
| 1-8 | `00-U` .. `07-U` | U | [1,2]..[15,16] |

### Batch 1 (32 requests: multi-round reauth + reset check + reset update)

| # | Label | Type | Iris Pair | Parent | Notes |
|---|-------|------|-----------|--------|-------|
| 1-4 | `00-RA-R1` .. `03-RA-R1` | RA | [17,18]..[23,24] | `00-U` .. `03-U` | Round 1 reauth |
| 5-8 | `04-RA-R2` .. `07-RA-R2` | RA | [25,26]..[31,32] | `00-U` .. `03-U` | Round 2 reauth (same parents) |
| 9-16 | `08-RC` .. `15-RC` | RC | [33,34]..[47,48] | -- | Reset checks |
| 17-24 | `16-RU-R1` .. `23-RU-R1` | RU | [49,50]..[63,64] | `00-U` .. `07-U` | Round 1 reset update |
| 25-32 | `24-RU-R2` .. `31-RU-R2` | RU | [65,66]..[79,80] | `00-U` .. `07-U` | Round 2 reset update (same parents) |

**Validates:**
- Multiple reauth requests against the same parent (each gets a unique `reauth_id` UUID).
- Multiple reset updates against the same parent (each gets a unique `reset_id` UUID).
- Correlation correctly distinguishes between rounds via UUID matching.
- 8 reset checks independent of any parent.
- Total: 40 requests across 2 batches.

---

## TC-11: Empty and Single-Request Batches

**Goal:** Edge case testing with minimal batch sizes interspersed with full batches.

### Batch 0 (1 Uniqueness)

| # | Label | Type | Iris Pair |
|---|-------|------|-----------|
| 1 | `00-U-Solo` | U | [1, 2] |

### Batch 1 (32 requests: all reference the single parent)

| # | Label | Type | Iris Pair | Parent |
|---|-------|------|-----------|--------|
| 1 | `00-D` | D | -- | `00-U-Solo` |
| 2-32 | `01-RC` .. `31-RC` | RC | [3,4]..[63,64] | -- |

### Batch 2 (1 Uniqueness: new enrollment)

| # | Label | Type | Iris Pair |
|---|-------|------|-----------|
| 1 | `00-U-Solo2` | U | [65, 66] |

### Batch 3 (1 IdentityDeletion)

| # | Label | Type | Parent |
|---|-------|------|--------|
| 1 | `00-D-Solo` | D | `00-U-Solo2` |

**Validates:**
- Single-request batches process correctly.
- Transition between small and large batches.
- Cross-batch resolution spanning 4 batches with variable sizes.

---

## TC-12: Duplicate Request After Deletion (Re-enrollment Race)

**Goal:** Enroll, delete, then immediately re-enroll the same iris pair to test the system's handling of the tombstone/reuse scenario.

### Batch 0 (16 Uniqueness)

| # | Label | Type | Iris Pair |
|---|-------|------|-----------|
| 1-16 | `00-U` .. `15-U` | U | [1,2]..[31,32] |

### Batch 1 (16 Deletions)

| # | Label | Type | Parent |
|---|-------|------|--------|
| 1-16 | `00-D` .. `15-D` | D | `00-U` .. `15-U` |

### Batch 2 (32 re-enrollments: same iris pairs twice each)

| # | Label | Type | Iris Pair | Notes |
|---|-------|------|-----------|-------|
| 1-16 | `00-U-Re1` .. `15-U-Re1` | U | [1,2]..[31,32] | First re-enrollment |
| 17-32 | `16-U-Re2` .. `31-U-Re2` | U | [1,2]..[31,32] | Second re-enrollment (duplicate) |

**Validates:**
- Same iris pair re-enrolled after deletion.
- Double re-enrollment of the same pair in the same batch.
- System assigns fresh `serial_id` values for re-enrollments.
- `matched_serial_ids` correctly reflects the duplicate in the second set.

---

