# Request-Level Test Cases

Test configurations for `iris-mpc-utils/assets/service-client/rq_tc_*.toml`. All use `SharesGeneratorOptions::FromFile` with an iris shares NDJSON file containing at least 4 entries.

## Conventions

- Batches are 0-indexed in this document; TOML arrays are sequential.
- Label format: `{sort_prefix}-{Type}-{Description}` (e.g. `00-U-Enroll`).
- Iris pair indexes reference 1-based rows in the NDJSON file. Each pair is `[left, right]`.
- "Parent" always references a Uniqueness request label from a **strictly earlier** batch.
- Notation: **U** = Uniqueness, **D** = IdentityDeletion, **RA** = Reauthorisation, **RC** = ResetCheck, **RU** = ResetUpdate.

---

## rq_tc_uniqueness: Enrollment Lifecycle

**Goal:** Verify the full uniqueness enrollment lifecycle: first enrollment (no match), duplicate detection, deletion, double-deletion failure, and re-enrollment after deletion.

### Batch 0 (1 Uniqueness enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-Enroll` | U | [1, 2] | `is_match=false` (first enrollment) |

### Batch 1 (1 Uniqueness — duplicate)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-EnrollDup` | U | [1, 2] | `is_match=true` (duplicate of batch 0) |

### Batch 2 (1 IdentityDeletion)

| # | Label | Type | Parent | Expected |
|---|-------|------|--------|----------|
| 1 | `00-D-Delete` | D | `00-U-Enroll` | `success=true` |

### Batch 3 (1 IdentityDeletion — repeat)

| # | Label | Type | Parent | Expected |
|---|-------|------|--------|----------|
| 1 | `00-D-DeleteAgain` | D | `00-U-Enroll` | failure (already deleted) |

### Batch 4 (1 Uniqueness — re-enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-ReEnroll` | U | [1, 2] | `is_match=false` (identity was deleted) |

**Validates:**
- First enrollment returns no match.
- Duplicate enrollment of the same iris pair returns a match.
- Deletion succeeds for an existing identity.
- Repeated deletion of the same identity fails.
- Re-enrollment after deletion returns no match (fresh serial_id).

---

## rq_tc_reset_update: Reset Update Lifecycle

**Goal:** Verify that reset_update replaces an identity's iris code, so the original iris code no longer matches.

### Batch 0 (1 Uniqueness enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-Enroll` | U | [1, 2] | `is_match=false` (first enrollment) |

### Batch 1 (1 Uniqueness — duplicate)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-EnrollDup` | U | [1, 2] | `is_match=true` (duplicate of batch 0) |

### Batch 2 (1 ResetUpdate — replace iris code)

| # | Label | Type | Iris Pair | Parent | Expected |
|---|-------|------|-----------|--------|----------|
| 1 | `00-RU-Reset` | RU | [3, 4] | `00-U-Enroll` | success (replaces iris code) |

### Batch 3 (1 Uniqueness — original iris code)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-ReEnroll` | U | [1, 2] | `is_match=false` (original iris was replaced) |

**Validates:**
- First enrollment returns no match.
- Duplicate enrollment confirms the iris code exists.
- ResetUpdate replaces the identity's iris code with a different one.
- Enrolling the original iris code after reset returns no match (it was replaced by [3, 4]).

---

## rq_tc_reauth: Reauthorisation Lifecycle

**Goal:** Verify that reauth fails against a deleted identity and succeeds against a live one.

### Batch 0 (1 Uniqueness enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-Enroll` | U | [1, 2] | enrollment succeeds |

### Batch 1 (1 IdentityDeletion)

| # | Label | Type | Parent | Expected |
|---|-------|------|--------|----------|
| 1 | `00-D-Delete` | D | `00-U-Enroll` | `success=true` |

### Batch 2 (1 Reauthorisation — against deleted identity)

| # | Label | Type | Iris Pair | Parent | Expected |
|---|-------|------|-----------|--------|----------|
| 1 | `00-RA-Fail` | RA | [1, 2] | `00-U-Enroll` | failure (identity was deleted) |

### Batch 3 (1 Uniqueness — re-enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-ReEnroll` | U | [1, 2] | enrollment succeeds |

### Batch 4 (1 Reauthorisation — against live identity)

| # | Label | Type | Iris Pair | Parent | Expected |
|---|-------|------|-----------|--------|----------|
| 1 | `00-RA-Succeed` | RA | [1, 2] | `00-U-ReEnroll` | success |

**Validates:**
- Reauth against a deleted identity fails.
- Reauth against a live (re-enrolled) identity succeeds.
- Cross-batch parent resolution works across 5 batches.

---

## rq_tc_reset_check: Reset Check Lifecycle

**Goal:** Same flow as reauth — verify that reset_check fails against a deleted identity and succeeds against a live one.

### Batch 0 (1 Uniqueness enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-Enroll` | U | [1, 2] | enrollment succeeds |

### Batch 1 (1 IdentityDeletion)

| # | Label | Type | Parent | Expected |
|---|-------|------|--------|----------|
| 1 | `00-D-Delete` | D | `00-U-Enroll` | `success=true` |

### Batch 2 (1 ResetCheck — against deleted identity)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-RC-Fail` | RC | [1, 2] | failure (identity was deleted) |

### Batch 3 (1 Uniqueness — re-enrollment)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-U-ReEnroll` | U | [1, 2] | enrollment succeeds |

### Batch 4 (1 ResetCheck — against live identity)

| # | Label | Type | Iris Pair | Expected |
|---|-------|------|-----------|----------|
| 1 | `00-RC-Succeed` | RC | [1, 2] | success |

**Validates:**
- ResetCheck against a deleted identity fails.
- ResetCheck against a live (re-enrolled) identity succeeds.
- RC requests are parentless — they enqueue immediately after shares upload.

---
