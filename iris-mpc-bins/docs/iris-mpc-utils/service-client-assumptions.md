# Service Client: System Invariants & Assumptions

## Serial ID

- Each iris pair entry in the database has a unique `SerialId` (`u32`), assigned when a Uniqueness request is processed and the iris pair is enrolled.
- The SerialId is the primary key for referencing an enrolled identity across all subsequent operations.

## Request Types & Dependencies

| Request | Requires SerialId | Source of SerialId | Correlation Key |
|---------|-------------------|--------------------|-----------------|
| Uniqueness | No | Produces one on enrollment | `signup_id` |
| IdentityDeletion | Yes | Parent Uniqueness response | `serial_id` |
| Reauthorization | Yes | Parent Uniqueness response | `reauth_id` |
| ResetCheck | No | Independent | `reset_id` |
| ResetUpdate | Yes | Parent Uniqueness response | `reset_id` |

- Requests that require a SerialId (IdentityDeletion, Reauthorization, ResetUpdate) must reference a parent Uniqueness request whose response provides the SerialId.
- ResetCheck is independent and does not require a prior Uniqueness enrollment.

## Batch Ordering

- **Parent-before-child**: A child request's parent Uniqueness must be in a strictly earlier batch. Same-batch parent-child is not permitted.
- **Labels are global**: All labels across all batches must be unique.
- **Iris indexes are global**: No two requests across any batch may reference the same iris code index.

## Uniqueness Response Semantics

- `is_match=false`: The iris pair is new. The system enrolls it and returns the new `serial_id`.
- `is_match=true`: The iris pair matches an existing entry. `matched_serial_ids` contains the matching entries. No new enrollment occurs.
- The client resolves child request parents using `serial_id` (preferred) or the first entry in `matched_serial_ids`.

## Response Model

- Every enqueued request receives exactly one response from each of the 3 MPC nodes (`N_PARTIES=3`).
- A request is "fully correlated" when all 3 node responses have been received.
- **No timeout**: If any node fails to respond, the client blocks indefinitely in the response dequeue loop.
- Uncorrelated messages (e.g. from other clients sharing the queue) are left in the queue and not deleted.
- The response queue is purged at client initialization.

## Cross-Batch State Propagation

- After a batch completes, SerialIds from correlated Uniqueness requests are propagated to child requests in later batches via their `SignupId` parent descriptors.
- No other state is carried between batches.

## Error Handling

- If a response indicates an error (`error=true` or `success=false`), the client fails immediately via `response.validate()`.
- There is currently no support for expected-error / negative testing (e.g. deleting a non-existent SerialId and asserting the error).

## Shares Upload

- Iris code shares must be uploaded to S3 before the corresponding request is enqueued.
- Each of the 3 MPC nodes receives its own encrypted share.
- Shares are generated either from random computation (`FromCompute`) or from a pre-built NDJSON file (`FromFile`).

## Simple vs Complex Configs

- **Simple**: Generates `batch_count` identical batches, each containing `batch_size` request pairs. If the request kind requires a parent, each item is a Uniqueness+child pair interleaved in the same batch. Within-batch parent-child is handled by the exec loop's round-trip processing.
- **Complex**: A pre-defined set of batches with explicit labels and parent references. Subject to all validation rules above (no same-batch parents, unique labels, unique iris indexes, valid parent labels).
