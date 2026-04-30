//! Wire protocol for the iris-mpc remote worker pool.
//!
//! Defines the request/response types exchanged between Hawk Main (the
//! `LeaderHandle` side of the ampc-actor-utils workpool) and remote worker
//! processes that own iris store shards. Each `WorkerRequest` variant maps
//! 1:1 to a method on the `IrisWorkerPool` trait
//! (`iris-mpc-cpu::execution::hawk_main::iris_worker`); the worker side is
//! expected to implement those semantics faithfully — see the per-variant
//! doc comments and the trait docstring for the rules that aren't visible
//! from the wire types alone.
//!
//! ## Encoding
//!
//! v1 uses bincode with a single leading version byte. Encode helpers
//! produce `Vec<u8>` suitable for wrapping in a workpool `Payload::Bytes`.
//! Decode helpers reject mismatched versions up front — this is intentional;
//! we deploy the leader and workers from the same release.
//!
//! ## Sharding
//!
//! The wire protocol is shard-agnostic. `VectorId`-routing is the leader's
//! responsibility: scatter-gather requests are partitioned by
//! `hash(VectorId) % num_shards` before being put on the wire. The worker
//! sees only the slice intended for it and is expected to error (not silently
//! no-op) if it receives a `VectorId` outside its shard.
//!
//! ## Idempotency
//!
//! Most methods are idempotent under retry (`cache_queries` no-ops on
//! known QueryIds, `evict_queries` silently skips unknown ones,
//! `delete_irises` writes the same dummy sentinel, reads are pure).
//! `insert_irises` is idempotent at the store level (`(VectorId, iris)`
//! re-insert is a no-op for `set_hash`), but the leader must not call
//! `evict_queries` on a `QueryId` until all `insert_irises` referencing
//! it have been ack'd.

use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    vector_id::VectorId,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Wire-protocol version. Bump for any breaking change to the bincode
/// shape of `WorkerRequest` / `WorkerResponse`.
pub const PROTOCOL_VERSION: u8 = 1;

/// Opaque identifier for a cached query. Allocated by the leader, opaque
/// to the worker beyond cache lookup. Mirrors `iris_worker::QueryId`.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueryId(pub u64);

/// Selects a specific preprocessed variant of a cached query.
///
/// Each cached iris produces 31 rotations × 2 orientations (normal +
/// mirrored) — `QuerySpec` picks one of those 62 variants for distance
/// computation. Rotation index 15 (`CENTER_ROTATION`) is the identity.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct QuerySpec {
    pub query_id: QueryId,
    /// Rotation index in `[0, 31)`. `CENTER_ROTATION = 15` is identity.
    pub rotation: u8,
    /// If true, the worker uses the mirrored-then-preprocessed variant.
    pub mirrored: bool,
}

/// Center (identity) rotation index.
pub const CENTER_ROTATION: u8 = 15;

/// An iris share: code + mask. Wire-equivalent to
/// `iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris` (same fields
/// in the same order, both serde-derived). A future refactor that moves
/// `GaloisRingSharedIris` into a shared crate would let us reuse it
/// directly; until then this mirror keeps the wire crate independent of
/// `iris-mpc-cpu`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IrisShare {
    pub code: GaloisRingIrisCodeShare,
    pub mask: GaloisRingTrimmedMaskCodeShare,
}

/// Request from the leader to a worker.
///
/// Each variant maps to an `IrisWorkerPool` trait method. Field shapes
/// match the trait signatures; see the trait for full semantics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkerRequest {
    /// `cache_queries`. Worker preprocesses + rotates each iris into 62
    /// variants and stores them under the given `QueryId`. Re-caching an
    /// already-cached id is a no-op.
    CacheQueries { queries: Vec<(QueryId, IrisShare)> },

    /// `compute_dot_products`. For each batch, computes the dot product
    /// of the cached query's selected rotation against each `VectorId`
    /// in the worker's iris store. Returns one inner `Vec<u16>` per
    /// batch.
    ComputeDotProducts {
        batches: Vec<(QuerySpec, Vec<VectorId>)>,
    },

    /// `fetch_irises`. Returns one `IrisShare` per requested id, in
    /// input order. Missing entries return the worker's default empty
    /// iris (which yields max-distance under dot products).
    FetchIrises { ids: Vec<VectorId> },

    /// `insert_irises`. Resolves each `QueryId` to its cached original
    /// iris and inserts at the given `VectorId`. Returns the
    /// **per-shard** `set_hash` checksum after all inserts are applied;
    /// the leader combines across shards.
    InsertIrises { inserts: Vec<(QueryId, VectorId)> },

    /// `compute_pairwise_distances`. Each pair: first operand is a
    /// preprocessed rotation (`QuerySpec`), second is the **raw**
    /// (unpreprocessed) iris of a cached query (`QueryId`). `None`
    /// pairs produce a max-distance sentinel.
    ComputePairwiseDistances {
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    },

    /// `evict_queries`. Frees cached query data. Unknown ids are
    /// silently skipped.
    EvictQueries { query_ids: Vec<QueryId> },

    /// `delete_irises`. Replaces each id with the party's dummy
    /// sentinel iris (max-distance under dot products).
    DeleteIrises { ids: Vec<VectorId> },
}

/// Response from a worker to the leader.
///
/// First seven variants are 1:1 with `WorkerRequest`. Per-method failures
/// are conveyed as `Err(message)` inside the variant.
///
/// `ProtocolError` is the variant-agnostic escape hatch: returned when the
/// worker can't even decode the request (version mismatch, truncated bytes)
/// and therefore doesn't know which response shape the leader expected. The
/// leader treats it as a hard failure regardless of the original method.
///
/// Transport-level failures (worker crash, message lost in transit) are
/// surfaced by the workpool layer as `WorkpoolError::JobsLost`, not by
/// this enum.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkerResponse {
    CacheQueries(Result<(), String>),
    ComputeDotProducts(Result<Vec<Vec<u16>>, String>),
    FetchIrises(Result<Vec<IrisShare>, String>),
    InsertIrises(Result<u64, String>),
    ComputePairwiseDistances(Result<Vec<u16>, String>),
    EvictQueries(Result<(), String>),
    DeleteIrises(Result<(), String>),
    ProtocolError(String),
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("empty payload")]
    Empty,
    #[error("unsupported protocol version: got {got}, expected {expected}")]
    VersionMismatch { got: u8, expected: u8 },
    #[error("bincode: {0}")]
    Bincode(#[from] bincode::Error),
}

/// Encode a `WorkerRequest` to a byte vector with the leading version byte.
pub fn encode_request(req: &WorkerRequest) -> Result<Vec<u8>, CodecError> {
    encode_with_version(req)
}

/// Decode a `WorkerRequest`, validating the leading version byte.
pub fn decode_request(bytes: &[u8]) -> Result<WorkerRequest, CodecError> {
    decode_with_version(bytes)
}

/// Encode a `WorkerResponse` to a byte vector with the leading version byte.
pub fn encode_response(rsp: &WorkerResponse) -> Result<Vec<u8>, CodecError> {
    encode_with_version(rsp)
}

/// Decode a `WorkerResponse`, validating the leading version byte.
pub fn decode_response(bytes: &[u8]) -> Result<WorkerResponse, CodecError> {
    decode_with_version(bytes)
}

fn encode_with_version<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let body = bincode::serialize(value)?;
    let mut out = Vec::with_capacity(1 + body.len());
    out.push(PROTOCOL_VERSION);
    out.extend_from_slice(&body);
    Ok(out)
}

fn decode_with_version<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, CodecError> {
    let (&version, body) = bytes.split_first().ok_or(CodecError::Empty)?;
    if version != PROTOCOL_VERSION {
        return Err(CodecError::VersionMismatch {
            got: version,
            expected: PROTOCOL_VERSION,
        });
    }
    Ok(bincode::deserialize(body)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris_mpc_common::galois_engine::degree4::{
        GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
    };

    fn sample_iris() -> IrisShare {
        IrisShare {
            code: GaloisRingIrisCodeShare::default_for_party(0),
            mask: GaloisRingTrimmedMaskCodeShare::default_for_party(0),
        }
    }

    fn sample_vid(serial: u32) -> VectorId {
        VectorId::from_serial_id(serial)
    }

    #[test]
    fn cache_queries_roundtrip() {
        let req = WorkerRequest::CacheQueries {
            queries: vec![(QueryId(1), sample_iris()), (QueryId(2), sample_iris())],
        };
        let bytes = encode_request(&req).unwrap();
        let decoded = decode_request(&bytes).unwrap();
        match decoded {
            WorkerRequest::CacheQueries { queries } => {
                assert_eq!(queries.len(), 2);
                assert_eq!(queries[0].0, QueryId(1));
                assert_eq!(queries[1].0, QueryId(2));
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }

    #[test]
    fn compute_dot_products_roundtrip() {
        let req = WorkerRequest::ComputeDotProducts {
            batches: vec![(
                QuerySpec {
                    query_id: QueryId(7),
                    rotation: CENTER_ROTATION,
                    mirrored: false,
                },
                vec![sample_vid(0), sample_vid(1), sample_vid(2)],
            )],
        };
        let bytes = encode_request(&req).unwrap();
        let decoded = decode_request(&bytes).unwrap();
        match decoded {
            WorkerRequest::ComputeDotProducts { batches } => {
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].0.query_id, QueryId(7));
                assert_eq!(batches[0].1.len(), 3);
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }

    #[test]
    fn pairwise_with_none_pairs_roundtrip() {
        let req = WorkerRequest::ComputePairwiseDistances {
            pairs: vec![
                Some((
                    QuerySpec {
                        query_id: QueryId(1),
                        rotation: CENTER_ROTATION,
                        mirrored: true,
                    },
                    QueryId(2),
                )),
                None,
            ],
        };
        let bytes = encode_request(&req).unwrap();
        let decoded = decode_request(&bytes).unwrap();
        match decoded {
            WorkerRequest::ComputePairwiseDistances { pairs } => {
                assert_eq!(pairs.len(), 2);
                assert!(pairs[0].is_some());
                assert!(pairs[1].is_none());
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }

    #[test]
    fn response_ok_and_err_roundtrip() {
        let ok = WorkerResponse::InsertIrises(Ok(0xDEAD_BEEF));
        let bytes = encode_response(&ok).unwrap();
        match decode_response(&bytes).unwrap() {
            WorkerResponse::InsertIrises(Ok(v)) => assert_eq!(v, 0xDEAD_BEEF),
            other => panic!("wrong variant: {:?}", other),
        }

        let err = WorkerResponse::ComputeDotProducts(Err("missing query".into()));
        let bytes = encode_response(&err).unwrap();
        match decode_response(&bytes).unwrap() {
            WorkerResponse::ComputeDotProducts(Err(msg)) => assert_eq!(msg, "missing query"),
            other => panic!("wrong variant: {:?}", other),
        }

        let proto = WorkerResponse::ProtocolError("bad bytes".into());
        let bytes = encode_response(&proto).unwrap();
        match decode_response(&bytes).unwrap() {
            WorkerResponse::ProtocolError(msg) => assert_eq!(msg, "bad bytes"),
            other => panic!("wrong variant: {:?}", other),
        }
    }

    #[test]
    fn version_mismatch_rejected() {
        let req = WorkerRequest::EvictQueries {
            query_ids: vec![QueryId(1)],
        };
        let mut bytes = encode_request(&req).unwrap();
        bytes[0] = PROTOCOL_VERSION.wrapping_add(1);
        match decode_request(&bytes) {
            Err(CodecError::VersionMismatch { got, expected }) => {
                assert_eq!(got, PROTOCOL_VERSION.wrapping_add(1));
                assert_eq!(expected, PROTOCOL_VERSION);
            }
            other => panic!("expected VersionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn empty_payload_rejected() {
        match decode_request(&[]) {
            Err(CodecError::Empty) => {}
            other => panic!("expected Empty, got {:?}", other),
        }
    }
}
