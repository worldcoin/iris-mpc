use super::{
    rot::VecRots, BothEyes, HawkSession, HawkSessionRef, MapEdges, VecEdges, VecRequests, VectorId,
    LEFT, RIGHT,
};
use crate::{hawkers::aby3_store::QueryRef, hnsw::VectorStore};
use futures::future::JoinAll;
use itertools::{izip, Itertools};
use std::iter;

pub async fn calculate_is_match(
    queries: &BothEyes<VecRequests<VecRots<QueryRef>>>,
    vector_ids: BothEyes<VecRequests<VecEdges<VectorId>>>,
    sessions: &BothEyes<Vec<HawkSessionRef>>,
) -> BothEyes<VecRequests<MapEdges<bool>>> {
    let [vectors_left, vectors_right] = vector_ids;

    // Parallelize left and right sessions (IO only).
    let (out_l, out_r) = futures::join!(
        per_side(&queries[LEFT], vectors_left, &sessions[LEFT]),
        per_side(&queries[RIGHT], vectors_right, &sessions[RIGHT]),
    );
    [out_l, out_r]
}

async fn per_side(
    queries: &VecRequests<VecRots<QueryRef>>,
    vector_ids: VecRequests<VecEdges<VectorId>>,
    sessions: &Vec<HawkSessionRef>,
) -> VecRequests<MapEdges<bool>> {
    let n_requests = vector_ids.len();
    let n_sessions = sessions.len();
    let n_per_session = n_requests.div_ceil(n_sessions);
    assert_eq!(queries.len(), n_requests);

    // Organize as: request -> (query and its related vectors).
    let requests = izip!(queries, vector_ids).map(|(q, vector_ids)| {
        // TODO: All rotations.
        (q.center().clone(), vector_ids)
    });

    // Prepare the requests into one chunk per session.
    let chunks = chunk_vecs(requests, n_per_session);

    // Process the chunks in parallel (CPU and IO).
    let results = izip!(chunks, sessions)
        .map(|(chunk, session)| per_session(chunk, session.clone()))
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await;
    assert!(results.len() == n_sessions.min(n_requests));

    let r = results.into_iter().flat_map(Result::unwrap).collect_vec();
    assert_eq!(r.len(), n_requests);
    r
}

async fn per_session(
    requests: VecRequests<(QueryRef, VecEdges<VectorId>)>,
    session: HawkSessionRef,
) -> VecRequests<MapEdges<bool>> {
    let mut session = session.write().await;

    let mut out = Vec::with_capacity(requests.len());
    for (query, vectors) in requests {
        let matches = per_query(query, vectors, &mut session).await;
        out.push(matches);
    }
    out
}

async fn per_query(
    query: QueryRef,
    vector_ids: Vec<VectorId>,
    session: &mut HawkSession,
) -> MapEdges<bool> {
    let distances = session
        .aby3_store
        .eval_distance_batch(&query, &*vector_ids)
        .await;

    let is_matches = session.aby3_store.is_match_batch(&distances).await;

    izip!(vector_ids, is_matches).collect()
}

/// Chunk an iterator into vectors of `chunk_size` each.
/// The last chunk may be smaller.
/// Unlike `Itertools::chunks()` which borrows, this moves the items.
fn chunk_vecs<T>(
    v_iter: impl IntoIterator<Item = T>,
    chunk_size: usize,
) -> impl Iterator<Item = Vec<T>> {
    let mut v_iter = v_iter.into_iter();
    iter::from_fn(move || {
        let chunk = v_iter.by_ref().take(chunk_size).collect_vec();
        if chunk.is_empty() {
            None // Stop.
        } else {
            Some(chunk)
        }
    })
}
