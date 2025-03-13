use super::{
    rot::VecRots, BothEyes, HawkSession, HawkSessionRef, MapEdges, VecEdges, VecRequests, VectorId,
    LEFT, RIGHT,
};
use crate::{hawkers::aby3::aby3_store::QueryRef, hnsw::VectorStore};
use futures::future::JoinAll;
use iris_mpc_common::ROTATIONS;
use itertools::{izip, Itertools};
use std::{collections::HashMap, error::Error, iter, sync::Arc};

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
    // A request is to compare all rotations to a list of vectors.
    // A task is to compare one rotation to the vectors.
    let n_requests = vector_ids.len();
    let n_tasks = n_requests * ROTATIONS;
    let n_sessions = sessions.len();
    assert_eq!(queries.len(), n_requests);

    // Arc the requested vectors rather than cloning them.
    let vector_ids = vector_ids.into_iter().map(Arc::new);

    // For each request, broadcast the vectors to the rotations.
    // Concatenate the tasks for all requests, to maximize parallelism.
    let tasks = VecRots::flatten_broadcast(izip!(queries, vector_ids));
    assert_eq!(tasks.len(), n_tasks);

    // Prepare the tasks into one chunk per session.
    let chunks = split_tasks(tasks, n_sessions);

    // Process the chunks in parallel (CPU and IO).
    let results = izip!(chunks, sessions)
        .map(|(chunk, session)| per_session(chunk, session.clone()))
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await;
    assert_eq!(results.len(), n_sessions.min(n_tasks));

    // Undo the chunking per session.
    let results = unsplit_tasks(results);
    assert_eq!(results.len(), n_tasks);

    // Undo the flattening of rotations.
    let results = VecRots::unflatten(results);

    // Aggregate the results over rotations. ANY match.
    let results = results
        .into_iter()
        .map(aggregate_rotation_results)
        .collect_vec();

    assert_eq!(results.len(), n_requests);
    results
}

async fn per_session(
    tasks: VecRequests<(QueryRef, Arc<VecEdges<VectorId>>)>,
    session: HawkSessionRef,
) -> VecRequests<MapEdges<bool>> {
    let mut session = session.write().await;

    let mut out = Vec::with_capacity(tasks.len());
    for (query, vectors) in tasks {
        let matches = per_query(query, &vectors, &mut session).await;
        out.push(matches);
    }
    out
}

async fn per_query(
    query: QueryRef,
    vector_ids: &[VectorId],
    session: &mut HawkSession,
) -> MapEdges<bool> {
    let distances = session
        .aby3_store
        .eval_distance_batch(&[query], vector_ids)
        .await;

    let is_matches = session.aby3_store.is_match_batch(&distances).await;

    izip!(vector_ids, is_matches)
        .map(|(v, m)| (*v, m))
        .collect()
}

/// Split a Vec into at most `n_sessions` chunks.
/// The last chunk may be smaller.
/// Unlike `Itertools::chunks()` which borrows, this moves the items.
fn split_tasks<T>(v_iter: Vec<T>, n_sessions: usize) -> impl Iterator<Item = Vec<T>> {
    let chunk_size = v_iter.len().div_ceil(n_sessions);
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

fn unsplit_tasks<T, E: Error>(chunks: Vec<Result<Vec<T>, E>>) -> Vec<T> {
    chunks.into_iter().flat_map(Result::unwrap).collect_vec()
}

fn aggregate_rotation_results(results: VecRots<MapEdges<bool>>) -> MapEdges<bool> {
    results.iter().fold(HashMap::new(), |mut acc, m| {
        for (v, is_match) in m {
            *acc.entry(*v).or_default() |= is_match;
        }
        acc
    })
}
