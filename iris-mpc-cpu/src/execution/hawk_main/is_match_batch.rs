use super::{
    rot::VecRots, BothEyes, HawkSession, HawkSessionRef, MapEdges, VecEdges, VecRequests, VectorId,
    LEFT, RIGHT,
};
use crate::{hawkers::aby3::aby3_store::QueryRef, hnsw::VectorStore};
use futures::future::JoinAll;
use iris_mpc_common::ROTATIONS;
use itertools::{izip, Itertools};
use std::{collections::HashMap, sync::Arc};
use tokio::task::JoinError;

pub async fn calculate_missing_is_match(
    search_queries: &BothEyes<VecRequests<VecRots<QueryRef>>>,
    missing_vector_ids: BothEyes<VecRequests<VecEdges<VectorId>>>,
    sessions: &BothEyes<Vec<HawkSessionRef>>,
) -> eyre::Result<BothEyes<VecRequests<MapEdges<bool>>>> {
    let [missing_vectors_left, missing_vectors_right] = missing_vector_ids;

    // Parallelize left and right sessions (IO only).
    let (out_l, out_r) = futures::join!(
        per_side(&search_queries[LEFT], missing_vectors_left, &sessions[LEFT]),
        per_side(
            &search_queries[RIGHT],
            missing_vectors_right,
            &sessions[RIGHT]
        ),
    );
    Ok([out_l?, out_r?])
}

async fn per_side(
    queries: &VecRequests<VecRots<QueryRef>>,
    missing_vector_ids: VecRequests<VecEdges<VectorId>>,
    sessions: &Vec<HawkSessionRef>,
) -> eyre::Result<VecRequests<MapEdges<bool>>> {
    // A request is to compare all rotations to a list of vectors - it is the length of the vector ids.
    let n_requests = missing_vector_ids.len();
    // A task is to compare one rotation to the vectors.
    let n_tasks = n_requests * ROTATIONS;
    let n_sessions = sessions.len();
    assert_eq!(queries.len(), n_requests);

    tracing::info!("per_side_match_batch sessions: {}", n_sessions);
    tracing::info!("per_side_match_batch requests:{}", n_requests);
    tracing::info!("per_side_match_batch tasks: {}", n_tasks);

    // Arc the requested vectors rather than cloning them.
    let missing_vector_ids = missing_vector_ids.into_iter().map(Arc::new);

    // For each request, broadcast the vectors to the rotations.
    // Concatenate the tasks for all requests, to maximize parallelism.
    let tasks = VecRots::flatten_broadcast(izip!(queries, missing_vector_ids));
    assert_eq!(tasks.len(), n_tasks);

    // Prepare the tasks into one chunk per session.
    let chunks = split_tasks(tasks, n_sessions);
    assert!(chunks.len() <= n_sessions);

    // Process the chunks in parallel (CPU and IO).
    let results = izip!(chunks, sessions)
        .map(|(chunk, session)| per_session(chunk, session.clone()))
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, JoinError>>()?
        .into_iter()
        .collect::<eyre::Result<Vec<_>>>()?;

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
    Ok(results)
}

async fn per_session(
    tasks: VecRequests<(QueryRef, Arc<VecEdges<VectorId>>)>,
    session: HawkSessionRef,
) -> eyre::Result<VecRequests<MapEdges<bool>>> {
    let mut session = session.write().await;

    let mut out = Vec::with_capacity(tasks.len());
    for (query, vectors) in tasks {
        let matches = per_query(query, &vectors, &mut session).await?;
        out.push(matches);
    }
    Ok(out)
}

async fn per_query(
    query: QueryRef,
    vector_ids: &[VectorId],
    session: &mut HawkSession,
) -> eyre::Result<MapEdges<bool>> {
    let distances = session
        .aby3_store
        .eval_distance_batch(&[query], vector_ids)
        .await?;

    let is_matches = session.aby3_store.is_match_batch(&distances).await?;

    Ok(izip!(vector_ids, is_matches)
        .map(|(v, m)| (*v, m))
        .collect())
}

/// Split a Vec into at most `n_sessions` chunks.
/// Chunks may not be of equal size.
/// Unlike `Itertools::chunks()` which borrows, this moves the items.
fn split_tasks<T>(tasks: Vec<T>, n_sessions: usize) -> Vec<Vec<T>> {
    let n_sessions = n_sessions.min(tasks.len());
    let chunk_size = tasks.len() / n_sessions;
    let rest_size = tasks.len() % n_sessions;

    let mut task_iter = tasks.into_iter();
    (0..n_sessions)
        .map(|chunk_i| {
            let one_rest = (chunk_i < rest_size) as usize;
            task_iter.by_ref().take(chunk_size + one_rest).collect_vec()
        })
        .collect_vec()
}

fn unsplit_tasks<T>(chunks: Vec<Vec<T>>) -> Vec<T> {
    chunks.into_iter().flatten().collect_vec()
}

fn aggregate_rotation_results(results: VecRots<MapEdges<bool>>) -> MapEdges<bool> {
    results.iter().fold(HashMap::new(), |mut acc, m| {
        for (v, is_match) in m {
            *acc.entry(*v).or_default() |= is_match;
        }
        acc
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_split_tasks() -> eyre::Result<()> {
        let tasks = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let expect = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        let per_query = |q| q * 10;
        let per_session = |chunk: Vec<i32>| Ok(chunk.into_iter().map(per_query).collect_vec());

        for n_sessions in [1, 2, 3, 7, 8, 9, 10] {
            let chunks = split_tasks(tasks.clone(), n_sessions);
            assert_eq!(chunks.len(), n_sessions.min(tasks.len()));

            let results = chunks
                .into_iter()
                .map(per_session)
                .collect::<eyre::Result<Vec<_>>>()?;

            let results = unsplit_tasks(results);
            assert_eq!(results, expect);
        }
        Ok(())
    }
}
