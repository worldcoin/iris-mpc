use super::{BothEyes, HawkSession, MapEdges, VecEdges, VecRequests, VectorId, LEFT, RIGHT};
use crate::{
    execution::hawk_main::VecRotations,
    hawkers::aby3::aby3_store::{Aby3Query, Aby3Store},
    hnsw::VectorStore,
};
use eyre::Result;
use futures::future::JoinAll;
use itertools::{izip, Itertools};
use std::{collections::HashMap, sync::Arc, time::Instant};
use tokio::task::JoinError;

pub async fn is_match_batch(
    search_queries: &BothEyes<VecRequests<VecRotations<Aby3Query>>>,
    vector_ids: BothEyes<VecRequests<VecEdges<VectorId>>>,
    sessions: &BothEyes<Vec<HawkSession>>,
) -> Result<BothEyes<VecRequests<MapEdges<bool>>>> {
    let start = Instant::now();
    let [vectors_left, vectors_right] = vector_ids;

    // Parallelize left and right sessions (IO only).
    let (out_l, out_r) = futures::join!(
        per_side(&search_queries[LEFT], vectors_left, &sessions[LEFT]),
        per_side(&search_queries[RIGHT], vectors_right, &sessions[RIGHT]),
    );

    metrics::histogram!("is_match_batch_duration").record(start.elapsed().as_secs_f64());
    Ok([out_l?, out_r?])
}

async fn per_side(
    queries: &VecRequests<VecRotations<Aby3Query>>,
    missing_vector_ids: VecRequests<VecEdges<VectorId>>,
    sessions: &Vec<HawkSession>,
) -> Result<VecRequests<MapEdges<bool>>> {
    // A request is to compare all rotations to a list of vectors - it is the length of the vector ids.
    let n_requests = missing_vector_ids.len();
    // If there are no missing vectors, there is nothing to match.
    if n_requests == 0 {
        return Ok(VecRequests::new());
    }
    // A task is to compare one rotation to the vectors.
    let n_tasks = n_requests * VecRotations::<Aby3Query>::n_rotations();
    let n_sessions = sessions.len();
    assert_eq!(queries.len(), n_requests);

    tracing::info!("per_side_match_batch sessions: {}", n_sessions);
    tracing::info!("per_side_match_batch requests:{}", n_requests);
    tracing::info!("per_side_match_batch tasks: {}", n_tasks);

    // Arc the requested vectors rather than cloning them.
    let missing_vector_ids = missing_vector_ids.into_iter().map(Arc::new);

    // For each request, broadcast the vectors to the rotations.
    // Concatenate the tasks for all requests, to maximize parallelism.
    let tasks = VecRotations::flatten_broadcast(izip!(queries, missing_vector_ids));
    assert_eq!(tasks.len(), n_tasks);

    // Prepare the tasks into one chunk per session.
    let chunks = split_tasks(tasks, n_sessions);
    assert!(chunks.len() <= n_sessions);

    // Process the chunks in parallel (CPU and IO).
    let results = izip!(chunks, sessions)
        .map(|(chunk, session)| per_session(chunk, session.clone()))
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await;

    // Undo the chunking per session.
    let results = unsplit_tasks(results)?;
    assert_eq!(results.len(), n_tasks);

    // Undo the flattening of rotations.
    let results = VecRotations::unflatten(results);

    // Aggregate the results over rotations. ANY match.
    let results = results
        .into_iter()
        .map(aggregate_rotation_results)
        .collect_vec();

    assert_eq!(results.len(), n_requests);
    Ok(results)
}

async fn per_session(
    tasks: VecRequests<(Aby3Query, Arc<VecEdges<VectorId>>)>,
    session: HawkSession,
) -> Result<VecRequests<MapEdges<bool>>> {
    let mut store = session.aby3_store.write().await;

    let mut out = Vec::with_capacity(tasks.len());
    for (query, vectors) in tasks {
        let matches = per_query(query, &vectors, &mut store).await?;
        out.push(matches);
    }
    Ok(out)
}

async fn per_query(
    query: Aby3Query,
    vector_ids: &[VectorId],
    store: &mut Aby3Store,
) -> Result<MapEdges<bool>> {
    let distances = store.eval_distance_batch(&query, vector_ids).await?;

    let is_matches = store.is_match_batch(&distances).await?;

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

fn unsplit_tasks<T>(chunks: Vec<std::result::Result<Result<Vec<T>>, JoinError>>) -> Result<Vec<T>> {
    chunks
        .into_iter()
        .flatten()
        .collect::<Result<Vec<_>>>()
        .map(|v| v.into_iter().flatten().collect())
}

fn aggregate_rotation_results(results: VecRotations<MapEdges<bool>>) -> MapEdges<bool> {
    results.iter().fold(HashMap::new(), |mut acc, m| {
        for (v, is_match) in m {
            *acc.entry(*v).or_default() |= is_match;
        }
        acc
    })
}

#[cfg(test)]
mod test {
    use super::super::test_utils::setup_hawk_actors;
    use super::*;
    use crate::execution::hawk_main::scheduler::parallelize;
    use crate::execution::hawk_main::test_utils::{init_iris_db, make_request};
    use crate::execution::hawk_main::{HawkActor, Orientation};

    #[tokio::test]
    async fn test_split_tasks() {
        let tasks = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let expect = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        let per_query = |q| q * 10;
        let per_session = |chunk: Vec<i32>| {
            Ok::<Result<Vec<i32>>, JoinError>(Ok(chunk.into_iter().map(per_query).collect_vec()))
        };

        for n_sessions in [1, 2, 3, 7, 8, 9, 10] {
            let chunks = split_tasks(tasks.clone(), n_sessions);
            assert_eq!(chunks.len(), n_sessions.min(tasks.len()));

            let results = chunks.into_iter().map(per_session).collect_vec();

            let results = unsplit_tasks(results).unwrap();
            assert_eq!(results, expect);
        }
    }

    #[tokio::test]
    async fn test_is_match_batch() -> Result<()> {
        let actors = setup_hawk_actors().await?;

        parallelize(actors.into_iter().map(go_is_match_batch)).await?;

        Ok(())
    }

    async fn go_is_match_batch(mut actor: HawkActor) -> Result<HawkActor> {
        init_iris_db(&mut actor).await?;

        let [sessions, _mirror] = actor.new_sessions_orient().await?;

        let batch_size = 3;
        let request = make_request(batch_size, actor.party_id);
        let search_queries = &request.queries(Orientation::Normal);

        let missing_vector_ids = [
            vec![
                vec![],                          // Empty.
                vec![VectorId::from_0_index(1)], // Match because request 1 is the same as db entry 1.
                vec![VectorId::from_0_index(1)], // Non-match.
            ],
            vec![
                vec![VectorId::from_0_index(0), VectorId::from_0_index(1)], // Match and non-match.
                vec![VectorId::from_0_index(999)],                          // Non-existing vector.
                vec![VectorId::new(1, 999)],                                // Non-existing version.
            ],
        ];

        let result = is_match_batch(search_queries, missing_vector_ids.clone(), &sessions).await?;

        assert_eq!(
            result,
            [
                expected_matches(&missing_vector_ids[LEFT]),
                expected_matches(&missing_vector_ids[RIGHT]),
            ]
        );

        // Do not drop the connections too early.
        Ok(actor)
    }

    fn expected_matches(requested_ids: &[VecEdges<VectorId>]) -> Vec<HashMap<VectorId, bool>> {
        let mut expect = vec![];
        for (iris_id, vector_ids) in requested_ids.iter().enumerate() {
            let mut map = HashMap::new();

            for vid in vector_ids {
                // We get matches by construction of init_db and make_request.
                let is_match = iris_id == vid.index() as usize;
                map.insert(*vid, is_match);
            }

            expect.push(map);
        }
        expect
    }
}
