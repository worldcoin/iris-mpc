use super::{
    rot::VecRots,
    scheduler::{schedule, Batch, Task},
    BothEyes, HawkSession, HawkSessionRef, VecRequests, LEFT, RIGHT,
};
use crate::{hawkers::aby3::aby3_store::QueryRef, hnsw::VectorStore};
use eyre::Result;
use futures::future::JoinAll;
use iris_mpc_common::ROTATIONS;
use itertools::{izip, Itertools};
use std::{collections::HashMap, sync::Arc, vec};
use tokio::task::JoinError;

pub async fn intra_batch_is_match(
    sessions: &BothEyes<Vec<HawkSessionRef>>,
    search_queries: &BothEyes<VecRequests<VecRots<QueryRef>>>,
) -> Result<VecRequests<Vec<usize>>> {
    let n_sessions = sessions[LEFT].len();
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_eyes = 2;
    let n_requests = search_queries[LEFT].len();
    assert_eq!(n_requests, search_queries[RIGHT].len());

    let batches = schedule(n_sessions, n_eyes, n_requests, ROTATIONS).batches;

    // TODO: move this up to the caller.
    let search_queries = [
        Arc::new(search_queries[LEFT].clone()),
        Arc::new(search_queries[RIGHT].clone()),
    ];

    let per_session = |batch: Batch| {
        let search_queries = search_queries[batch.i_eye].clone();
        let session = sessions[batch.i_eye][batch.i_session].clone();
        async move {
            let mut session = session.write().await;
            per_session(&search_queries, &mut session, batch).await
        }
    };

    let results = batches
        .into_iter()
        .map(per_session)
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await;

    aggregate_results(n_requests, results)
}

async fn per_session(
    search_queries: &VecRequests<VecRots<QueryRef>>,
    session: &mut HawkSession,
    batch: Batch,
) -> Vec<IsMatch> {
    // Enumerate the pairs of requests.
    // These are unordered pairs: if we do (i, j) we skip (j, i).
    let pairs = batch
        .tasks
        .into_iter()
        .flat_map(|task| {
            (0..task.i_request).map(move |earlier_request| IsMatch {
                eye: batch.i_eye,
                task,
                earlier_request,
            })
        })
        .collect_vec();

    // Compare the rotated and processed irises of one request, to the centered unprocessed iris of the other request.
    let query_pairs = pairs
        .iter()
        .map(|pair| {
            (
                &search_queries[pair.task.i_request][pair.task.i_rotation].processed_query,
                &search_queries[pair.earlier_request].center().query,
            )
        })
        .collect_vec();

    let distances = session
        .aby3_store
        .eval_pairwise_distances(query_pairs)
        .await;
    let distances = session.aby3_store.lift_distances(distances).await.unwrap();
    let is_matches = session.aby3_store.is_match_batch(&distances).await;

    izip!(pairs, is_matches)
        .filter_map(|(pair, is_match)| is_match.then_some(pair))
        .collect_vec()
}

struct IsMatch {
    eye: usize,
    task: Task,
    earlier_request: usize,
}

fn aggregate_results(
    n_requests: usize,
    results: Vec<Result<Vec<IsMatch>, JoinError>>,
) -> Result<VecRequests<Vec<usize>>> {
    let mut join = HashMap::new();

    // For each pair of request, reduce the result of all rotations with boolean ANY.
    for batch in results {
        for match_result in batch? {
            let request_pair = (match_result.task.i_request, match_result.earlier_request);
            let eyes_match = join.entry(request_pair).or_insert([false, false]);
            eyes_match[match_result.eye] = true;
        }
    }

    let mut match_lists = vec![Vec::new(); n_requests];

    // Find pairs with left AND right match.
    for ((i_request, earlier_request), [left, right]) in join {
        if left && right {
            // This request is a duplicate of a request that came before it in the batch.
            match_lists[i_request].push(earlier_request);
        }
    }

    Ok(match_lists)
}
