use super::{
    scheduler::{Batch, Schedule, Task},
    BothEyes, HawkSession, VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::{scheduler::parallelize, VecRotations},
    hawkers::aby3::aby3_store::Aby3Query,
    hnsw::VectorStore,
    protocol::shared_iris::ArcIris,
};
use eyre::Result;
use itertools::{izip, Itertools};
use std::{collections::HashMap, sync::Arc, time::Instant, vec};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IntraMatch {
    pub other_request_i: usize,
    pub is_match: BothEyes<bool>,
}

pub async fn intra_batch_is_match(
    sessions: &BothEyes<Vec<HawkSession>>,
    search_queries: &Arc<BothEyes<VecRequests<VecRotations<Aby3Query>>>>,
) -> Result<VecRequests<Vec<IntraMatch>>> {
    let start = Instant::now();
    let n_sessions = sessions[LEFT].len();
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_requests = search_queries[LEFT].len();
    assert_eq!(n_requests, search_queries[RIGHT].len());
    let n_rotations = VecRotations::<Aby3Query>::n_rotations();

    let batches = Schedule::new(n_sessions, n_requests, n_rotations).intra_match_batches();

    let (tx, rx) = unbounded_channel::<IsMatch>();

    let per_session = |batch: Batch| {
        let session = sessions[batch.i_eye][batch.i_session].clone();
        let search_queries = search_queries.clone();
        let tx = tx.clone();
        async move { per_session(&search_queries, &session, batch, tx).await }
    };

    parallelize(batches.into_iter().map(per_session)).await?;

    let res = aggregate_results(n_requests, rx).await?;

    metrics::histogram!("intra_batch_duration").record(start.elapsed().as_secs_f64());
    Ok(res)
}

async fn per_session(
    search_queries: &BothEyes<VecRequests<VecRotations<Aby3Query>>>,
    session: &HawkSession,
    batch: Batch,
    tx: UnboundedSender<IsMatch>,
) -> Result<()> {
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
    let query_pairs: Vec<Option<(ArcIris, ArcIris)>> = pairs
        .iter()
        .map(|pair| {
            let iris1_proc =
                &search_queries[batch.i_eye][pair.task.i_request][pair.task.i_rotation].iris_proc;
            let iris2 = &search_queries[batch.i_eye][pair.earlier_request]
                .center()
                .iris;
            Some((iris1_proc.clone(), iris2.clone()))
        })
        .collect_vec();

    let mut store = session.aby3_store.write().await;

    let distances = store.eval_pairwise_distances(query_pairs).await?;
    let is_matches = store.is_match_batch(&distances).await?;

    for (pair, is_match) in izip!(pairs, is_matches) {
        if is_match {
            tx.send(pair)?;
        }
    }

    Ok(())
}

struct IsMatch {
    eye: usize,
    task: Task,
    earlier_request: usize,
}

async fn aggregate_results(
    n_requests: usize,
    mut rx: UnboundedReceiver<IsMatch>,
) -> Result<VecRequests<Vec<IntraMatch>>> {
    rx.close();
    let mut join = HashMap::new();

    // For each pair of request, reduce the result of all rotations with boolean ANY.
    while let Some(match_result) = rx.recv().await {
        let request_pair = (match_result.task.i_request, match_result.earlier_request);
        let eyes_match = join.entry(request_pair).or_insert([false, false]);
        eyes_match[match_result.eye] = true;
    }

    let mut match_lists = vec![Vec::new(); n_requests];

    // Report pairs with a left OR right match.
    #[allow(clippy::iter_over_hash_type, reason = "TODO")]
    for ((i_request, earlier_request), [left, right]) in join {
        if left || right {
            // This request matches a request that came before it in the batch.
            match_lists[i_request].push(IntraMatch {
                other_request_i: earlier_request,
                is_match: [left, right],
            });
        }
    }

    Ok(match_lists)
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::setup_hawk_actors;
    use super::*;
    use crate::execution::hawk_main::test_utils::make_request_intra_match;
    use crate::execution::hawk_main::{HawkActor, Orientation};
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_intra_batch_is_match() -> Result<()> {
        let actors = setup_hawk_actors().await?;

        parallelize(actors.into_iter().map(go_intra_batch)).await?;

        Ok(())
    }

    async fn go_intra_batch(mut actor: HawkActor) -> Result<HawkActor> {
        let sessions = actor.new_sessions().await?;

        let batch_size = 3;
        let request = make_request_intra_match(batch_size, actor.party_id);
        let search_queries = &request.queries(Orientation::Normal);

        let result = intra_batch_is_match(&sessions, search_queries).await?;

        assert_eq!(
            result,
            vec![
                vec![], // First request cannot have a match.
                vec![], // Second request has no matches.
                // Third request matches the first one (see make_request_intra_match).
                vec![IntraMatch {
                    other_request_i: 0,
                    is_match: [true, true],
                }],
            ]
        );
        actor.sync_peers().await?;
        Ok(actor)
    }
}
