use super::{
    BothEyes, HawkSession, HawkSessionRef, InsertPlan, MapEdges, VecEdges, VecRequests, VectorId,
    LEFT, RIGHT,
};
use crate::{hawkers::aby3::aby3_store::QueryRef, hnsw::VectorStore};
use futures::future::JoinAll;
use itertools::{izip, Itertools};
use std::iter;

pub async fn calculate_is_match(
    plans: &BothEyes<VecRequests<InsertPlan>>,
    vector_ids: VecRequests<BothEyes<VecEdges<VectorId>>>,
    sessions: &BothEyes<Vec<HawkSessionRef>>,
) -> VecRequests<BothEyes<MapEdges<bool>>> {
    let n_requests = vector_ids.len();
    let n_sessions = sessions[LEFT].len();
    let n_per_session = n_requests.div_ceil(n_sessions);
    assert_eq!(plans[LEFT].len(), n_requests);
    assert_eq!(plans[RIGHT].len(), n_requests);

    // Organize as: request -> eye -> (query and its related vectors).
    let requests = izip!(&plans[LEFT], &plans[RIGHT], vector_ids).map(|(pl, pr, vector_ids)| {
        let [vectors_l, vectors_r] = vector_ids;
        [(pl.query.clone(), vectors_l), (pr.query.clone(), vectors_r)]
    });
    // Prepare the requests into one chunk per session.
    let per_session = chunk_vecs(requests, n_per_session);

    // Process the chunks in parallel (CPU and IO).
    let results = izip!(per_session, &sessions[LEFT], &sessions[RIGHT])
        .map(|(chunk, sl, sr)| calculate_is_match_session(chunk, [sl.clone(), sr.clone()]))
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await;
    assert!(results.len() == n_sessions.min(n_requests));

    let r = results.into_iter().flat_map(Result::unwrap).collect_vec();
    assert_eq!(r.len(), n_requests);
    r
}

async fn calculate_is_match_session(
    requests: VecRequests<BothEyes<(QueryRef, VecEdges<VectorId>)>>,
    session: BothEyes<HawkSessionRef>,
) -> VecRequests<BothEyes<MapEdges<bool>>> {
    let mut session_l = session[LEFT].write().await;
    let mut session_r = session[RIGHT].write().await;

    let mut out = Vec::with_capacity(requests.len());
    for request in requests {
        let [(query_l, vectors_l), (query_r, vectors_r)] = &request;
        // Parallelize left and right sessions (IO only).
        let (out_l, out_r) = futures::join!(
            calculate_is_match_one(query_l, vectors_l, &mut session_l),
            calculate_is_match_one(query_r, vectors_r, &mut session_r),
        );
        out.push([out_l, out_r]);
    }
    out
}

async fn calculate_is_match_one(
    query: &QueryRef,
    vector_ids: &[VectorId],
    session: &mut HawkSession,
) -> MapEdges<bool> {
    let distances = session
        .aby3_store
        .eval_distance_batch(query, vector_ids)
        .await;

    let is_matches = session.aby3_store.is_match_batch(&distances).await;

    izip!(vector_ids, is_matches)
        .map(|(v, m)| (*v, m))
        .collect()
}

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
