use super::{
    rot::VecRots, scheduler::schedule, BothEyes, HawkSession, HawkSessionRef, MapEdges, VecEdges,
    VecRequests, VectorId, LEFT, RIGHT,
};
use crate::{hawkers::aby3::aby3_store::QueryRef, hnsw::VectorStore};
use futures::future::JoinAll;
use iris_mpc_common::ROTATIONS;
use itertools::{izip, Itertools};
use std::{collections::HashMap, sync::Arc};
use tokio::task::JoinError;

pub async fn intra_batch_is_match(
    search_queries: &BothEyes<VecRequests<VecRots<QueryRef>>>,
    sessions: &BothEyes<Vec<HawkSessionRef>>,
) -> VecRequests<Vec<usize>> {
    let batches = schedule(
        2,
        search_queries[LEFT].len(),
        sessions[LEFT].len(),
        ROTATIONS,
    );

    let results = batches
        .iter()
        .map(|batch| async { false })
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await;

    let other_queries = [LEFT, RIGHT].map(|side| {
        search_queries[side]
            .iter()
            .map(|rots| rots.center().clone())
            .collect_vec()
    });

    vec![]
}
