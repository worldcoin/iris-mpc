use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    galois_engine::degree4::preprocess_iris_message_shares,
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
    iris_db::db::IrisDB,
    job::{BatchMetadata, BatchQuery, IrisQueryBatchEntries},
};
use itertools::Itertools;
use rand::SeedableRng;
use std::{sync::Arc, time::Duration};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::{
    execution::local::get_free_local_addresses,
    hnsw::searcher::{build_layer_updates, ConnectPlan, LayerMode, UpdateEntryPoint},
    protocol::shared_iris::GaloisRingSharedIris,
    utils::constants::N_PARTIES,
};

use super::{
    iris_worker::{IrisWorkerPool, QueryId},
    scheduler::parallelize,
    HawkActor, HawkArgs, HawkRequest, VectorId, LEFT, RIGHT,
};

pub async fn setup_hawk_actors() -> Result<Vec<HawkActor>> {
    let go = |addresses: Vec<String>, index: usize| {
        async move {
            let args = HawkArgs::parse_from([
                "hawk_main",
                "--addresses",
                &addresses.join(","),
                "--outbound-addrs",
                &addresses.join(","),
                "--party-index",
                &index.to_string(),
            ]);

            // Make the test async.
            sleep(Duration::from_millis(index as u64)).await;

            HawkActor::from_cli(&args, CancellationToken::new()).await
        }
    };

    let addresses = get_free_local_addresses(N_PARTIES).await?;

    let actors = parallelize((0..N_PARTIES).map(|i| go(addresses.clone(), i))).await?;

    Ok(actors)
}

pub async fn init_iris_db(actor: &mut HawkActor) -> Result<()> {
    let db_size = 5;
    let shares = make_iris_share(db_size, actor.party_id);

    for (share, _mirror) in shares {
        let iris = Arc::new(share);
        for side in [LEFT, RIGHT] {
            let qid = QueryId::new();
            actor.worker_pools[side]
                .cache_queries(vec![(qid, iris.clone())])
                .await?;
            let vector_id = {
                let mut reg = actor.registry[side].write().await;
                let id = reg.allocate_next_id();
                reg.insert(id, ());
                id
            };
            actor.worker_pools[side]
                .insert_irises(vec![(qid, vector_id)])
                .await?;
            actor.worker_pools[side].evict_queries(vec![qid]).await?;
        }
    }
    Ok(())
}

/// Populate the graphs such that all vectors are reachable.
pub async fn init_graph(actor: &mut HawkActor) -> Result<()> {
    let db_size = 5;
    let layer_mode = actor.searcher().layer_mode.clone();
    let id = |i: usize| VectorId::from_0_index(i as u32);
    let next = |i: usize| (i + 1) % db_size;
    let edges = |i: usize| vec![id(next(i))];

    for side in [LEFT, RIGHT] {
        let mut graph = actor.graph_store[side].write().await;
        for i in 0..db_size {
            let plan = ConnectPlan {
                inserted_vector: id(i),
                updates: build_layer_updates(id(i), edges(i), vec![edges(next(i))], 0),
                update_ep: if i == 0 {
                    match layer_mode {
                        LayerMode::Standard { .. } => UpdateEntryPoint::SetUnique { layer: 0 },
                        LayerMode::LinearScan { .. } => UpdateEntryPoint::False,
                    }
                } else {
                    UpdateEntryPoint::False
                },
            };
            graph.insert_apply(plan).await;
        }
    }

    Ok(())
}

pub fn make_request(batch_size: usize, party_id: usize) -> HawkRequest {
    let shares = make_iris_share(batch_size, party_id);
    let our_batch = make_batch(batch_size);
    let my_batch = batch_of_party(&our_batch, &shares);
    HawkRequest::from(my_batch)
}

// Create a batch where the last request matches the first one.
pub fn make_request_intra_match(batch_size: usize, party_id: usize) -> HawkRequest {
    let shares = make_iris_share(batch_size, party_id);
    let our_batch = make_batch(batch_size);
    let mut my_batch = batch_of_party(&our_batch, &shares);

    // Copy the original iris of the first request into the last request.
    // In the new design, the worker pool caches from `left/right_iris_requests`
    // (the originals), so we modify those to create the intra-batch duplicate.
    let n = my_batch.left_iris_requests.code.len();
    for x in [
        &mut my_batch.left_iris_requests,
        &mut my_batch.right_iris_requests,
    ] {
        x.code[n - 1] = x.code[0].clone();
        x.mask[n - 1] = x.mask[0].clone();
    }

    HawkRequest::from(my_batch)
}

/// Create a batch for mirror intra-batch testing with different left/right eyes.
///
/// Request 2 is a mirror attack of request 0: its left eye is mirror(request_0.right)
/// and its right eye is mirror(request_0.left). With the correct "b" operand
/// (same-eye raw iris from Normal queries), mirror detection should find a match.
/// With the buggy "b" operand (wrong eye from Mirror queries), it won't.
pub fn make_request_intra_match_mirror(batch_size: usize, party_id: usize) -> HawkRequest {
    let iris_rng = &mut AesRng::seed_from_u64(1337);

    let db = IrisDB::new_random_rng(batch_size * 2, iris_rng);
    // Use first batch_size irises for left eye, next batch_size for right eye.
    let mut left_shares: Vec<_> = db.db[..batch_size]
        .iter()
        .map(|iris| {
            (
                GaloisRingSharedIris::generate_shares_locally(iris_rng, iris.clone())[party_id]
                    .clone(),
                GaloisRingSharedIris::generate_mirrored_shares_locally(iris_rng, iris.clone())
                    [party_id]
                    .clone(),
            )
        })
        .collect();
    let mut right_shares: Vec<_> = db.db[batch_size..]
        .iter()
        .map(|iris| {
            (
                GaloisRingSharedIris::generate_shares_locally(iris_rng, iris.clone())[party_id]
                    .clone(),
                GaloisRingSharedIris::generate_mirrored_shares_locally(iris_rng, iris.clone())
                    [party_id]
                    .clone(),
            )
        })
        .collect();

    // Request 2 is a mirror attack of request 0:
    //   request_2.left = mirror(request_0.right)
    //   request_2.right = mirror(request_0.left)
    // Each entry is (normal_shares, mirrored_shares) for the same iris. Swapping
    // the tuple is equivalent to mirroring: (shares(x), shares(mirror(x))) becomes
    // (shares(mirror(x)), shares(x)). Apply at the shares level so all downstream
    // derivations (rotated, interpolated, mirrored-interpolated) stay consistent.
    let n = left_shares.len();
    let (right_normal_0, right_mirrored_0) = right_shares[0].clone();
    let (left_normal_0, left_mirrored_0) = left_shares[0].clone();
    left_shares[n - 1] = (right_mirrored_0, right_normal_0);
    right_shares[n - 1] = (left_mirrored_0, left_normal_0);

    let our_batch = make_batch(batch_size);

    let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] =
        receive_batch_shares(&left_shares);
    let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] =
        receive_batch_shares(&right_shares);

    let my_batch = BatchQuery {
        left_iris_requests,
        right_iris_requests,
        left_iris_rotated_requests,
        right_iris_rotated_requests,
        left_iris_interpolated_requests,
        right_iris_interpolated_requests,
        left_mirrored_iris_interpolated_requests,
        right_mirrored_iris_interpolated_requests,
        ..our_batch
    };

    HawkRequest::from(my_batch)
}

pub fn make_batch(batch_size: usize) -> BatchQuery {
    let mut batch = BatchQuery {
        luc_lookback_records: 2,
        ..BatchQuery::default()
    };
    for i in 0..batch_size {
        batch.push_matching_request(
            format!("sns_{i}"),
            format!("request_{i}"),
            UNIQUENESS_MESSAGE_TYPE,
            BatchMetadata::default(),
            vec![],
            false,
        );
    }
    batch
}

pub fn receive_batch_shares(
    shares_with_mirror: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
) -> [IrisQueryBatchEntries; 4] {
    let mut out = [(); 4].map(|_| IrisQueryBatchEntries::default());
    for (share, mirrored_share) in shares_with_mirror.iter().cloned() {
        let one = preprocess_iris_message_shares(
            share.code,
            share.mask,
            mirrored_share.code,
            mirrored_share.mask,
        )
        .unwrap();
        out[0].code.push(one.code);
        out[0].mask.push(one.mask);
        out[1].code.extend(one.code_rotated);
        out[1].mask.extend(one.mask_rotated);
        out[2].code.extend(one.code_interpolated.clone());
        out[2].mask.extend(one.mask_interpolated.clone());
        out[3].code.extend(one.code_mirrored);
        out[3].mask.extend(one.mask_mirrored);
    }
    out
}

pub fn batch_of_party(
    batch: &BatchQuery,
    shares_with_mirror: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
) -> BatchQuery {
    let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] =
        receive_batch_shares(shares_with_mirror);
    let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] =
        receive_batch_shares(shares_with_mirror);

    BatchQuery {
        left_iris_requests,
        right_iris_requests,
        left_iris_rotated_requests,
        right_iris_rotated_requests,
        left_iris_interpolated_requests,
        right_iris_interpolated_requests,
        left_mirrored_iris_interpolated_requests,
        right_mirrored_iris_interpolated_requests,
        ..batch.clone()
    }
}

// TODO: Simplify and optimize share generation.
pub fn make_iris_share(
    batch_size: usize,
    party_id: usize,
) -> Vec<(GaloisRingSharedIris, GaloisRingSharedIris)> {
    make_iris_share_with_seed(batch_size, party_id, 1337)
}

pub fn make_iris_share_with_seed(
    batch_size: usize,
    party_id: usize,
    seed: u64,
) -> Vec<(GaloisRingSharedIris, GaloisRingSharedIris)> {
    let iris_rng = &mut AesRng::seed_from_u64(seed);

    // Generate: iris_id -> share
    IrisDB::new_random_rng(batch_size, iris_rng)
        .db
        .into_iter()
        .map(|iris| {
            (
                GaloisRingSharedIris::generate_shares_locally(iris_rng, iris.clone())[party_id]
                    .clone(),
                GaloisRingSharedIris::generate_mirrored_shares_locally(iris_rng, iris)[party_id]
                    .clone(),
            )
        })
        .collect_vec()
}
