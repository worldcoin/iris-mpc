use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
    iris_db::db::IrisDB,
    job::{BatchMetadata, BatchQuery},
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
    scheduler::parallelize, tests::batch_of_party, HawkActor, HawkArgs, HawkRequest, VectorId,
    LEFT, RIGHT,
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

    let mut iris_stores = [
        actor.iris_store[LEFT].write().await,
        actor.iris_store[RIGHT].write().await,
    ];
    let mut registries = [
        actor.registry[LEFT].write().await,
        actor.registry[RIGHT].write().await,
    ];
    for (share, _mirror) in shares {
        iris_stores[LEFT].append(Arc::new(share.clone()));
        registries[LEFT].append(());
        // TODO: Different share.
        iris_stores[RIGHT].append(Arc::new(share.clone()));
        registries[RIGHT].append(());
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
    use crate::execution::hawk_main::tests::receive_batch_shares;
    let iris_rng = &mut AesRng::seed_from_u64(1337);

    let db = IrisDB::new_random_rng(batch_size * 2, iris_rng);
    // Use first batch_size irises for left eye, next batch_size for right eye.
    let left_shares: Vec<_> = db.db[..batch_size]
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
    let right_shares: Vec<_> = db.db[batch_size..]
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

    let our_batch = make_batch(batch_size);

    let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] =
        receive_batch_shares(&left_shares);
    let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] =
        receive_batch_shares(&right_shares);

    let mut my_batch = BatchQuery {
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

    // Request 2 is a mirror attack of request 0:
    //   request_2.left = mirror(request_0.right)
    //   request_2.right = mirror(request_0.left)
    // This way, mirror detection computes:
    //   mirror_preproc(mirror(request_0.left)) = preproc(request_0.left)
    //   vs raw(request_0.left) → self-match
    let n = my_batch.left_iris_requests.code.len();
    let right_code_0 = my_batch.right_iris_requests.code[0].clone();
    let right_mask_0 = my_batch.right_iris_requests.mask[0].clone();
    let left_code_0 = my_batch.left_iris_requests.code[0].clone();
    let left_mask_0 = my_batch.left_iris_requests.mask[0].clone();
    // request_2.left = mirror(request_0.right)
    my_batch.left_iris_requests.code[n - 1] = right_code_0.mirrored_code();
    my_batch.left_iris_requests.mask[n - 1] = right_mask_0.mirrored();
    // request_2.right = mirror(request_0.left)
    my_batch.right_iris_requests.code[n - 1] = left_code_0.mirrored_code();
    my_batch.right_iris_requests.mask[n - 1] = left_mask_0.mirrored();

    HawkRequest::from(my_batch)
}

fn make_batch(batch_size: usize) -> BatchQuery {
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

// TODO: Simplify and optimize share generation.
fn make_iris_share(
    batch_size: usize,
    party_id: usize,
) -> Vec<(GaloisRingSharedIris, GaloisRingSharedIris)> {
    let iris_rng = &mut AesRng::seed_from_u64(1337);

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
