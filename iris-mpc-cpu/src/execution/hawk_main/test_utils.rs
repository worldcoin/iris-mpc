use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    galois_engine::degree4::preprocess_iris_message_shares,
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
    iris_db::db::IrisDB,
    job::{BatchMetadata, BatchQuery, IrisQueryBatchEntries},
    ROTATIONS,
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

use super::{scheduler::parallelize, HawkActor, HawkArgs, HawkRequest, VectorId, LEFT, RIGHT};

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
    for (share, _mirror) in shares {
        iris_stores[LEFT].append(Arc::new(share.clone()));
        // TODO: Different share.
        iris_stores[RIGHT].append(Arc::new(share.clone()));
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

    // Copy the iris of the first into the last request.
    let len = my_batch.left_iris_interpolated_requests.code.len();
    for x in [
        &mut my_batch.left_iris_interpolated_requests,
        &mut my_batch.right_iris_interpolated_requests,
    ] {
        // Whichever rotation of the last request <-- center from the first request.
        x.code[len - 1 - ROTATIONS / 2] = x.code[ROTATIONS / 2].clone();
        x.mask[len - 1 - ROTATIONS / 2] = x.mask[ROTATIONS / 2].clone();
    }

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
