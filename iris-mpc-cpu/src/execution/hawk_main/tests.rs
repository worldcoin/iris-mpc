use super::*;
use crate::{
    execution::local::get_free_local_addresses, protocol::shared_iris::GaloisRingSharedIris,
    utils::constants::N_PARTIES,
};
use aes_prng::AesRng;
use futures::future::JoinAll;
use iris_mpc_common::{
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE, iris_db::db::IrisDB, job::BatchMetadata,
};
use rand::SeedableRng;
use std::{ops::Not, time::Duration};
use tokio::time::sleep;
use tracing::{info_span, Instrument};
use tracing_test::traced_test;

/// `Seeded` initializer: the post-construction registry contains
/// every seeded `VectorId`, with `next_id = max(serial) + 1`.
#[tokio::test]
async fn test_seeded_initializer_populates_registry() -> Result<()> {
    use crate::hawkers::aby3::aby3_store::Aby3Store;
    use crate::protocol::shared_iris::GaloisRingSharedIris;
    use std::collections::HashMap;

    let addresses = vec![
        "0.0.0.0:21340".to_string(),
        "0.0.0.0:21341".to_string(),
        "0.0.0.0:21342".to_string(),
    ];
    let args = HawkArgs {
        party_index: 0,
        addresses: addresses.clone(),
        outbound_addrs: addresses,
        request_parallelism: 4,
        connection_parallelism: 2,
        hnsw_param_ef_constr: 320,
        hnsw_param_m: 256,
        hnsw_param_ef_search: 256,
        hnsw_param_ef_search_layers_override: None,
        hnsw_param_ef_supermatch: 4000,
        hnsw_param_ef_saturation_margin: 0,
        hnsw_layer_density: None,
        hnsw_min_layer_search_batch_size: None,
        hnsw_prf_key: None,
        numa: false,
        disable_persistence: true,
        hnsw_disable_memory_persistence: false,
        tls: None,
    };

    let serial_ids = [0u32, 1, 2, 5];
    let dummy = Arc::new(GaloisRingSharedIris::default_for_party(args.party_index));
    let make_store = || {
        let mut points = HashMap::new();
        for sid in serial_ids {
            points.insert(VectorId::from_serial_id(sid), dummy.clone());
        }
        Aby3Store::<HawkOps>::new_storage(Some(points))
    };
    let iris_store = [make_store(), make_store()];
    let graph = [(); 2].map(|_| GraphMem::new());

    let hawk_actor = HawkActor::from_cli_with_graph_and_store(
        &args,
        CancellationToken::new(),
        graph,
        iris_store,
    )
    .await?;

    for sid in serial_ids {
        assert!(
            hawk_actor.registry[LEFT]
                .read()
                .await
                .contains(&VectorId::from_serial_id(sid)),
            "left registry must contain seeded VectorId {sid} after construction"
        );
        assert!(
            hawk_actor.registry[RIGHT]
                .read()
                .await
                .contains(&VectorId::from_serial_id(sid)),
            "right registry must contain seeded VectorId {sid} after construction"
        );
    }
    assert_eq!(
        hawk_actor.registry[LEFT].read().await.next_id,
        *serial_ids.last().unwrap() + 1,
        "registry next_id must reflect (max seeded serial_id + 1)"
    );

    Ok(())
}

#[tokio::test]
#[traced_test]
async fn test_hawk_main() -> Result<()> {
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
                "--hnsw-param-ef-constr",
                &320.to_string(),
                "--hnsw-param-m",
                &256.to_string(),
            ]);

            // Make the test async.
            sleep(Duration::from_millis(100 * index as u64)).await;

            hawk_main(args).await.unwrap()
        }
    };

    let addresses = get_free_local_addresses(N_PARTIES).await?;

    let handles = (0..N_PARTIES)
        .map(|i| {
            let span = info_span!("mpc_node", idx = i);
            let future = go(addresses.clone(), i);
            tokio::spawn(future.instrument(span))
        })
        .collect::<JoinAll<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<HawkHandle>, _>>()?;

    // ---- Send requests ----

    let batch_size = 5;
    let iris_rng = &mut AesRng::seed_from_u64(1337);

    // Generate: iris_id -> party -> share
    let irises = IrisDB::new_random_rng(batch_size, iris_rng)
        .db
        .into_iter()
        .map(|iris| {
            (
                GaloisRingSharedIris::generate_shares_locally(iris_rng, iris.clone()),
                GaloisRingSharedIris::generate_mirrored_shares_locally(iris_rng, iris),
            )
        })
        .collect_vec();

    // Unzip: party -> iris_id -> (share, share_mirrored)
    let irises = (0..N_PARTIES)
        .map(|party_index| {
            irises
                .iter()
                .map(|(iris, iris_mirrored)| {
                    (
                        iris[party_index].clone(),
                        iris_mirrored[party_index].clone(),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let mut batch_0 = BatchQuery {
        luc_lookback_records: 2,
        ..BatchQuery::default()
    };
    for i in 0..batch_size {
        batch_0.push_matching_request(
            format!("sns_{i}"),
            format!("request_{i}"),
            UNIQUENESS_MESSAGE_TYPE,
            BatchMetadata::default(),
            vec![],
            false,
        );
    }

    let all_results = parallelize(izip!(0..handles.len(), &irises, handles.clone()).map(
        |(idx, shares, mut handle)| {
            let span = info_span!("mpc_node", idx = idx);
            let batch = batch_of_party(&batch_0, shares);
            async move {
                handle
                    .submit_batch_query(batch)
                    .instrument(span)
                    .await
                    .await
            }
        },
    ))
    .await?;

    let result = assert_all_equal(all_results);

    let inserted_indices = (0..batch_size as u32).collect_vec();

    assert_eq!(result.matches, vec![false; batch_size]);
    assert_eq!(result.merged_results, inserted_indices);
    assert_eq!(batch_size, result.request_ids.len());
    assert_eq!(batch_size, result.request_types.len());
    assert_eq!(batch_size, result.metadata.len());
    assert_eq!(batch_size, result.matches_with_skip_persistence.len());
    assert_eq!(result.match_ids, vec![Vec::<u32>::new(); batch_size]);
    assert_eq!(batch_size, result.partial_match_ids_left.len());
    assert_eq!(batch_size, result.partial_match_ids_right.len());
    assert_eq!(batch_size, result.partial_match_counters_left.len());
    assert_eq!(batch_size, result.partial_match_counters_right.len());
    assert_match_ids(&result);
    assert_eq!(batch_size, result.left_iris_requests.code.len());
    assert_eq!(batch_size, result.right_iris_requests.code.len());
    assert!(result.deleted_ids.is_empty());
    assert_eq!(batch_size, result.matched_batch_request_ids.len());
    assert_eq!(batch_size, result.successful_reauths.len());
    assert!(result.reauth_target_indices.is_empty());
    assert!(result.reauth_or_rule_used.is_empty());
    assert!(result.modifications.is_empty());
    assert_eq!(batch_size, result.actor_data.0.len());

    // --- Reauth ---

    let batch_1 = BatchQuery {
        request_types: vec![REAUTH_MESSAGE_TYPE.to_string(); batch_size],

        // Map the request ID to the inserted index.
        reauth_target_indices: izip!(&batch_0.request_ids, &inserted_indices)
            .map(|(req_id, inserted_index)| (req_id.clone(), *inserted_index))
            .collect(),
        reauth_use_or_rule: batch_0
            .request_ids
            .iter()
            .map(|req_id| (req_id.clone(), false))
            .collect(),

        ..batch_0.clone()
    };

    let failed_request_i = 1;
    let all_results = parallelize((0..N_PARTIES).map(|party_i| {
        // Mess with the shares to make one request fail.
        let mut shares = irises[party_i].clone();
        shares[failed_request_i].0 = GaloisRingSharedIris::dummy_for_party(party_i);

        let batch = batch_of_party(&batch_1, &shares);
        let mut handle = handles[party_i].clone();
        async move { handle.submit_batch_query(batch).await.await }
    }))
    .await?;

    let result = assert_all_equal(all_results);
    assert_eq!(
        result.successful_reauths,
        (0..batch_size).map(|i| i != failed_request_i).collect_vec()
    );

    // --- Rejected Uniqueness ---

    let batch_2 = batch_0;

    let all_results = parallelize((0..N_PARTIES).map(|party_i| {
        let batch = batch_of_party(&batch_2, &irises[party_i]);
        let mut handle = handles[party_i].clone();
        async move { handle.submit_batch_query(batch).await.await }
    }))
    .await?;
    let result = assert_all_equal(all_results);

    assert_eq!(
        result.match_ids.iter().map(|ids| ids[0]).collect_vec(),
        inserted_indices,
    );
    assert_eq!(result.merged_results, inserted_indices);
    assert_eq!(result.matches, vec![true; batch_size]);
    assert_match_ids(&result);

    tokio::time::sleep(Duration::from_millis(1100)).await;
    Ok(())
}

use super::test_utils::batch_of_party;

fn assert_all_equal(mut all_results: Vec<ServerJobResult>) -> ServerJobResult {
    // Ignore the actual secret shares because they are different for each party.
    for i in 1..all_results.len() {
        all_results[i].left_iris_requests = all_results[0].left_iris_requests.clone();
        all_results[i].right_iris_requests = all_results[0].right_iris_requests.clone();

        assert_eq!(
            all_results[i].identity_update_shares.len(),
            all_results[0].identity_update_shares.len(),
            "All parties must agree on the identity update shares"
        );
        all_results[i].identity_update_shares = all_results[0].identity_update_shares.clone();
    }

    assert!(
        all_results.iter().all_equal(),
        "All parties must agree on the results"
    );
    all_results[0].clone()
}

fn assert_match_ids(results: &ServerJobResult) {
    for (is_match, matches_both, matches_left, matches_right, count_left, count_right) in izip!(
        &results.matches,
        &results.match_ids,
        &results.partial_match_ids_left,
        &results.partial_match_ids_right,
        &results.partial_match_counters_left,
        &results.partial_match_counters_right,
    ) {
        assert_eq!(
            *is_match,
            matches_both.is_empty().not(),
            "Matches must have some matched IDs"
        );
        assert!(
            matches_both
                .iter()
                .all(|id| matches_left.contains(id) && matches_right.contains(id)),
            "Matched IDs must be repeated in left and rights lists"
        );
        assert!(
            matches_left.len() <= *count_left,
            "Partial counts must be consistent"
        );
        assert!(
            matches_right.len() <= *count_right,
            "Partial counts must be consistent"
        );
    }
}
