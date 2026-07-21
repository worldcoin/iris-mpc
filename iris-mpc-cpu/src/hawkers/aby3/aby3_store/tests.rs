use std::collections::HashMap;

use super::*;
use crate::{
    execution::{
        hawk_main::{
            iris_worker::{cache_iris, cache_irises},
            scheduler::parallelize,
        },
        session::SessionHandles,
    },
    hawkers::{
        aby3::test_utils::{
            eval_vector_distance, get_owner_index, lazy_random_setup,
            setup_local_store_aby3_players, shared_random_setup,
        },
        plaintext_store::PlaintextStore,
    },
    hnsw::{
        graph::graph_diff::{
            self,
            explicit::{ExplicitNeighborhoodDiffer, SortBy},
            node_equiv::ensure_node_equivalence,
        },
        GraphMem, HnswSearcher, SortedNeighborhood, LINEAR_SCAN_MAX_GRAPH_LAYER,
    },
    network::mpc::NetworkType,
    protocol::shared_iris::GaloisRingSharedIris,
};
use aes_prng::AesRng;
use iris_mpc_common::iris_db::db::IrisDB;
use itertools::{izip, Itertools};
use rand::SeedableRng;
use tokio::task::JoinSet;
use tracing::{info_span, Instrument};
use tracing_test::traced_test;

#[tokio::test(flavor = "multi_thread")]
async fn test_gr_hnsw() -> Result<()> {
    let mut rng = AesRng::seed_from_u64(0_u64);
    let database_size = 10;
    let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
    let shared_irises: Vec<_> = cleartext_database
        .iter()
        .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
        .collect();

    let stores = setup_local_store_aby3_players(NetworkType::Local).await?;

    let mut jobs = JoinSet::new();
    for store in stores.iter() {
        let player_index = get_owner_index(store).await?;
        let irises: Vec<ArcIris> = (0..database_size)
            .map(|id| Arc::new(shared_irises[id][player_index].clone()))
            .collect();
        let queries = cache_irises(store.lock().await.workers.as_ref(), irises).await?;
        let mut rng = rng.clone();
        let store = store.clone();
        jobs.spawn(async move {
            let mut store = store.lock().await;
            let mut aby3_graph = GraphMem::new();
            let db = HnswSearcher::new_with_test_parameters();

            let mut inserted = vec![];
            for query in queries.iter() {
                let insertion_layer = db.gen_layer_rng(&mut rng).unwrap();
                let inserted_vector = db
                    .insert(&mut *store, &mut aby3_graph, query, insertion_layer)
                    .await
                    .unwrap();
                inserted.push(inserted_vector)
            }
            tracing::debug!("FINISHED INSERTING");
            let mut matching_results = vec![];
            for v in inserted.into_iter() {
                let query = store.cache_query_from_store(&v).await.unwrap();
                let neighbors = db
                    .search(&mut *store, &aby3_graph, &query, 1)
                    .await
                    .unwrap();
                tracing::debug!("Finished checking query");
                matching_results.push(db.is_match(&mut *store, &[neighbors]).await.unwrap())
            }
            matching_results
        });
    }
    let matching_results = jobs.join_all().await;
    for (party_id, party_results) in matching_results.iter().enumerate() {
        for (index, result) in party_results.iter().enumerate() {
            assert!(
                *result,
                "Failed at index {:?} for party {:?}",
                index, party_id
            );
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_gr_premade_hnsw() -> Result<()> {
    let mut rng = AesRng::seed_from_u64(0_u64);
    let database_size = 10;
    let network_t = NetworkType::Local;
    let (mut cleartext_data, secret_data) =
        lazy_random_setup(&mut rng, database_size, network_t.clone()).await?;

    let mut rng = AesRng::seed_from_u64(0_u64);
    let mut vector_graph_stores = shared_random_setup(&mut rng, database_size, network_t).await?;

    for ((v_from_scratch, _), (premade_v, _)) in vector_graph_stores.iter().zip(secret_data.iter())
    {
        let v_from_scratch = v_from_scratch.lock().await;
        let premade_v = premade_v.lock().await;
        assert_eq!(
            v_from_scratch.checksum().await,
            premade_v.checksum().await,
            "Registry checksum mismatch: from-scratch vs premade"
        );
    }
    let hawk_searcher = HnswSearcher::new_with_test_parameters();

    for i in 0..database_size {
        let vector_id = VectorId::from_0_index(i as u32);
        let query = cleartext_data
            .0
            .storage
            .get_vector(&vector_id)
            .unwrap()
            .clone();
        let cleartext_neighbors = hawk_searcher
            .search(&mut cleartext_data.0, &cleartext_data.1, &query, 1)
            .await?;
        assert!(
            hawk_searcher
                .is_match(&mut cleartext_data.0, &[cleartext_neighbors])
                .await?,
        );

        let mut jobs = JoinSet::new();
        for (v, g) in vector_graph_stores.iter_mut() {
            let hawk_searcher = hawk_searcher.clone();
            let v_lock = v.lock().await;
            let g = g.clone();
            let q = v_lock.cache_query_from_store(&vector_id).await.unwrap();
            drop(v_lock);
            let v = v.clone();
            jobs.spawn(async move {
                let mut v_lock = v.lock().await;
                let secret_neighbors: SortedNeighborhood<_> =
                    hawk_searcher.search(&mut *v_lock, &g, &q, 1).await.unwrap();

                hawk_searcher
                    .is_match(&mut *v_lock, &[secret_neighbors])
                    .await
            });
        }
        let scratch_results = jobs.join_all().await;

        let mut jobs = JoinSet::new();
        for (v, g) in secret_data.iter() {
            let hawk_searcher = hawk_searcher.clone();
            let v = v.clone();
            let g = g.clone();
            jobs.spawn(async move {
                let mut v_lock = v.lock().await;
                let query = v_lock.cache_query_from_store(&vector_id).await.unwrap();
                let secret_neighbors: SortedNeighborhood<_> = hawk_searcher
                    .search(&mut *v_lock, &g, &query, 1)
                    .await
                    .unwrap();

                hawk_searcher
                    .is_match(&mut *v_lock, &[secret_neighbors])
                    .await
            });
        }
        let premade_results = jobs.join_all().await;

        for (premade_res, scratch_res) in izip!(scratch_results, premade_results) {
            assert!(premade_res?);
            assert!(scratch_res?);
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_gr_aby3_store_plaintext() -> Result<()> {
    let mut rng = AesRng::seed_from_u64(0_u64);
    let db_dim = 4;
    let plaintext_database = IrisDB::new_random_rng(db_dim, &mut rng).db;
    let shared_irises: Vec<_> = plaintext_database
        .iter()
        .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
        .collect();
    let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    // Now do the work for the plaintext store
    let mut plaintext_store = PlaintextStore::<FhdOps>::new();
    let plaintext_preps: Vec<_> = (0..db_dim)
        .map(|id| Arc::new(plaintext_database[id].clone()))
        .collect();
    let mut plaintext_inserts = Vec::new();
    for p in plaintext_preps.iter() {
        plaintext_inserts.push(plaintext_store.insert(p).await);
    }

    // pairs of indices to compare
    let it1 = (0..db_dim).combinations(2);
    let it2 = (0..db_dim).combinations(2);

    let plaintext_queries: Vec<_> = plaintext_database.into_iter().map(Arc::new).collect();

    let mut plain_results = HashMap::new();
    for comb1 in it1.clone() {
        for comb2 in it2.clone() {
            // compute distances in plaintext
            let dist1_plain = plaintext_store
                .eval_distance(&plaintext_queries[comb1[0]], &plaintext_inserts[comb1[1]])
                .await?;
            let dist2_plain = plaintext_store
                .eval_distance(&plaintext_queries[comb2[0]], &plaintext_inserts[comb2[1]])
                .await?;
            let bit = plaintext_store
                .less_than(&dist1_plain, &dist2_plain)
                .await?;
            plain_results.insert((comb1.clone(), comb2.clone()), bit);
        }
    }

    let mut aby3_inserts = vec![];
    for store in local_stores.iter_mut() {
        let player_index = get_owner_index(store).await?;
        let player_irises: Vec<_> = (0..db_dim)
            .map(|id| Arc::new(shared_irises[id][player_index].clone()))
            .collect();
        let mut player_inserts = vec![];
        let mut store_lock = store.lock().await;
        for iris in player_irises.iter() {
            let query = cache_iris(store_lock.workers.as_ref(), iris.clone()).await?;
            let vid = store_lock.insert(&query).await;
            player_inserts.push(vid);
        }
        aby3_inserts.push(player_inserts);
    }

    for comb1 in it1 {
        for comb2 in it2.clone() {
            let mut jobs = JoinSet::new();
            for store in local_stores.iter() {
                let player_index = get_owner_index(store).await?;
                let player_inserts = aby3_inserts[player_index].clone();
                let store = store.clone();
                let index10 = comb1[0];
                let index11 = comb1[1];
                let index20 = comb2[0];
                let index21 = comb2[1];
                jobs.spawn(async move {
                    let mut store = store.lock().await;
                    let dist1_aby3 = eval_vector_distance(
                        &mut store,
                        &player_inserts[index10],
                        &player_inserts[index11],
                    )
                    .await?;
                    let dist2_aby3 = eval_vector_distance(
                        &mut store,
                        &player_inserts[index20],
                        &player_inserts[index21],
                    )
                    .await?;
                    store.less_than(&dist1_aby3, &dist2_aby3).await
                });
            }
            let res = jobs.join_all().await;
            for bit in res {
                assert_eq!(
                    bit?,
                    plain_results[&(comb1.clone(), comb2.clone())],
                    "Failed at combo: {:?}, {:?}",
                    comb1,
                    comb2
                )
            }
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_oblivious_swap() -> Result<()> {
    let list_len = 6_u32;
    let plain_list = (0..list_len)
        .map(|i| (VectorId::from_0_index(i), (i, i)))
        .collect_vec();
    let swap_bits_for_plain = vec![true, false];
    let indices_for_plain = vec![(0, 1), (4, 5)];
    let swap_bits_for_secret = vec![true, false, false];
    let indices_for_secret = vec![(1, 2), (0, 4), (3, 5)];

    let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    let mut jobs = JoinSet::new();
    for store in local_stores.iter_mut() {
        let store = store.clone();
        let swap_bits_for_plain = swap_bits_for_plain.clone();
        let swap_bits_for_secret = swap_bits_for_secret.clone();
        let plain_list = plain_list.clone();
        let indices_for_plain = indices_for_plain.clone();
        let indices_for_secret = indices_for_secret.clone();
        jobs.spawn(async move {
            let mut store_lock = store.lock().await;
            let role = store_lock.session.own_role();
            let swap_bits1 = swap_bits_for_plain
                .iter()
                .map(|b| Share::from_const(Bit::new(*b), role))
                .collect_vec();
            let swap_bits2 = swap_bits_for_secret
                .iter()
                .map(|b| Share::from_const(Bit::new(*b), role))
                .collect_vec();
            let list = plain_list
                .iter()
                .map(|(v, d)| {
                    (
                        v.index(),
                        DistanceShare::new(
                            Share::from_const(d.0, role),
                            Share::from_const(d.1, role),
                        ),
                    )
                })
                .collect_vec();
            let tmp_list = store_lock
                .oblivious_swap_batch_plain_ids(swap_bits1, &list, &indices_for_plain)
                .await?;
            store_lock
                .oblivious_swap_batch(swap_bits2, &tmp_list, &indices_for_secret)
                .await
        });
    }
    let res = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut expected_list = plain_list.clone();
    expected_list.swap(4, 5);
    expected_list.swap(0, 4);
    expected_list.swap(3, 5);

    for (i, exp) in expected_list.iter().enumerate() {
        let id = (res[0][i].0 + res[1][i].0 + res[2][i].0).get_a().convert();
        assert_eq!(id, exp.0.index());

        let distance = {
            let code_dot = (res[0][i].1.code_dot + res[1][i].1.code_dot + res[2][i].1.code_dot)
                .get_a()
                .convert();
            let mask_dot = (res[0][i].1.mask_dot + res[1][i].1.mask_dot + res[2][i].1.mask_dot)
                .get_a()
                .convert();
            (code_dot, mask_dot)
        };
        assert_eq!(distance, exp.1);
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_oblivious_min() -> Result<()> {
    let list_len = 6_u32;
    let mut plain_list = (0..list_len).map(|i| (i, 1)).collect_vec();
    // place the smallest distance at index 3
    plain_list.swap(5, 3);

    let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    let mut jobs = JoinSet::new();
    for store in local_stores.iter_mut() {
        let store = store.clone();
        let plain_list = plain_list.clone();
        jobs.spawn(async move {
            let mut store_lock = store.lock().await;
            let role = store_lock.session.own_role();
            let list = plain_list
                .iter()
                .map(|(code_dist, mask_dist)| {
                    DistanceShare::new(
                        Share::from_const(*code_dist, role),
                        Share::from_const(*mask_dist, role),
                    )
                })
                .collect_vec();
            store_lock.oblivious_min_distance(&list).await
        });
    }
    let res = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let expected = plain_list
        .into_iter()
        .min_by(|a, b| (b.0 * a.1).cmp(&(a.0 * b.1)))
        .unwrap();

    let distance = {
        let code_dot = (res[0].code_dot + res[1].code_dot + res[2].code_dot)
            .get_a()
            .convert();
        let mask_dot = (res[0].mask_dot + res[1].mask_dot + res[2].mask_dot)
            .get_a()
            .convert();
        (code_dot, mask_dot)
    };
    assert_eq!(distance, expected);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_oblivious_argmin() -> Result<()> {
    let list_len = 6_u32;
    let mut plain_list = (0..list_len).map(|i| (i, (i, 1))).collect_vec();
    // place the smallest distance at index 3
    plain_list.swap(5, 3);

    let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    let mut jobs = JoinSet::new();
    for store in local_stores.iter_mut() {
        let store = store.clone();
        let plain_list = plain_list.clone();
        jobs.spawn(async move {
            let mut store_lock = store.lock().await;
            let role = store_lock.session.own_role();
            let list = plain_list
                .iter()
                .map(|(id, (code_dist, mask_dist))| {
                    (
                        VectorId::from_serial_id(*id),
                        DistanceShare::new(
                            Share::from_const(*code_dist, role),
                            Share::from_const(*mask_dist, role),
                        ),
                    )
                })
                .collect_vec();
            store_lock.get_argmin_distance(&list).await
        });
    }
    let res = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let expected = plain_list
        .into_iter()
        .min_by(|(_, a), (_, b)| (b.0 * a.1).cmp(&(a.0 * b.1)))
        .unwrap();

    let distance = {
        let id = res[0].0;
        assert_eq!(id, res[1].0);
        assert_eq!(id, res[2].0);

        let (id, dist) = res
            .into_iter()
            .reduce(|(acc_id, acc_d), (_, a_d)| {
                let code_dist = acc_d.code_dot + a_d.code_dot;
                let mask_dist = acc_d.mask_dot + a_d.mask_dot;
                (acc_id, DistanceShare::new(code_dist, mask_dist))
            })
            .unwrap();
        let code_dot = dist.code_dot.get_a().convert();
        let mask_dot = dist.mask_dot.get_a().convert();

        (id.serial_id(), (code_dot, mask_dot))
    };
    assert_eq!(distance, expected);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_oblivious_min_batch() -> Result<()> {
    let list_len = 6_u32;
    let num_lists = 3;
    // create 3 lists of length 6
    // [[(1,1), (2,1), (3,1), (4,1), (6,1), (5,1)],
    // [(7,1), (8,1), (9,1), (12,1), (10,1), (11,1)],
    // [(13,1), (14,1), (18,1), (15,1), (16,1), (17,1)]]
    let mut flat_list = (1..=(list_len * num_lists)).map(|i| (i, 1)).collect_vec();
    flat_list.swap(5, 4);
    flat_list.swap(11, 9);
    flat_list.swap(17, 14);
    // [(1,1), (7,1), (13,1)],
    // [(2,1), (8,1), (14,1)],
    // [(3,1), (9,1), (18,1)],
    // [(4,1), (12,1), (15,1)],
    // [(6,1), (10,1), (16,1)],
    // [(5,1), (11,1), (17,1)]
    let mut plain_list = Vec::with_capacity(list_len as usize);
    for i in 0..list_len {
        let mut slice = Vec::with_capacity(num_lists as usize);
        for j in 0..num_lists {
            slice.push(flat_list[(i + list_len * j) as usize]);
        }
        plain_list.push(slice);
    }

    let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    let mut jobs = JoinSet::new();
    for store in local_stores.iter_mut() {
        let store = store.clone();
        let plain_list = plain_list.clone();
        jobs.spawn(async move {
            let mut store_lock = store.lock().await;
            let role = store_lock.session.own_role();
            let list = plain_list
                .iter()
                .map(|sub_list| {
                    sub_list
                        .iter()
                        .map(|(code_dist, mask_dist)| {
                            DistanceShare::new(
                                Share::from_const(*code_dist, role),
                                Share::from_const(*mask_dist, role),
                            )
                        })
                        .collect_vec()
                })
                .collect_vec();
            store_lock.oblivious_min_distance_batch(list).await
        });
    }
    let res = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let expected = flat_list
        .chunks_exact(list_len as usize)
        .map(|sublist| {
            sublist
                .iter()
                .min_by(|a, b| (b.0 * a.1).cmp(&(a.0 * b.1)))
                .unwrap()
        })
        .collect_vec();

    for (i, exp) in expected.into_iter().enumerate() {
        let distance = {
            let code_dot = (res[0][i].code_dot + res[1][i].code_dot + res[2][i].code_dot)
                .get_a()
                .convert();
            let mask_dot = (res[0][i].mask_dot + res[1][i].mask_dot + res[2][i].mask_dot)
                .get_a()
                .convert();
            (code_dot, mask_dot)
        };
        assert_eq!(distance, *exp);
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_gr_aby3_store_plaintext_batch() -> Result<()> {
    let mut rng = AesRng::seed_from_u64(0_u64);
    let db_size = 10;
    let plaintext_database = IrisDB::new_random_rng(db_size, &mut rng).db;
    let shared_irises: Vec<_> = plaintext_database
        .iter()
        .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
        .collect();
    let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    // Now do the work for the plaintext store
    let mut plaintext_store = PlaintextStore::<FhdOps>::new();
    let plaintext_preps: Vec<_> = (0..db_size)
        .map(|id| Arc::new(plaintext_database[id].clone()))
        .collect();
    let mut plaintext_inserts = Vec::with_capacity(db_size);
    for p in plaintext_preps.iter() {
        plaintext_inserts.push(plaintext_store.insert(p).await);
    }

    // compute distances in plaintext
    let dist1_plain = plaintext_store
        .eval_distance_batch(&plaintext_preps[0], &plaintext_inserts)
        .await?;
    let dist2_plain = plaintext_store
        .eval_distance_batch(&plaintext_preps[1], &plaintext_inserts)
        .await?;
    let dist_plain = dist1_plain
        .into_iter()
        .zip(dist2_plain.into_iter())
        .collect::<Vec<_>>();
    let bits_plain = plaintext_store.less_than_batch(&dist_plain).await?;

    let mut aby3_inserts = vec![];
    let mut queries = vec![];
    for store in local_stores.iter_mut() {
        let player_index = get_owner_index(store).await?;
        let irises: Vec<ArcIris> = (0..db_size)
            .map(|id| Arc::new(shared_irises[id][player_index].clone()))
            .collect();
        let mut store_lock = store.lock().await;
        let player_preps = cache_irises(store_lock.workers.as_ref(), irises).await?;
        queries.push(player_preps.clone());
        let mut player_inserts = vec![];
        for p in player_preps.iter() {
            player_inserts.push(store_lock.insert(p).await);
        }
        aby3_inserts.push(player_inserts);
    }

    let mut jobs = JoinSet::new();
    for store in local_stores.iter() {
        let player_index = get_owner_index(store).await?;
        let player_inserts = aby3_inserts[player_index].clone();
        let player_preps = queries[player_index].clone();
        let store = store.clone();
        jobs.spawn(async move {
            let mut store_lock = store.lock().await;
            let dist1_aby3 = store_lock
                .eval_distance_batch(&player_preps[0], &player_inserts)
                .await?;
            let dist2_aby3 = store_lock
                .eval_distance_batch(&player_preps[1], &player_inserts)
                .await?;
            let dist_aby3 = dist1_aby3
                .into_iter()
                .zip(dist2_aby3.into_iter())
                .collect::<Vec<_>>();
            store_lock.less_than_batch(&dist_aby3).await
        });
    }
    let bits_aby3 = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    assert_eq!(bits_aby3[0], bits_aby3[1]);
    assert_eq!(bits_aby3[0], bits_aby3[2]);
    assert_eq!(bits_aby3[0], bits_plain);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_gr_scratch_hnsw() {
    let mut rng = AesRng::seed_from_u64(0_u64);
    let database_size = 2;
    let searcher = HnswSearcher::new_with_test_parameters();
    let mut vectors_and_graphs = shared_random_setup(&mut rng, database_size, NetworkType::Local)
        .await
        .unwrap();

    for i in 0..database_size {
        let vector_id = VectorId::from_0_index(i as u32);
        let mut jobs = JoinSet::new();
        for (store, graph) in vectors_and_graphs.iter_mut() {
            let graph = graph.clone();
            let searcher = searcher.clone();
            let store_lock = store.lock().await;
            let q = store_lock.cache_query_from_store(&vector_id).await.unwrap();
            drop(store_lock);
            let store = store.clone();
            jobs.spawn(async move {
                let mut store = store.lock().await;
                let secret_neighbors: SortedNeighborhood<_> =
                    searcher.search(&mut *store, &graph, &q, 1).await.unwrap();
                searcher
                    .is_match(&mut *store, &[secret_neighbors])
                    .await
                    .unwrap()
            });
        }
        let res = jobs.join_all().await;
        for (party_index, r) in res.into_iter().enumerate() {
            assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_gr_non_existent_vectors() {
    let mut rng = AesRng::seed_from_u64(0_u64);
    let database_size = 2;
    let vectors_and_graphs = shared_random_setup(&mut rng, database_size, NetworkType::Local)
        .await
        .unwrap();

    let mut tasks = vec![];
    for (store, _graph) in vectors_and_graphs {
        let mut store = store.lock_owned().await;
        tasks.push(async move {
            let none = VectorId::from_0_index(999);
            let a = VectorId::from_0_index(0);
            let b = VectorId::from_0_index(1);

            let queries = store.vectors_as_queries(vec![a, b]).await?;
            let vectors = vec![a, b, none];
            let n_vecs = vectors.len();

            let distances = {
                let mut dist_a = store
                    .eval_distance_batch(&queries[0], &vectors)
                    .await
                    .unwrap();
                let dist_b = store
                    .eval_distance_batch(&queries[1], &vectors)
                    .await
                    .unwrap();
                dist_a.extend(dist_b);
                dist_a
            };

            let is_match = store.is_match_batch(&distances).await.unwrap();
            assert_eq!(
                is_match,
                [vec![true, false, false], vec![false, true, false]].concat(),
                "Vectors should match with themselves and not with the others"
            );

            let distances_to_none = vec![distances[2], distances[2 + n_vecs]];
            let pairs = distances_to_none
                .into_iter()
                .cartesian_product(distances)
                .collect_vec();
            let less_than = store.less_than_batch(&pairs).await.unwrap();

            assert_eq!(
                less_than,
                vec![false; pairs.len()],
                "Nothing is less than a distance to a non-existent vector"
            );

            Ok(())
        });
    }
    parallelize(tasks.into_iter()).await.unwrap();
}

/// Build the same HNSW graph in plaintext and under 3-party MPC using a
/// shared insertion-layer sequence, then assert bit-for-bit equality.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
async fn test_plaintext_vs_mpc_graph_equality() -> Result<()> {
    let mut rng = AesRng::seed_from_u64(0xA1B2C3D4_u64);
    let database_size = 256;
    // LinearScan mirrors HawkActor's searcher (hawk_main.rs). Layer
    // density bumped to 4 so enough nodes roll onto layer 1 to exercise
    // `linear_search_min_distance` — default (M) gives <1 expected entry
    // point at this size and silently skips the branch.
    let mut searcher = HnswSearcher::new_linear_scan(64, 32, 32, LINEAR_SCAN_MAX_GRAPH_LAYER);
    searcher.layer_distribution = crate::hnsw::searcher::LayerDistribution::new_geometric_from_M(4);
    let searcher = searcher;

    let insertion_layers: Vec<usize> = {
        let mut layer_rng = AesRng::from_rng(rng.clone())?;
        (0..database_size)
            .map(|_| searcher.gen_layer_rng(&mut layer_rng))
            .collect::<Result<Vec<_>>>()?
    };

    let cleartext_db = IrisDB::new_random_rng(database_size, &mut rng).db;
    let shared_irises: Vec<[GaloisRingSharedIris; 3]> = cleartext_db
        .iter()
        .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
        .collect();

    let mut plaintext_store = PlaintextStore::<FhdOps>::new();
    let mut plaintext_graph: GraphMem = GraphMem::new();
    let plaintext_span = info_span!("plaintext_insert");
    for (iris, &layer) in cleartext_db.iter().zip(insertion_layers.iter()) {
        let query = Arc::new(iris.clone());
        searcher
            .insert(&mut plaintext_store, &mut plaintext_graph, &query, layer)
            .instrument(plaintext_span.clone())
            .await?;
    }

    let stores = setup_local_store_aby3_players(NetworkType::Local).await?;
    let mut jobs = JoinSet::new();
    for (idx, store) in stores.into_iter().enumerate() {
        let role = get_owner_index(&store).await?;
        let irises: Vec<ArcIris> = shared_irises
            .iter()
            .map(|shares| Arc::new(shares[role].clone()))
            .collect();
        let layers = insertion_layers.clone();
        let searcher = searcher.clone();
        jobs.spawn(async move {
            let mpc_span = info_span!("mpc_insert", id = idx);
            let mut store = store.lock().await;
            let queries = cache_irises(store.workers.as_ref(), irises).await?;
            let mut graph: GraphMem = GraphMem::new();
            for (query, &layer) in queries.iter().zip(layers.iter()) {
                searcher
                    .insert(&mut *store, &mut graph, query, layer)
                    .instrument(mpc_span.clone())
                    .await?;
            }
            Ok::<(usize, GraphMem), eyre::Report>((role, graph))
        });
    }

    let mut mpc_graphs: Vec<Option<GraphMem>> = (0..3).map(|_| None).collect();
    while let Some(res) = jobs.join_next().await {
        let (role, graph) = res??;
        mpc_graphs[role] = Some(graph);
    }

    for (role, mpc_graph) in mpc_graphs.into_iter().enumerate() {
        let mpc_graph = mpc_graph.unwrap_or_else(|| panic!("party {role} did not finish"));
        if let Err(e) = ensure_node_equivalence(&mpc_graph, &plaintext_graph) {
            panic!("graph mismatch: {:?}", e);
        }
        if mpc_graph != plaintext_graph {
            let diff_output = graph_diff::run_diff(
                &mpc_graph,
                &plaintext_graph,
                ExplicitNeighborhoodDiffer::new(SortBy::Index),
            );
            panic!(
                "[Role {}] graphs are not equal. Diff: {}",
                role, diff_output
            );
        }
    }

    // If either assert fails the parameters no longer exercise the
    // LinearScan path (see layer-density comment above).
    assert_eq!(plaintext_graph.get_num_layers(), 2);
    assert!(plaintext_graph.entry_points.len() >= 2);

    Ok(())
}
