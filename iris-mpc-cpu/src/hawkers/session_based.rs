use super::galois_store::GaloisRingPoint;
use crate::{
    database_generators::{generate_galois_iris_shares, GaloisRingSharedIris},
    execution::session::Session,
    hawkers::plaintext_store::PointId,
};
use hawk_pack::{hnsw_db::HawkSearcher, GraphStore, VectorStore};
use iris_mpc_common::iris_db::iris::IrisCode;
use rand::{CryptoRng, RngCore};

#[derive(Debug, Default, Clone)]
pub struct PlayerStorage {
    points: Vec<GaloisRingPoint>,
}

type PlayerStorageEnsemble = (PlayerStorage, PlayerStorage, PlayerStorage);

impl PlayerStorage {
    fn insert(&mut self, query: &PointId) -> PointId {
        // The query is now accepted in the store. It keeps the same ID.
        *query
    }
}

impl PlayerStorage {
    pub fn new_with_shared_db(data: Vec<GaloisRingSharedIris>) -> Self {
        let points: Vec<GaloisRingPoint> = data
            .into_iter()
            .map(|d| GaloisRingPoint { data: d })
            .collect();
        PlayerStorage { points }
    }

    pub fn prepare_query(&mut self, raw_query: GaloisRingSharedIris) -> PointId {
        self.points.push(GaloisRingPoint { data: raw_query });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }
}

#[derive(Clone)]
pub struct SessionBasedProtocol {
    pub player_storage: PlayerStorage,
    session:            Session,
}

impl std::fmt::Debug for SessionBasedProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.player_storage.fmt(f)
    }
}

pub fn storage_setup_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    database: Vec<IrisCode>,
) -> eyre::Result<PlayerStorageEnsemble> {
    let mut p0 = Vec::new();
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();

    for iris in database {
        let all_shares = generate_galois_iris_shares(rng, iris);
        p0.push(all_shares[0].clone());
        p1.push(all_shares[1].clone());
        p2.push(all_shares[2].clone());
    }

    let player_0 = PlayerStorage::new_with_shared_db(p0);
    let player_1 = PlayerStorage::new_with_shared_db(p1);
    let player_2 = PlayerStorage::new_with_shared_db(p2);
    Ok((player_0, player_1, player_2))
}

impl SessionBasedProtocol {
    pub fn prepare_query(&mut self, code: GaloisRingSharedIris) -> PointId {
        self.player_storage.prepare_query(code)
    }
    pub fn session_as_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

impl VectorStore for SessionBasedProtocol {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.player_storage.insert(query);
        *query
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }
    /// maybe retrieve session as part of the struct implementing the vector
    /// store?
    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        // let session = self.session_as_mut();
        unimplemented!()
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        unimplemented!()
    }
}

pub async fn session_based_insert<V: VectorStore, G: GraphStore<V>>(
    session: &mut Session,
    searcher: &HawkSearcher,
    vector_store: &mut V,
    graph_store: &mut G,
    query: &V::QueryRef,
    rng: &mut impl RngCore,
) -> eyre::Result<()> {
    let neighbors = searcher
        .search_to_insert(vector_store, graph_store, query)
        .await;
    // Insert the new vector into the store.
    let inserted = vector_store.insert(query).await;
    searcher
        .insert_from_search_results(vector_store, graph_store, rng, inserted, neighbors)
        .await;
    Ok(())
}
pub async fn session_based_match<V: VectorStore, G: GraphStore<V>>(
    session: &mut Session,
    searcher: &HawkSearcher,
    db_vectors: &mut V,
    db_graph: &mut G,
    query: &V::QueryRef,
    rng: &mut impl RngCore,
) -> eyre::Result<bool> {
    let neighbors = searcher
        .search_to_insert(db_vectors, db_graph, &query)
        .await;
    Ok(searcher.is_match(db_vectors, &neighbors).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database_generators::generate_galois_iris_shares,
        hawkers::galois_store::setup_local_store_aby3_players,
    };
    use aes_prng::AesRng;
    use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher};
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_based_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;

        let mut aby3_store = setup_local_store_aby3_players().unwrap();
        let mut aby3_graph = GraphMem::new();
        let db = HawkSearcher::default();

        let queries = (0..database_size)
            .map(|id| {
                aby3_store.prepare_query(generate_galois_iris_shares(
                    &mut rng,
                    cleartext_database[id].clone(),
                ))
            })
            .collect::<Vec<_>>();

        // insert queries
        for query in queries.iter() {
            let neighbors = db
                .search_to_insert(&mut aby3_store, &mut aby3_graph, query)
                .await;
        }
    }
}
