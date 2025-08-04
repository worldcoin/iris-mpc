use super::constants::COUNT_OF_PARTIES;
use crate::utils::{GaloisRingSharedIrisPair, HawkConfigs, IrisCodePair};
use eyre::Result;
use iris_mpc_common::{
    config::Config,
    iris_db::iris::IrisCode,
    postgres::{AccessMode, PostgresClient},
    IrisSerialId, IrisVersionId,
};
use iris_mpc_cpu::{
    genesis::plaintext::{
        run_plaintext_genesis, GenesisArgs, GenesisConfig, GenesisDstDbState, GenesisSrcDbState,
        GenesisState, PersistentState,
    },
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{graph::graph_store::GraphPg as GraphStore, GraphMem},
    protocol::shared_iris::GaloisRingSharedIris,
};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

/// represents a party in the MPC computation
pub struct MpcNode {
    // databases
    pub gpu_iris_store: IrisStore,
    pub cpu_iris_store: IrisStore,
    pub graph_store: GraphStore<PlaintextStore>,

    // inputs
    pub config: Config,
    pub genesis_args: GenesisArgs,
    pub rng_state: u64,
}

impl MpcNode {
    pub async fn new(config: Config, genesis_args: GenesisArgs, rng_state: u64) -> Self {
        let cpu_client = PostgresClient::new(
            &config.get_db_url(),
            &config.get_db_schema(config.hnsw_schema_name_suffix()),
            AccessMode::ReadWrite,
        )
        .await
        .unwrap();

        let gpu_client = PostgresClient::new(
            &config.get_db_url(),
            &config.get_db_schema(config.gpu_schema_name_suffix()),
            AccessMode::ReadWrite,
        )
        .await
        .unwrap();

        cpu_client.migrate().await;
        gpu_client.migrate().await;

        Self {
            gpu_iris_store: IrisStore::new(&gpu_client).await.unwrap(),
            cpu_iris_store: IrisStore::new(&cpu_client).await.unwrap(),
            graph_store: GraphStore::new(&cpu_client).await.unwrap(),
            config,
            genesis_args,
            rng_state,
        }
    }

    pub async fn clear_all_tables(&self) -> Result<()> {
        // only the cpu database uses the graph store.
        self.graph_store.clear_hawk_graph_tables().await?;

        // delete irises
        self.gpu_iris_store.rollback(0).await?;
        self.cpu_iris_store.rollback(0).await?;

        // clear modifications tables
        let mut tx = self.cpu_iris_store.tx().await?;
        self.cpu_iris_store
            .clear_modifications_table(&mut tx)
            .await?;
        tx.commit().await?;

        let mut tx = self.gpu_iris_store.tx().await?;
        self.gpu_iris_store
            .clear_modifications_table(&mut tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Adds arbitrary irises to the database. The iris ID will be the new
    /// number of entries after the insertion
    pub async fn insert_gpu_iris_store(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        let starting_len = self.gpu_iris_store.count_irises().await?;

        let mut tx = self.gpu_iris_store.tx().await?;
        // use the idx as the id field
        let mut to_insert = vec![];
        for (idx, shares) in shares.iter().enumerate() {
            to_insert.push(StoredIrisRef {
                // warning: id should be >= 1
                id: (starting_len + idx + 1) as _,
                left_code: &shares.0.code.coefs,
                left_mask: &shares.0.mask.coefs,
                right_code: &shares.1.code.coefs,
                right_mask: &shares.1.mask.coefs,
            })
        }

        self.gpu_iris_store
            .insert_irises(&mut tx, &to_insert)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn init_iris_stores(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        self.clear_all_tables().await?;
        self.insert_gpu_iris_store(shares).await?;
        Ok(())
    }

    /// takes the plaintext irises and does the following
    /// 1. uses them to run an in-memory version of Genesis and return the result
    /// 2. converts them into secret-shared form and stores them in the GPU iris database, in
    ///    preparation for a test run of Genesis.
    pub async fn setup_from_plaintext_irises(
        &self,
        pairs: &[IrisCodePair],
    ) -> Result<GenesisState> {
        let genesis_input = get_genesis_input(pairs);

        let genesis_config = GenesisConfig {
            hnsw_M: self.config.hnsw_param_M,
            hnsw_ef_constr: self.config.hnsw_param_ef_constr,
            hnsw_ef_search: self.config.hnsw_param_ef_search,
            hawk_prf_key: Some(self.rng_state),
        };

        let genesis_state =
            construct_initial_genesis_state(genesis_config, self.genesis_args, genesis_input);
        let expected_genesis_state = run_plaintext_genesis(genesis_state).await?;

        let shares = encode_plaintext_iris_for_party(pairs, self.rng_state, self.config.party_id);
        self.init_iris_stores(shares.as_slice()).await?;
        Ok(expected_genesis_state)
    }
}

/// Encapsulates API pointers to set of network databases.
pub struct MpcNodes {
    /// Pointer to set of network node db providers.
    nodes: [MpcNode; COUNT_OF_PARTIES],
}

/// Constructor.
impl MpcNodes {
    pub async fn new(config: &HawkConfigs, genesis_args: GenesisArgs, rng_state: u64) -> Self {
        Self {
            nodes: [
                MpcNode::new(config[0].clone(), genesis_args, rng_state).await,
                MpcNode::new(config[1].clone(), genesis_args, rng_state).await,
                MpcNode::new(config[2].clone(), genesis_args, rng_state).await,
            ],
        }
    }

    /// Returns an iterator over references to the node database providers.
    pub fn iter(&self) -> impl Iterator<Item = &MpcNode> {
        self.nodes.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = MpcNode> {
        self.nodes.into_iter()
    }
}

fn construct_initial_genesis_state(
    genesis_config: GenesisConfig,
    genesis_args: GenesisArgs,
    input: HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)>,
) -> GenesisState {
    GenesisState {
        src_db: GenesisSrcDbState {
            irises: input,
            modifications: (),
        },
        dst_db: GenesisDstDbState {
            irises: HashMap::new(),
            graphs: [GraphMem::new(), GraphMem::new()],
            persistent_state: PersistentState {
                last_indexed_iris_id: None,
                last_indexed_modification_id: None,
            },
        },
        config: genesis_config,
        args: genesis_args,
        s3_deletions: Vec::new(),
    }
}

fn get_genesis_input(
    pairs: &[IrisCodePair],
) -> HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)> {
    let mut r = HashMap::new();
    for (idx, (left, right)) in pairs.iter().enumerate() {
        r.insert(idx as _, (0, left.clone(), right.clone()));
    }
    r
}

fn encode_plaintext_iris_for_party(
    pairs: &[IrisCodePair],
    rng_state: u64,
    party_idx: usize,
) -> Vec<GaloisRingSharedIrisPair> {
    pairs
        .iter()
        .map(|code_pair| {
            // Set RNG for each pair to match shares_encoding.rs behavior
            let mut shares_seed = StdRng::seed_from_u64(rng_state);

            // Set MPC participant specific Iris shares from Iris code + entropy.
            let (code_l, code_r) = code_pair;
            let shares_l =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_l.to_owned());
            let shares_r =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_r.to_owned());

            (shares_l[party_idx].clone(), shares_r[party_idx].clone())
        })
        .collect()
}
