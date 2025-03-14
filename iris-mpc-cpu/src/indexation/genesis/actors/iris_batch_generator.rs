use super::{
    super::utils::fetch_iris_v1_deletions as fetch_iris_v1_deletions_from_s3,
    super::Supervisor,
    super::{errors::IndexationError, messages},
};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisStore;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};

// ------------------------------------------------------------------------
// Declaration + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Generates batches of Iris identifiers for processing.
#[derive(Actor)]
pub struct IrisBatchGenerator {
    // System configuration information.
    config: Config,

    // Set of Iris serial identifiers to exclude from indexing.
    deletions_for_exclusion: Option<Vec<i64>>,

    // ID of most recent Iris data indexed.
    height_of_indexed: Option<i64>,

    // ID of most recent Iris data within store.
    height_of_protocol: Option<i64>,

    // Current batch of Iris serial identifiers marked for indexation.
    batch_for_indexation: Option<Vec<i64>>,

    // Size of each processing batch.
    size_of_batch: u32,

    // Iris store provider.
    store: Option<IrisStore>,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl IrisBatchGenerator {
    // TODO: move to config.
    const DEFAULT_BATCH_SIZE: u32 = 42;

    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        assert!(config.database.is_some());

        Self {
            config,
            deletions_for_exclusion: None,
            height_of_indexed: None,
            height_of_protocol: None,
            batch_for_indexation: None,
            size_of_batch: Self::DEFAULT_BATCH_SIZE,
            store: None,
            supervisor_ref,
        }
    }
}

// Methods.
impl IrisBatchGenerator {
    // Processes an indexation step.
    async fn do_indexation_step(&mut self) {
        // Update internal state.
        self.update_state().await;

        // Signal end of indexation if at tip.
        if self.batch_for_indexation.is_none() {
            let msg = messages::OnEnd;
            self.supervisor_ref.tell(msg).await.unwrap();
        // Signal new batch is awaiting indexation.
        } else {
            let msg = messages::OnBeginBatch {
                batch: self.batch_for_indexation.as_ref().unwrap().to_owned(),
            };
            self.supervisor_ref.tell(msg).await.unwrap();
        }
    }

    // Fetches set of deletions that can be excluded from indexing.
    async fn fetch_deletions_for_exclusion(&mut self) -> Result<Vec<i64>, IndexationError> {
        let mut deletions = [
            self.fetch_deletions_for_exclusion_v1().await?,
            self.fetch_deletions_for_exclusion_v2().await?,
        ]
        .concat();

        // TODO remove once mocked service calls are fully implemented.
        deletions.dedup();
        deletions.sort();

        Ok(deletions)
    }

    // Fetches set of V2 deletions that can be excluded from indexing.
    async fn fetch_deletions_for_exclusion_v1(&self) -> Result<Vec<i64>, IndexationError> {
        // TODO: remove use of config.clone.
        fetch_iris_v1_deletions_from_s3(&self.config).await
    }

    // Fetches set of V2 deletions that can be excluded from indexing.
    async fn fetch_deletions_for_exclusion_v2(&mut self) -> Result<Vec<i64>, IndexationError> {
        self.set_store().await.unwrap();
        let deletions = self
            .store
            .as_ref()
            .unwrap()
            .fetch_iris_v2_deletions_by_party_id(self.config.party_id)
            .await
            .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
            .unwrap();

        Ok(deletions)
    }

    // Fetches height of protocol from a store.
    async fn fetch_height_of_protocol(&mut self) -> Result<i64, IndexationError> {
        self.set_store().await.unwrap();
        self.store
            .as_ref()
            .unwrap()
            .count_irises()
            .await
            .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
            .map(|val| val as i64)
    }

    // Fetches height of indexed from a store.
    async fn fetch_height_of_indexed(&self) -> Result<i64, IndexationError> {
        // TODO: fetch from store | keep internal memo.
        Ok(1)
    }

    // Returns next range of Iris id's for processing.
    fn get_next_batch_for_indexation(&mut self) -> Option<Vec<i64>> {
        let count_of_unprocessed =
            self.height_of_protocol.unwrap() - self.height_of_indexed.unwrap();
        if count_of_unprocessed == 0 {
            return None;
        }

        let mut next_batch = Vec::<i64>::new();
        let to_be_indexed = self.height_of_indexed.unwrap()..self.height_of_protocol.unwrap();
        for i in to_be_indexed {
            if self.deletions_for_exclusion.as_ref().unwrap().contains(&i) {
                continue;
            }
            next_batch.push(i);
            if next_batch.len() == self.size_of_batch as usize {
                break;
            }
        }

        Some(next_batch)
    }

    // Sets pointer to remote store.
    async fn set_store(&mut self) -> Result<(), IndexationError> {
        if self.store.is_none() {
            match IrisStore::new_from_config(&self.config).await {
                Ok(store) => {
                    self.store = Some(store);
                }
                Err(_) => return Err(IndexationError::PostgresConnectionError),
            }
        }

        Ok(())
    }

    // Updates state fields as per current indexation progress.
    async fn update_state(&mut self) {
        // Set deletions for exclusion.
        if self.deletions_for_exclusion.is_none() {
            self.deletions_for_exclusion =
                Some(self.fetch_deletions_for_exclusion().await.unwrap());
        }

        // Set current protocol height.
        self.height_of_protocol = Some(self.fetch_height_of_protocol().await.unwrap());

        // Set current indexed height.
        self.height_of_indexed = Some(self.fetch_height_of_indexed().await.unwrap());

        // Set next batch.
        self.batch_for_indexation = self.get_next_batch_for_indexation();
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

// Message: OnIndexationBegin.
impl Message<messages::OnBegin> for IrisBatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnBegin,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        // Initiate first indexation step.
        self.do_indexation_step().await;
    }
}

// Message: OnIndexationOfBatchEnd.
impl Message<messages::OnEndOfBatch> for IrisBatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnEndOfBatch,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        // Initiate next indexation step.
        self.do_indexation_step().await;
    }
}
