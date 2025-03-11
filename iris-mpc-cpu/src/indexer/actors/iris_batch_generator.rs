#[allow(dead_code)]
use super::{
    super::supervisors::IndexFromDbSupervisor as Supervisor,
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

    // ID of most recent Iris data indexed.
    height_of_indexed: Option<i64>,

    // ID of most recent Iris data within store.
    height_of_protocol: Option<i64>,

    // Latest range of Iris identifiers signalled for processing.
    range_for_indexation: Option<(i64, i64)>,

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
            height_of_indexed: None,
            height_of_protocol: None,
            range_for_indexation: None,
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

        // Signal end of indexation if upto tip.
        if self.range_for_indexation.is_none() {
            self.supervisor_ref
                .tell(messages::OnIndexationEnd)
                .await
                .unwrap();
        // Signal next batch to be indexed.
        } else {
            self.supervisor_ref
                .tell(messages::OnBatchIndexationStart {
                    batch_range: self.range_for_indexation.unwrap(),
                })
                .await
                .unwrap();
        }
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
        // TODO: pull latest height from store.
        Ok(0)
    }

    // Returns next range of Iris id's for processing.
    fn get_next_range_for_indexation(&mut self) -> Option<(i64, i64)> {
        let count_of_unprocessed =
            self.height_of_protocol.unwrap() - self.height_of_indexed.unwrap();
        if count_of_unprocessed == 0 {
            None
        } else if count_of_unprocessed < self.size_of_batch as i64 {
            Some((
                self.height_of_indexed.unwrap(),
                self.height_of_indexed.unwrap() + count_of_unprocessed,
            ))
        } else {
            Some((
                self.height_of_indexed.unwrap(),
                self.height_of_indexed.unwrap() + self.size_of_batch as i64,
            ))
        }
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
        // Set current protocol height.
        self.height_of_protocol = Some(self.fetch_height_of_protocol().await.unwrap());

        // Set current indexed height.
        self.height_of_indexed = Some(self.fetch_height_of_indexed().await.unwrap());

        // Set next batch range.
        self.range_for_indexation = self.get_next_range_for_indexation();
    }
}

// ------------------------------------------------------------------------
// Message handlers.
// ------------------------------------------------------------------------

// Message: OnIndexationStart.
impl Message<messages::OnIndexationStart> for IrisBatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnIndexationStart,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        self.do_indexation_step().await;
    }
}

// Message: OnBatchIndexationEnd.
impl Message<messages::OnBatchIndexationEnd> for IrisBatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnBatchIndexationEnd,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        self.do_indexation_step().await;
    }
}
