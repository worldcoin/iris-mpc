use super::{
    super::utils::fetch_iris_v1_deletions as fetch_iris_v1_deletions_from_s3,
    super::Supervisor,
    super::{
        errors::IndexationError,
        messages::{OnBegin, OnBeginBatch, OnEnd, OnEndBatch},
        utils::logger,
    },
};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisStore;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use std::iter::Peekable;
use std::ops::Range;

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Generates batches of Iris identifiers for processing.
pub struct BatchGenerator {
    // System configuration information.
    config: Config,

    // Iterator over range of Iris serial identifiers to be indexed.
    indexation_range_iter: Peekable<Range<i64>>,

    // Set of Iris serial identifiers to exclude from indexing.
    indexation_exclusions: Vec<i64>,

    // Iris store provider.
    store: Option<IrisStore>,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl BatchGenerator {
    // TODO: move to config.
    const DEFAULT_BATCH_SIZE: usize = 42;

    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        assert!(config.database.is_some());

        Self {
            config,
            indexation_exclusions: vec![],
            indexation_range_iter: (0..0).peekable(),
            store: None,
            supervisor_ref,
        }
    }
}

// Methods.
impl BatchGenerator {
    // Processes an indexation step.
    async fn do_indexation_step(&mut self) {
        // Build a batch.
        let mut batch = Vec::<i64>::new();
        while self.indexation_range_iter.peek().is_some() {
            // Set next id.
            let next_id = self.indexation_range_iter.by_ref().next().unwrap();
            if self.indexation_exclusions.contains(&next_id) {
                logger::log_info::<Self>(
                    format!("Excluding deletion :: serial-id={}", next_id).as_str(),
                    None,
                );
                continue;
            }

            // Extend batch - escape if built.
            batch.push(next_id);
            if batch.len() == Self::DEFAULT_BATCH_SIZE {
                batch.sort();
                break;
            }
        }

        // Signal either new batch or end of indexation.
        if batch.is_empty() {
            let msg = OnEnd;
            self.supervisor_ref.tell(msg).await.unwrap();
        } else {
            let msg = OnBeginBatch { serial_ids: batch };
            self.supervisor_ref.tell(msg).await.unwrap();
        }
    }

    // Fetches set of deletions to be excluded from indexing.
    async fn fetch_deletions_for_exclusion(&mut self) -> Result<Vec<i64>, IndexationError> {
        let mut deletions = [
            self.fetch_deletions_for_exclusion_v1().await?,
            self.fetch_deletions_for_exclusion_v2().await?,
        ]
        .concat();

        deletions.dedup();
        deletions.sort();

        Ok(deletions)
    }

    // Fetches set of V2 deletions from AWS-S3.
    async fn fetch_deletions_for_exclusion_v1(&self) -> Result<Vec<i64>, IndexationError> {
        fetch_iris_v1_deletions_from_s3(&self.config).await
    }

    // Fetches set of V2 deletions from store.
    async fn fetch_deletions_for_exclusion_v2(&mut self) -> Result<Vec<i64>, IndexationError> {
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

    // Fetches height of protocol from store.
    async fn fetch_height_of_protocol(&mut self) -> Result<i64, IndexationError> {
        self.store
            .as_ref()
            .unwrap()
            .count_irises()
            .await
            .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
            .map(|val| val as i64)
    }

    // Fetches height of indexed from store.
    async fn fetch_height_of_indexed(&self) -> Result<i64, IndexationError> {
        // TODO: fetch from store.
        Ok(1)
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
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

impl Message<OnBegin> for BatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, _: OnBegin, _: Context<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message::<Self>("OnBegin", None);

        // Crank next indexation step.
        self.do_indexation_step().await;
    }
}

impl Message<OnEndBatch> for BatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, _: OnEndBatch, _: Context<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message::<Self>("OnEndBatch", None);

        // Crank next indexation step.
        self.do_indexation_step().await;
    }
}

// ------------------------------------------------------------------------
// Actor lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for BatchGenerator {
    // By default mailbox is limited to 1000 messages.
    type Mailbox = BoundedMailbox<Self>;

    /// Actor name - overrides auto-derived name.
    fn name() -> &'static str {
        "BatchGenerator"
    }

    /// Lifecycle event handler: on_start.
    ///
    /// State initialisation hook.
    async fn on_start(&mut self, _: ActorRef<Self>) -> Result<(), BoxError> {
        logger::log_lifecycle::<Self>("on_start", None);

        // Initialise store pointer.
        self.set_store().await?;

        // Initialise indexation exclusions.
        self.indexation_exclusions = self.fetch_deletions_for_exclusion().await?;
        logger::log_info::<Self>(
            format!(
                "Count of deletions for exclusion = {}",
                self.indexation_exclusions.len()
            )
            .as_str(),
            None,
        );

        // Initialise indexation range.
        let height_of_protocol = self.fetch_height_of_protocol().await?;
        let height_of_indexed = self.fetch_height_of_indexed().await?;
        self.indexation_range_iter = (height_of_indexed..height_of_protocol).peekable();
        logger::log_info::<Self>(
            format!(
                "Range of serial-id's to index = {}..{}",
                height_of_indexed, height_of_protocol
            )
            .as_str(),
            None,
        );

        Ok(())
    }
}
