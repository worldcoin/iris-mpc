use super::{
    super::Supervisor,
    super::{
        errors::IndexationError,
        messages::{OnBegin, OnBeginBatch, OnEnd, OnEndBatch},
        types::IrisSerialId,
        utils::logger,
    },
};
use aws_sdk_s3::{config::Region as S3_Region, Client as S3_CLient};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisStore;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use rand::prelude::*;
use std::iter::Peekable;
use std::ops::Range;

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Generates batches of Iris identifiers for processing.
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // System configuration information.
    config: Config,

    // Iterator over range of Iris serial identifiers to be indexed.
    indexation_range_iter: Peekable<Range<IrisSerialId>>,

    // Set of Iris serial identifiers to exclude from indexing.
    indexation_exclusions: Vec<IrisSerialId>,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl BatchGenerator {
    // TODO: move to config.
    const DEFAULT_BATCH_SIZE: usize = 42;

    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            batch_count: 0,
            config,
            indexation_exclusions: vec![],
            indexation_range_iter: (0..0).peekable(),
            supervisor_ref,
        }
    }
}

// Methods.
impl BatchGenerator {
    // Processes an indexation step.
    async fn do_indexation_step(&mut self) {
        // Build a batch.
        let mut batch = Vec::<IrisSerialId>::new();
        while self.indexation_range_iter.peek().is_some() {
            // Set next id.
            let next_id = self.indexation_range_iter.by_ref().next().unwrap();

            // Skip exclusions.
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

        // Increment batch count.
        if batch.is_empty() == false {
            self.batch_count += 1;
        }

        // Signal.
        if batch.is_empty() {
            // End of indexation.
            let msg = OnEnd {
                batch_count: self.batch_count,
            };
            self.supervisor_ref.tell(msg).await.unwrap();
        } else {
            // New batch.
            let msg = OnBeginBatch {
                batch_idx: self.batch_count,
                batch_size: batch.len(),
                iris_serial_ids: batch,
            };
            self.supervisor_ref.tell(msg).await.unwrap();
        }
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

impl Message<OnBegin> for BatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, msg: OnBegin, _: Context<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message::<Self, OnBegin>(&msg);

        // Crank indexation step.
        self.do_indexation_step().await;
    }
}

impl Message<OnEndBatch> for BatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, msg: OnEndBatch, _: Context<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message::<Self, OnEndBatch>(&msg);

        // Crank indexation step.
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

        // Set store client.
        let store = IrisStore::new_from_config(&self.config).await?;

        // Set indexation exclusions.
        self.indexation_exclusions = [
            fetch_iris_v1_deletions(&self.config).await.unwrap(),
            fetch_iris_v2_deletions(&store, &self.config).await.unwrap(),
        ]
        .concat();

        // Set indexation range.
        let height_of_protocol = fetch_height_of_protocol(&store).await?;
        let height_of_indexed = fetch_height_of_indexed(&store).await?;
        self.indexation_range_iter = (height_of_indexed..height_of_protocol + 1).peekable();

        // Emit log entries.
        logger::log_info::<Self>(
            format!(
                "Range of serial-id's to index = {}..{}",
                height_of_indexed, height_of_protocol
            )
            .as_str(),
            None,
        );
        logger::log_info::<Self>(
            format!(
                "Deletions for exclusion = {}",
                self.indexation_exclusions.len()
            )
            .as_str(),
            None,
        );

        Ok(())
    }
}

// ------------------------------------------------------------------------
// Helper methods.
// ------------------------------------------------------------------------

/// Fetches height of indexed from store.
///
/// # Arguments
///
/// * `store` - Iris store provider.
///
/// # Returns
///
/// Height of indexed Iris's.
///
async fn fetch_height_of_indexed(_: &IrisStore) -> Result<IrisSerialId, IndexationError> {
    // TODO: fetch from store.
    Ok(1)
}

/// Fetches height of protocol from store.
///
/// # Arguments
///
/// * `store` - Iris store provider.
///
/// # Returns
///
/// Height of stored Iris's.
///
async fn fetch_height_of_protocol(store: &IrisStore) -> Result<IrisSerialId, IndexationError> {
    store
        .count_irises()
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
        .map(|val| val as IrisSerialId)
}

/// Fetches V1 serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - System configuration information.
///
/// # Returns
///
/// A set of Iris V1 serial identifiers marked as deleted.
///
async fn fetch_iris_v1_deletions(config: &Config) -> Result<Vec<IrisSerialId>, IndexationError> {
    // Destructure AWS configuration settings.
    let aws_endpoint = config
        .aws
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?
        .endpoint
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?;
    let aws_region = config
        .aws
        .as_ref()
        .unwrap()
        .region
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?;

    // Set AWS S3 client.
    let aws_config = aws_config::from_env()
        .region(S3_Region::new(aws_region.clone()))
        .load()
        .await;
    let s3_cfg = aws_sdk_s3::config::Builder::from(&aws_config)
        .endpoint_url(aws_endpoint.clone())
        .force_path_style(true)
        .build();
    let _ = S3_CLient::from_conf(s3_cfg);

    // Set AWS S3 response.
    // TODO: test once resource has been deployed

    // TODO: remove temporary code that returns a random set of identifiers.
    let mut rng = rand::thread_rng();
    let mut identifiers: Vec<IrisSerialId> = (1..1000).choose_multiple(&mut rng, 50);
    identifiers.sort();

    Ok(identifiers)
}

/// Fetches V2 serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `store` - Iris store provider.
/// * `config` - System configuration information.
///
/// # Returns
///
/// A set of Iris V2 serial identifiers marked as deleted.
///
async fn fetch_iris_v2_deletions(
    store: &IrisStore,
    config: &Config,
) -> Result<Vec<IrisSerialId>, IndexationError> {
    let deletions = store
        .fetch_iris_v2_deletions_by_party_id(config.party_id)
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
        .unwrap();

    Ok(deletions)
}
