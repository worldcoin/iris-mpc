use super::super::{
    errors::IndexationError,
    messages::{
        OnBeginIndexation, OnBeginIndexationOfBatch, OnEndIndexation, OnEndIndexationOfBatch,
    },
    types::IrisSerialId,
    utils::{self, fetcher, logger},
};
use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};
use kameo_actors::message_bus as mbus;
use std::{iter::Peekable, ops::Range};

// ------------------------------------------------------------------------
// Component state.
// ------------------------------------------------------------------------

// Generates batches of Iris identifiers for processing.
#[allow(dead_code)]
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // System configuration information.
    config: Config,

    // Iterator over range of Iris serial identifiers to be indexed.
    indexation_range_iter: Peekable<Range<IrisSerialId>>,

    // Set of Iris serial identifiers to exclude from indexing.
    indexation_exclusions: Vec<IrisSerialId>,

    // Reference to message bus mediating intra-actor communications.
    mbus_ref: ActorRef<mbus::MessageBus>,
}

// Constructor.
impl BatchGenerator {
    pub fn new(config: Config, mbus_ref: ActorRef<mbus::MessageBus>) -> Self {
        Self {
            batch_count: 0,
            config,
            indexation_exclusions: vec![],
            indexation_range_iter: (0..0).peekable(),
            mbus_ref,
        }
    }
}

// Initializer.
impl BatchGenerator {
    async fn init(&mut self) -> Result<(), IndexationError> {
        // Set indexation exclusions.
        self.indexation_exclusions = fetcher::fetch_iris_deletions(&self.config).await.unwrap();

        // Set indexation range.
        let store = utils::pgres::get_store_instance(&self.config).await;
        let height_of_protocol = fetcher::fetch_height_of_protocol(&store).await?;
        let height_of_indexed = fetcher::fetch_height_of_indexed(&store).await?;
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
// Component methods.
// ------------------------------------------------------------------------

impl BatchGenerator {
    // Processes an indexation step.
    async fn do_indexation_step(&mut self) {
        // Build a batch.
        let mut batch = Vec::<IrisSerialId>::new();
        while self.indexation_range_iter.peek().is_some()
            && batch.len() < self.config.max_batch_size
        {
            // Set next id.
            let next_id = self.indexation_range_iter.by_ref().next().unwrap();

            // Skip exclusions.
            if !self.indexation_exclusions.contains(&next_id) {
                batch.push(next_id);
            } else {
                logger::log_info::<Self>(
                    format!("Excluding deletion :: serial-id={}", next_id).as_str(),
                    None,
                );
            }
        }

        if batch.is_empty() {
            // End of indexation.
            let msg = OnEndIndexation {
                batch_count: self.batch_count,
            };
            self.mbus_ref.tell(mbus::Publish(msg)).await.unwrap();
        } else {
            // New batch.
            self.batch_count += 1;
            let msg = OnBeginIndexationOfBatch {
                batch_idx: self.batch_count,
                batch_size: batch.len(),
                iris_serial_ids: batch,
            };
            self.mbus_ref.tell(mbus::Publish(msg)).await.unwrap();
        }
    }
}

// ------------------------------------------------------------------------
// Component message handlers.
// ------------------------------------------------------------------------

impl Message<OnBeginIndexation> for BatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnBeginIndexation,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginIndexation>(&msg);

        // Crank indexation step.
        self.do_indexation_step().await;
    }
}

impl Message<OnEndIndexationOfBatch> for BatchGenerator {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnEndIndexationOfBatch,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnEndIndexationOfBatch>(&msg);

        // Crank indexation step.
        self.do_indexation_step().await;
    }
}

// ------------------------------------------------------------------------
// Component lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for BatchGenerator {
    // Internal error type.
    type Error = IndexationError;

    /// Name - overrides auto-derived name.
    fn name() -> &'static str {
        "BatchGenerator"
    }

    // Lifecycle event handler: on_start.
    async fn on_start(&mut self, actor_ref: ActorRef<Self>) -> Result<(), Self::Error> {
        logger::log_lifecycle::<Self>("on_start", None);

        // Register message handlers.
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnBeginIndexation>(),
            ))
            .await
            .unwrap();
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnEndIndexationOfBatch>(),
            ))
            .await
            .unwrap();

        // Intitialise state.
        self.init().await?;

        Ok(())
    }
}
