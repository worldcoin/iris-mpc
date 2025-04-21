use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};
use kameo_actors::{message_bus as mbus, DeliveryStrategy};
use {
    super::super::{
        errors::IndexationError,
        messages::{
            OnBeginIndexation, OnBeginIndexationOfBatch, OnBeginIndexationOfBatchItem,
            OnEndIndexation,
        },
        utils::logger,
    },
    super::{BatchGenerator, GraphIndexer, SharesFetcher},
};

// ------------------------------------------------------------------------
// Component state.
// ------------------------------------------------------------------------

// Genesis indexation supervisor.
#[derive(Clone)]
pub struct Supervisor {
    config: Config,
    mbus_ref: ActorRef<mbus::MessageBus>,
}

// Ctor.
impl Supervisor {
    pub fn new(config: Config) -> Self {
        assert!(config.aws.is_some() && config.database.is_some());

        Self {
            config,
            mbus_ref: kameo::spawn(mbus::MessageBus::new(DeliveryStrategy::Guaranteed)),
        }
    }
}

// ------------------------------------------------------------------------
// Component message handlers.
// ------------------------------------------------------------------------

impl Message<OnBeginIndexationOfBatch> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnBeginIndexationOfBatch,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginIndexationOfBatch>(&msg);

        // For each item in batch signal that it is ready to be processing.
        for (idx, serial_id) in msg.iris_serial_ids.iter().enumerate() {
            let msg = OnBeginIndexationOfBatchItem {
                batch_idx: msg.batch_idx,
                batch_item_idx: idx + 1,
                iris_serial_id: *serial_id,
            };
            self.mbus_ref.tell(mbus::Publish(msg)).await.unwrap();
        }
    }
}

impl Message<OnEndIndexation> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnEndIndexation,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnEndIndexation>(&msg);

        ctx.actor_ref().stop_gracefully().await.unwrap();
    }
}

// ------------------------------------------------------------------------
// Component lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for Supervisor {
    // Internal error type.
    type Error = IndexationError;

    /// Name - overrides auto-derived name.
    fn name() -> &'static str {
        "Supervisor"
    }

    /// Lifecycle event handler: on_start.
    ///
    /// # Arguments
    ///
    /// * `actor_ref` - Self referential kameo actor pointer.
    ///
    async fn on_start(&mut self, actor_ref: ActorRef<Self>) -> Result<(), Self::Error> {
        logger::log_lifecycle::<Self>("on_start", None);

        // Register message handlers with message bus.
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnBeginIndexationOfBatch>(),
            ))
            .await
            .unwrap();
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnEndIndexation>(),
            ))
            .await
            .unwrap();

        // Spawn components.
        kameo::spawn(BatchGenerator::new(
            self.config.clone(),
            self.mbus_ref.clone(),
        ));
        kameo::spawn(SharesFetcher::new(
            self.config.clone(),
            self.mbus_ref.clone(),
        ));
        kameo::spawn(GraphIndexer::new(
            self.config.clone(),
            self.mbus_ref.clone(),
        ));

        // Signal start.
        self.mbus_ref
            .tell(mbus::Publish(OnBeginIndexation))
            .await
            .unwrap();

        Ok(())
    }
}
