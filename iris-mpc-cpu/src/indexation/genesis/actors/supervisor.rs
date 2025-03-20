use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use {
    super::super::{
        errors::IndexationError,
        messages::{
            OnBeginGraphIndexation, OnBeginIndexation, OnBeginIndexationOfBatch,
            OnBeginIndexationOfBatchItem, OnEndIndexation, OnEndIndexationOfBatch,
            OnFetchIrisShares,
        },
        utils::logger,
    },
    super::{BatchGenerator, GraphIndexer, SharesFetcher},
};

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Genesis indexation supervisor.
#[derive(Clone)]
pub struct Supervisor {
    a1_ref: Option<ActorRef<BatchGenerator>>,
    a2_ref: Option<ActorRef<SharesFetcher>>,
    a3_ref: Option<ActorRef<GraphIndexer>>,
    config: Config,
}

impl Supervisor {
    // Ctor.
    pub fn new(config: Config) -> Self {
        assert!(config.aws.is_some() && config.database.is_some());

        Self {
            a1_ref: None,
            a2_ref: None,
            a3_ref: None,
            config,
        }
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
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

        // Signal to other interested actors.
        // I.E. GraphIndexer.
        self.a3_ref
            .as_ref()
            .unwrap()
            .tell(msg.clone())
            .await
            .unwrap();

        // For each item in batch, signal that it is ready to be processing.
        for (idx, serial_id) in msg.iris_serial_ids.iter().enumerate() {
            let msg = OnBeginIndexationOfBatchItem {
                batch_idx: msg.batch_idx,
                batch_item_idx: idx + 1,
                iris_serial_id: *serial_id,
            };
            self.a2_ref.as_ref().unwrap().tell(msg).await.unwrap();
        }
    }
}

impl Message<OnBeginGraphIndexation> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnBeginGraphIndexation,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginGraphIndexation>(&msg);

        // Signal to other interested actors.
        self.a3_ref
            .as_ref()
            .unwrap()
            .tell(msg.clone())
            .await
            .unwrap();
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

impl Message<OnEndIndexationOfBatch> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnEndIndexationOfBatch,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnEndIndexationOfBatch>(&msg);

        // Signal to other interested actors.
        self.a1_ref
            .as_ref()
            .unwrap()
            .tell(msg.clone())
            .await
            .unwrap();
    }
}

impl Message<OnFetchIrisShares> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnFetchIrisShares,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnFetchIrisShares>(&msg);

        // Signal to other interested actors.
        self.a3_ref.as_ref().unwrap().tell(msg).await.unwrap()
    }
}

// ------------------------------------------------------------------------
// Actor lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for Supervisor {
    // By default mailbox is limited to 1000 messages.
    type Mailbox = BoundedMailbox<Self>;
    type Error = IndexationError;

    /// Actor name - overrides auto-derived name.
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

        // Instantiate associated actors.
        let a1 = BatchGenerator::new(self.config.clone(), actor_ref.clone());
        let a2 = SharesFetcher::new(self.config.clone(), actor_ref.clone());
        let a3 = GraphIndexer::new(self.config.clone(), actor_ref.clone());

        // Spawn associated actors.
        self.a1_ref = Some(kameo::spawn(a1));
        self.a2_ref = Some(kameo::spawn(a2));
        self.a3_ref = Some(kameo::spawn(a3));

        // Signal start.
        self.a1_ref
            .as_ref()
            .unwrap()
            .tell(OnBeginIndexation)
            .await
            .map_err(|_| IndexationError::BeginIndexationError)
            .unwrap();

        Ok(())
    }
}
