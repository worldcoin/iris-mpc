use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context as Ctx, Message},
    Actor,
};
use {
    super::super::messages::{
        OnBeginGraphIndexation, OnBeginIndexation, OnBeginIndexationOfBatch,
        OnBeginIndexationOfBatchItem, OnEndIndexation, OnEndIndexationOfBatch, OnFetchIrisShares,
    },
    super::super::utils::logger,
    super::{BatchGenerator, GraphIndexer, HawkManager, SharesFetcher},
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
    a4_ref: Option<ActorRef<HawkManager>>,
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
            a4_ref: None,
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
        _: Ctx<'_, Self, Self::Reply>,
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
        _: Ctx<'_, Self, Self::Reply>,
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
        ctx: Ctx<'_, Self, Self::Reply>,
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
        _: Ctx<'_, Self, Self::Reply>,
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
        _: Ctx<'_, Self, Self::Reply>,
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

    /// Actor name - overrides auto-derived name.
    fn name() -> &'static str {
        "Supervisor"
    }

    /// Lifecycle event handler: on_start.
    ///
    /// # Arguments
    ///
    /// * `ref_to_self` - Self referential kameo actor pointer.
    ///
    async fn on_start(&mut self, ref_to_self: ActorRef<Self>) -> Result<(), BoxError> {
        logger::log_lifecycle::<Self>("on_start", None);

        // Instantiate associated actors.
        let a1 = BatchGenerator::new(self.config.clone(), ref_to_self.clone());
        let a2 = SharesFetcher::new(self.config.clone(), ref_to_self.clone());
        let a3 = GraphIndexer::new(self.config.clone(), ref_to_self.clone());
        let a4 = HawkManager::new(self.config.clone(), ref_to_self.clone());

        // Spawn associated actors.
        self.a1_ref = Some(kameo::spawn(a1));
        self.a2_ref = Some(kameo::spawn(a2));
        self.a3_ref = Some(kameo::spawn(a3));
        self.a4_ref = Some(kameo::spawn(a4));

        // Signal start.
        self.a1_ref
            .as_ref()
            .unwrap()
            .tell(OnBeginIndexation)
            .await?;

        Ok(())
    }
}
