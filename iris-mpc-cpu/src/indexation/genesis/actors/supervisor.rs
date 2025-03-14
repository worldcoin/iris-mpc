use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use {
    super::super::signals::{OnBegin, OnBeginBatch, OnBeginBatchItem, OnEnd, OnFetchIrisShares},
    super::super::utils::logger,
    super::{BatchGenerator, GraphDataWriter, GraphIndexer, SharesFetcher},
};

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Name for logging purposes.
const NAME: &str = "Supervisor";

// Actor: Genesis indexation supervisor.
#[derive(Clone)]
pub struct Supervisor {
    a1_ref: Option<ActorRef<BatchGenerator>>,
    a2_ref: Option<ActorRef<SharesFetcher>>,
    a3_ref: Option<ActorRef<GraphIndexer>>,
    a4_ref: Option<ActorRef<GraphDataWriter>>,
    config: Config,
}

impl Supervisor {
    // Ctor.
    pub fn new(config: Config) -> Self {
        assert!(config.database.is_some());

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

impl Message<OnEnd> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, _: OnEnd, _: Context<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message(NAME, "OnEnd", None);
    }
}

impl Message<OnBegin> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, _: OnBegin, _: Context<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message(NAME, "OnBegin", None);
    }
}

impl Message<OnBeginBatch> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnBeginBatch,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message(NAME, "OnBeginBatch", None);

        // Signal to other interested actors.
        self.a3_ref
            .as_ref()
            .unwrap()
            .tell(msg.clone())
            .await
            .unwrap();

        // For each item in batch, signal that it is ready to be processing.
        for serial_id in msg.serial_ids {
            let msg = OnBeginBatchItem { serial_id };
            self.a2_ref.as_ref().unwrap().tell(msg).await.unwrap();
        }
    }
}

impl Message<OnFetchIrisShares> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnFetchIrisShares,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message(
            NAME,
            "OnFetchOfIrisShares",
            Some(format!("iris serial-id = {}", msg.serial_id).as_str()),
        );

        // Signal that Iris shares are ready for indexation.
        self.a3_ref.as_ref().unwrap().tell(msg).await.unwrap()
    }
}

// ------------------------------------------------------------------------
// Actor lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for Supervisor {
    type Mailbox = BoundedMailbox<Self>;

    /// Lifecycle event handler: on_start.
    ///
    /// # Arguments
    ///
    /// * `ref_to_self` - Self referential kameo actor pointer.
    ///
    async fn on_start(&mut self, ref_to_self: ActorRef<Self>) -> Result<(), BoxError> {
        logger::log_lifecycle(NAME, "on_start", None);

        // Instantiate associated actors.
        let a1 = BatchGenerator::new(self.config.clone(), ref_to_self.clone());
        let a2 = SharesFetcher::new(self.config.clone(), ref_to_self.clone());
        let a3 = GraphIndexer::new(self.config.clone(), ref_to_self.clone());
        let a4 = GraphDataWriter::new(self.config.clone(), ref_to_self.clone());

        // Spawn associated actors.
        self.a1_ref = Some(kameo::spawn(a1));
        self.a2_ref = Some(kameo::spawn(a2));
        self.a3_ref = Some(kameo::spawn(a3));
        self.a4_ref = Some(kameo::spawn(a4));

        // Signal start.
        self.a1_ref.as_ref().unwrap().tell(OnBegin).await?;

        Ok(())
    }
}
