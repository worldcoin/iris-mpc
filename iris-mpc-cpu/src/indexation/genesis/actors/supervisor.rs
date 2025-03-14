use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use {
    super::super::signals,
    super::super::utils::logger,
    super::{GraphDataWriter, GraphIndexer, IrisBatchGenerator, IrisSharesFetcher},
};

// ------------------------------------------------------------------------
// Declaration + state + ctor + methods.
// ------------------------------------------------------------------------

// Name for logging purposes.
const NAME: &str = "Supervisor";

// Actor: Genesis indexation supervisor.
pub struct Supervisor {
    a1_ref: Option<ActorRef<IrisBatchGenerator>>,
    a2_ref: Option<ActorRef<IrisSharesFetcher>>,
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

impl Message<signals::OnEnd> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: signals::OnEnd,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_signal(NAME, "OnEnd", None);
    }
}

impl Message<signals::OnBegin> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: signals::OnBegin,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_signal(NAME, "OnBegin", None);
    }
}

impl Message<signals::OnBeginBatch> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: signals::OnBeginBatch,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_signal(NAME, "OnBeginBatch", None);

        // TODO: spawn pool to process concurrently.
        for iris_id in msg.batch {
            // Signal that a batch item is ready for processing.
            let msg = signals::OnBeginBatchItem {
                id_of_iris: iris_id,
            };
            self.a2_ref.as_ref().unwrap().tell(msg).await.unwrap();
        }
    }
}

impl Message<signals::OnFetchOfIrisShares> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: signals::OnFetchOfIrisShares,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_signal(
            NAME,
            "OnFetchOfIrisShares",
            Some(format!("iris serial-id = {}", msg.serial_id).as_str()),
        );

        // Signal that Iris shares are ready for indexation.
        let msg = signals::OnBeginIrisSharesIndexation {
            serial_id: msg.serial_id,
            shares: msg.shares,
        };
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
        let a1 = IrisBatchGenerator::new(self.config.clone(), ref_to_self.clone());
        let a2 = IrisSharesFetcher::new(self.config.clone(), ref_to_self.clone());
        let a3 = GraphIndexer::new(self.config.clone(), ref_to_self.clone());
        let a4 = GraphDataWriter {};

        // Spawn associated actors.
        self.a1_ref = Some(kameo::spawn(a1));
        self.a2_ref = Some(kameo::spawn(a2));
        self.a3_ref = Some(kameo::spawn(a3));
        self.a4_ref = Some(kameo::spawn(a4));

        // Signal start.
        self.a1_ref.as_ref().unwrap().tell(signals::OnBegin).await?;

        Ok(())
    }
}
