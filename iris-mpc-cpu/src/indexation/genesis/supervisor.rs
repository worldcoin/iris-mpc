use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use tracing;
use {
    super::components::{GraphDataWriter, GraphIndexer, IrisBatchGenerator, IrisSharesFetcher},
    super::messages,
};

// ------------------------------------------------------------------------
// Declaration + state + ctor + methods.
// ------------------------------------------------------------------------

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
            config,
            a1_ref: None,
            a2_ref: None,
            a3_ref: None,
            a4_ref: None,
        }
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

impl Message<messages::OnEnd> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnEnd,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnIndexationEnd :: Supervisor");
    }
}

impl Message<messages::OnBegin> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnBegin,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnIndexationStart :: Supervisor");
    }
}

impl Message<messages::OnBeginOfBatchIndexation> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: messages::OnBeginOfBatchIndexation,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnBatchIndexationStart :: Supervisor");

        // TODO: spawn pool to process concurrently.
        for iris_id in msg.batch {
            self.a2_ref
                .as_ref()
                .unwrap()
                .tell(messages::OnBeginOfBatchItemIndexation {
                    id_of_iris: iris_id,
                })
                .await
                .unwrap();
        }
    }
}

impl Message<messages::OnFetchOfIrisShares> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: messages::OnFetchOfIrisShares,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnFetchOfIrisData :: Supervisor");

        self.a3_ref
            .as_ref()
            .unwrap()
            .tell(messages::OnIndexationOfFetchedIrisDataBegin {
                fetched_iris_data: msg,
            })
            .await
            .unwrap()
    }
}

// ------------------------------------------------------------------------
// Actor lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for Supervisor {
    type Mailbox = BoundedMailbox<Self>;

    async fn on_start(&mut self, ref_to_self: ActorRef<Self>) -> Result<(), BoxError> {
        tracing::info!("Supervisor :: lifecycle :: on_start");

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
        self.a1_ref
            .as_ref()
            .unwrap()
            .tell(messages::OnBegin)
            .await?;

        Ok(())
    }
}
