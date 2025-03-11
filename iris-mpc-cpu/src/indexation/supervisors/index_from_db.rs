use super::super::{
    actors::{IrisBatchGenerator, IrisDataFetcher},
    messages,
};
use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    error::BoxError,
    mailbox::bounded::BoundedMailbox,
    message::{Context, Message},
    Actor,
};
use tracing;

// ------------------------------------------------------------------------
// Declaration + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Genesis indexation supervisor.
pub struct Supervisor {
    a1_ref: Option<ActorRef<IrisBatchGenerator>>,
    a2_ref: Option<ActorRef<IrisDataFetcher>>,
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
        }
    }
}

// ------------------------------------------------------------------------
// Message handlers.
// ------------------------------------------------------------------------

impl Message<messages::OnIndexationEnd> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnIndexationEnd,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnIndexationEnd :: Supervisor");
    }
}

impl Message<messages::OnIndexationStart> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnIndexationStart,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnIndexationStart :: Supervisor");
    }
}

impl Message<messages::OnBatchIndexationStart> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: messages::OnBatchIndexationStart,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnBatchIndexationStart :: Supervisor");

        for iris_id in msg.batch_range.0..msg.batch_range.1 {
            self.a2_ref
                .as_ref()
                .unwrap()
                .tell(messages::OnBatchElementIndexationStart {
                    id_of_iris: iris_id,
                })
                .await
                .unwrap();
        }
    }
}

impl Message<messages::OnIrisDataPulledFromStore> for Supervisor {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnIrisDataPulledFromStore,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        tracing::info!("Event :: OnIrisDataPulledFromStore :: Supervisor");
    }
}

// ------------------------------------------------------------------------
// Lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for Supervisor {
    type Mailbox = BoundedMailbox<Self>;

    async fn on_start(&mut self, ref_to_self: ActorRef<Self>) -> Result<(), BoxError> {
        tracing::info!("Supervisor :: lifecycle :: on_start");

        // Instantiate associated actors.
        let a1 = IrisBatchGenerator::new(self.config.clone(), ref_to_self.clone());
        let a2 = IrisDataFetcher::new(self.config.clone(), ref_to_self.clone());

        // Spawn associated actors.
        self.a1_ref = Some(kameo::spawn(a1));
        self.a2_ref = Some(kameo::spawn(a2));

        // Signal start.
        self.a1_ref
            .as_ref()
            .unwrap()
            .tell(messages::OnIndexationStart)
            .await?;

        Ok(())
    }
}
