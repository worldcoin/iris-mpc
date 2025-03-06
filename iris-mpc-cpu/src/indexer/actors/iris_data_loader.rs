use crate::indexer::messages;
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisPgresStore;
use kameo::{
    actor::{
        pubsub::{PubSub, Publish},
        ActorRef,
    },
    message::{Context, Message},
    Actor,
};
use tracing::info;

// ------------------------------------------------------------------------
// Declaration + state.
// ------------------------------------------------------------------------

// Actor: Reads Iris data from remote store.
#[derive(Actor)]
pub struct IrisDataLoader {
    broker: ActorRef<PubSub<messages::OnIrisDataPulledFromStore>>,
    store: Option<IrisPgresStore>,
}

// ------------------------------------------------------------------------
// Constructors.
// ------------------------------------------------------------------------

impl IrisDataLoader {
    pub async fn new(
        config: Config,
        broker: ActorRef<PubSub<messages::OnIrisDataPulledFromStore>>,
    ) -> Self {
        // Instantiate a store pointer -> connects to dB.
        let _ = IrisPgresStore::new_from_config(&config).await.unwrap();

        Self {
            broker,
            store: None,
        }
    }
}

// ------------------------------------------------------------------------
// Message Handlers.
// ------------------------------------------------------------------------

// Message handler.
impl Message<messages::OnIrisIdPulledFromStore> for IrisDataLoader {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: messages::OnIrisIdPulledFromStore,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        info!("OnIrisIdPulledFromStore :: {}", msg.id_of_iris);

        info!("TODO: pull iris data from store");
    }
}
