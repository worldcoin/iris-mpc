#[allow(dead_code)]
use crate::indexer::messages;
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisPgresStore;
use kameo::{
    actor::{pubsub::PubSub, ActorRef},
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
    store: Option<IrisPgresStore>,
}

// ------------------------------------------------------------------------
// Constructors.
// ------------------------------------------------------------------------

impl IrisDataLoader {
    pub async fn new(
        config: Config,
        _: ActorRef<PubSub<messages::OnIrisDataPulledFromStore>>,
    ) -> Self {
        // Instantiate a store pointer -> connects to dB.
        let _ = IrisPgresStore::new_from_config(&config).await.unwrap();

        Self { store: None }
    }
}

// ------------------------------------------------------------------------
// Methods.
// ------------------------------------------------------------------------

impl IrisDataLoader {
    // Queries remote store for range of iris identifiers to be processed.
    async fn fetch_iris_data(&self, id_of_iris: i64) -> i64 {
        // JIT set store pointer.
        match self.store {
            Some(_) => {
                info!("TODO: intiialise poiinter to store");
            }
            None => {
                info!("TODO: intiialise poiinter to store");
                info!("TODO: replace mocked range with pulled range");
            }
        }

        info!("TODO: IrisIdStreamReader::fetch_iris_data");

        id_of_iris
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
        info!("Event :: OnIrisIdPulledFromStore :: {}", msg.id_of_iris);

        let _ = self.fetch_iris_data(msg.id_of_iris).await;
    }
}
