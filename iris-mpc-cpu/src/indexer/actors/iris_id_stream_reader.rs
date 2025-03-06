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
use std::ops::Range;
use tracing::info;

// ------------------------------------------------------------------------
// Declaration + state.
// ------------------------------------------------------------------------

// Actor: Reads Iris identifiers streamed from remote store.
#[derive(Actor)]
pub struct IrisIdStreamReader {
    broker: ActorRef<PubSub<messages::OnIrisIdPulledFromStore>>,
    store: Option<IrisPgresStore>,
}

// ------------------------------------------------------------------------
// Constructors.
// ------------------------------------------------------------------------

impl IrisIdStreamReader {
    pub async fn new(
        config: Config,
        broker: ActorRef<PubSub<messages::OnIrisIdPulledFromStore>>,
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
// Methods.
// ------------------------------------------------------------------------

impl IrisIdStreamReader {
    // Qeuries remote store for range of iris identifiers to be processed.
    fn fetch_iris_id_range(&self) -> Range<i64> {
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
        1..101
    }
}

// ------------------------------------------------------------------------
// Message Handlers.
// ------------------------------------------------------------------------

// Message: OnIndexationStart.
impl Message<messages::OnIndexationStart> for IrisIdStreamReader {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        _: messages::OnIndexationStart,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        info!("Event :: OnIndexationStart");

        // Mock streaming a set of identifiers from remote store.
        for iris_id in self.fetch_iris_id_range() {
            self.broker.tell(Publish(messages::OnIrisIdPulledFromStore {
                id_of_iris: iris_id,
            }));
            info!("Event :: OnIrisIdPulledFromStore :: {}", iris_id);
        }
    }
}
