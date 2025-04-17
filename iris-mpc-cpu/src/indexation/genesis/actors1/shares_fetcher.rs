use super::super::{
    errors::IndexationError,
    messages::{OnBeginIndexationOfBatchItem, OnFetchIrisShares},
    types::IrisGaloisShares,
    utils::{self, fetcher, logger},
};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisStore;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};
use kameo_actors::message_bus::MessageBus;

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Fetches Iris shares from remote store.
#[derive(Actor, Clone)]
#[allow(dead_code)]
pub struct SharesFetcher {
    // System configuration information.
    config: Config,

    // Store provider instance.
    iris_store: Option<IrisStore>,

    // Reference to message bus mediating intra-actor communications.
    mbus_ref: ActorRef<MessageBus>,
}

// Constructors.
impl SharesFetcher {
    pub fn new(config: Config, supervisor_ref: ActorRef<MessageBus>) -> Self {
        Self {
            config,
            iris_store: None,
            mbus_ref: supervisor_ref,
        }
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

impl Message<OnBeginIndexationOfBatchItem> for SharesFetcher {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: OnBeginIndexationOfBatchItem,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginIndexationOfBatchItem>(&msg);

        // JIT set pointer to store.
        if self.iris_store.is_none() {
            self.iris_store = Some(utils::pgres::get_store_instance(&self.config).await);
        }

        // Fetch iris data.
        let stored =
            fetcher::fetch_iris_data(self.iris_store.as_ref().unwrap(), msg.iris_serial_id)
                .await
                .unwrap();

        // Instantiate Galois shares.
        let shares = IrisGaloisShares::new(
            self.config.party_id,
            stored.left_code(),
            stored.left_mask(),
            stored.right_code(),
            stored.right_mask(),
        );

        // Signal to supervisor.
        let _ = OnFetchIrisShares {
            batch_idx: msg.batch_idx,
            batch_item_idx: msg.batch_item_idx,
            iris_serial_id: stored.id(),
            iris_shares: shares,
        };
        // self.mbus_ref.tell(msg).await.unwrap();

        Ok(())
    }
}
