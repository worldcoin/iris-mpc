use super::{
    super::utils::{fetcher, logger},
    super::Supervisor,
    super::{
        errors::IndexationError,
        messages::{OnBeginBatchItem, OnFetchIrisShares},
        types::IrisGaloisShares,
    },
};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisStore;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Fetches Iris shares from remote store.
#[derive(Actor, Clone)]
pub struct SharesFetcher {
    // System configuration information.
    config: Config,

    // Store provider instance.
    iris_store: Option<IrisStore>,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl SharesFetcher {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            config,
            iris_store: None,
            supervisor_ref,
        }
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

impl Message<OnBeginBatchItem> for SharesFetcher {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: OnBeginBatchItem,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginBatchItem>(&msg);

        // JIT set pointer to store.
        if self.iris_store.is_none() {
            match IrisStore::new_from_config(&self.config).await {
                Ok(store) => {
                    self.iris_store = Some(store);
                }
                Err(_) => return Err(IndexationError::PostgresConnectionError),
            }
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
        let msg = OnFetchIrisShares {
            batch_idx: msg.batch_idx,
            batch_item_idx: msg.batch_item_idx,
            iris_serial_id: stored.id(),
            iris_shares: shares,
        };
        self.supervisor_ref.tell(msg).await.unwrap();

        Ok(())
    }
}
