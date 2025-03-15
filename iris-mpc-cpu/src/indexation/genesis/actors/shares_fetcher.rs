use super::{
    super::utils::logger,
    super::Supervisor,
    super::{
        errors::IndexationError,
        messages::{OnBeginBatchItem, OnFetchIrisShares},
        types::IrisGaloisShares,
    },
};
use iris_mpc_common::config::Config;
use iris_mpc_store::{DbStoredIris as IrisData, Store as IrisStore};
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
#[allow(dead_code)]
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

// Methods.
impl SharesFetcher {
    // Queries remote store for range of iris identifiers to be processed.
    async fn fetch_iris_data(&mut self, id_of_iris: i64) -> Result<IrisData, IndexationError> {
        // JIT set pointer to remote store.
        if self.iris_store.is_none() {
            match IrisStore::new_from_config(&self.config).await {
                Ok(store) => {
                    self.iris_store = Some(store);
                }
                Err(_) => return Err(IndexationError::PostgresConnectionError),
            }
        }

        // Fetch iris data.
        self.iris_store
            .as_ref()
            .unwrap()
            .fetch_iris_by_serial_id(id_of_iris)
            .await
            .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
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
        logger::log_message::<Self>("OnBeginBatchItem", None);

        // Fetch data.
        let stored = self.fetch_iris_data(msg.serial_id).await.unwrap();

        // Instantiate shares.
        let shares = IrisGaloisShares::new(
            self.config.party_id,
            stored.left_code(),
            stored.left_mask(),
            stored.right_code(),
            stored.right_mask(),
        );

        // Signal to supervisor.
        let msg = OnFetchIrisShares {
            serial_id: stored.id(),
            shares,
        };
        self.supervisor_ref.tell(msg).await.unwrap();

        Ok(())
    }
}
