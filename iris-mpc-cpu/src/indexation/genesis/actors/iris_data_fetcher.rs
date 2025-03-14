use super::{
    super::super::utils::log_signal,
    super::Supervisor,
    super::{errors::IndexationError, signals, types::IrisGaloisShares},
};
use iris_mpc_common::config::Config;
use iris_mpc_store::{DbStoredIris as IrisData, Store as IrisStore};
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};

// ------------------------------------------------------------------------
// Declaration + state + ctor + methods.
// ------------------------------------------------------------------------

// Name for logging purposes.
const NAME: &str = "IrisDataFetcher";

// Fetches Iris shares from remote store.
#[derive(Actor)]
#[allow(dead_code)]
pub struct IrisSharesFetcher {
    // System configuration information.
    config: Config,

    // Store provider instance.
    iris_store: Option<IrisStore>,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl IrisSharesFetcher {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            config,
            iris_store: None,
            supervisor_ref,
        }
    }
}

// Methods.
impl IrisSharesFetcher {
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

impl Message<signals::OnBeginBatchItem> for IrisSharesFetcher {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: signals::OnBeginBatchItem,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        log_signal(NAME, "OnBeginBatchItem");

        // Fetch data.
        let iris_data = self.fetch_iris_data(msg.id_of_iris).await.unwrap();

        // Instantiate shares.
        let shares = IrisGaloisShares::new(
            self.config.party_id,
            iris_data.left_code(),
            iris_data.left_mask(),
            iris_data.right_code(),
            iris_data.right_mask(),
        );

        // Signal to supervisor.
        self.supervisor_ref
            .tell(signals::OnFetchOfIrisShares {
                serial_id: iris_data.id(),
                shares,
            })
            .await
            .unwrap();

        Ok(())
    }
}
