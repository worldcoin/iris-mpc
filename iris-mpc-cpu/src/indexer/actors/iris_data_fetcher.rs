#[allow(dead_code)]
use super::{
    super::supervisors::IndexFromDbSupervisor as Supervisor,
    super::{errors::IndexationError, messages},
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

// Actor: Reads Iris data from remote store.
#[derive(Actor)]
pub struct IrisDataFetcher {
    // System configuration information.
    config: Config,

    // Store provider instance.
    store: Option<IrisStore>,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl IrisDataFetcher {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            config,
            store: None,
            supervisor_ref,
        }
    }
}

impl IrisDataFetcher {
    // Queries remote store for range of iris identifiers to be processed.
    async fn fetch_iris_data(&mut self, id_of_iris: i64) -> Result<IrisData, IndexationError> {
        // JIT set pointer to remote store.
        if self.store.is_none() {
            match IrisStore::new_from_config(&self.config).await {
                Ok(store) => {
                    self.store = Some(store);
                }
                Err(_) => return Err(IndexationError::PostgresConnectionError),
            }
        }

        // Fetch iris data.
        self.store
            .as_ref()
            .unwrap()
            .fetch_iris_by_id(id_of_iris)
            .await
            .map_err(|_| IndexationError::PostgresFetchIrisByIdError)
    }
}

// ------------------------------------------------------------------------
// Message Handlers.
// ------------------------------------------------------------------------

// Message handler.
impl Message<messages::OnBatchElementIndexationStart> for IrisDataFetcher {
    type Reply = Result<IrisData, IndexationError>;

    async fn handle(
        &mut self,
        msg: messages::OnBatchElementIndexationStart,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        let g = self.fetch_iris_data(msg.id_of_iris).await.unwrap();

        self.supervisor_ref
            .tell(messages::OnIrisDataPulledFromStore {
                id_of_iris: g.id(),
                code_left: g.left_code().to_vec(),
                code_right: g.right_code().to_vec(),
                mask_left: g.left_mask().to_vec(),
                mask_right: g.right_mask().to_vec(),
            })
            .await
            .unwrap();

        Ok(g)
    }
}
