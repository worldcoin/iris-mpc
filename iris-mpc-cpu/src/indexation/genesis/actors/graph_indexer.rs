use super::{
    super::Supervisor,
    super::{errors::IndexationError, messages, types::IrisGaloisShares},
};
use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};

// ------------------------------------------------------------------------
// Declaration + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Issues query/insert operations over in-memory HNSW graph.
#[derive(Actor)]
#[allow(dead_code)]
pub struct GraphIndexer {
    // System configuration information.
    config: Config,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl GraphIndexer {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            config,
            supervisor_ref,
        }
    }
}

impl GraphIndexer {
    async fn do_index_graph(&self, serial_id: i64, _: IrisGaloisShares) {
        tracing::info!("TODO: Index graph :: Iris Serial ID {}", serial_id,);
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

// Message handler :: OnIndexationOfFetchedIrisDataBegin.
impl Message<messages::OnBeginOfIrisSharesIndexation> for GraphIndexer {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: messages::OnBeginOfIrisSharesIndexation,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        self.do_index_graph(msg.shares.serial_id, msg.shares.shares)
            .await;

        Ok(())
    }
}
