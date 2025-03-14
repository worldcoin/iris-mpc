use super::{
    super::supervisor::Supervisor,
    super::{errors::IndexationError, messages},
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

// // Iris ID, see pgres primary key.
// pub(crate) id_of_iris: i64,

// // Iris code share: left.
// pub(crate) left_code: Vec<u16>,

// // Iris mask share: left.
// pub(crate) left_mask: Vec<u16>,

// // Iris code share: right.
// pub(crate) right_code: Vec<u16>,

// // Iris mask share: right.
// pub(crate) right_mask: Vec<u16>,

impl GraphIndexer {
    async fn index_graph_from_fetched_iris_data(
        &self,
        serial_id: i64,
        _: Vec<u16>,
        _: Vec<u16>,
        _: Vec<u16>,
        _: Vec<u16>,
    ) {
        tracing::info!("TODO: Index graph :: Iris Serial ID {}", serial_id,);
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

// Message handler :: OnIndexationOfFetchedIrisDataBegin.
impl Message<messages::OnIndexationOfFetchedIrisDataBegin> for GraphIndexer {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: messages::OnIndexationOfFetchedIrisDataBegin,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        self.index_graph_from_fetched_iris_data(
            msg.fetched_iris_data.id_of_iris,
            msg.fetched_iris_data.left_code,
            msg.fetched_iris_data.left_mask,
            msg.fetched_iris_data.right_code,
            msg.fetched_iris_data.right_mask,
        )
        .await;

        Ok(())
    }
}
