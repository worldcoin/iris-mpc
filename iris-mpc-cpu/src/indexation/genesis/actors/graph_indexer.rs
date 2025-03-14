use super::{
    super::Supervisor,
    super::{
        errors::IndexationError,
        signals::{OnBeginBatch, OnFetchIrisShares},
        types::IrisGaloisShares,
        utils::logger,
    },
};
use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    message::{Context as Ctx, Message},
    Actor,
};

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Name for logging purposes.
const NAME: &str = "GraphIndexer";

// Actor: Issues query/insert operations over in-memory HNSW graph.
#[derive(Actor)]
#[allow(dead_code)]
pub struct GraphIndexer {
    // Batch of Iris Galois shares awaiting indexation.
    batch: Vec<IrisGaloisShares>,

    // System configuration information.
    config: Config,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl GraphIndexer {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            batch: Vec::new(),
            config,
            supervisor_ref,
        }
    }
}

impl GraphIndexer {
    async fn do_index_batch(&self) {
        logger::log_todo(
            NAME,
            format!("Index graph for Iris batch of size {}", self.batch.len()).as_str(),
        );
    }
}

// ------------------------------------------------------------------------
// Actor message handlers.
// ------------------------------------------------------------------------

impl Message<OnBeginBatch> for GraphIndexer {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(&mut self, msg: OnBeginBatch, _: Ctx<'_, Self, Self::Reply>) -> Self::Reply {
        logger::log_message(NAME, "OnBeginBatch", None);

        // Initialise new batch.
        self.batch = Vec::with_capacity(msg.serial_ids.len());
    }
}

impl Message<OnFetchIrisShares> for GraphIndexer {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: OnFetchIrisShares,
        _: Ctx<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message(NAME, "OnFetchOfIrisShares", None);

        // Grow next indexation batch & index when full.
        self.batch.push(msg.shares);
        if self.batch.len() == self.batch.capacity() {
            self.do_index_batch().await;
        }

        Ok(())
    }
}
