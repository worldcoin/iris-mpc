use super::{
    super::Supervisor,
    super::{
        errors::IndexationError,
        messages::{OnBeginBatch, OnBeginGraphIndexation, OnEndBatch, OnFetchIrisShares},
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

// Actor: Issues query/insert operations over in-memory HNSW graph.
#[derive(Actor)]
#[allow(dead_code)]
pub struct GraphIndexer {
    // Indexation target, i.e. set of Iris Galois secret shares.
    target: Vec<IrisGaloisShares>,

    // Batch ordinal identifier.
    batch_idx: usize,

    // System configuration information.
    config: Config,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl GraphIndexer {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        Self {
            target: Vec::new(),
            batch_idx: 0,
            config,
            supervisor_ref,
        }
    }
}

// Methods.
impl GraphIndexer {
    async fn do_index_batch(&self) {
        logger::log_todo::<Self>(
            format!(
                "Index graph for Iris batch {} of size {}",
                self.batch_idx,
                self.target.len()
            )
            .as_str(),
        );

        // TODO: remove temporary signal emission.
        let msg = OnEndBatch {
            batch_idx: self.batch_idx,
            batch_size: self.target.len(),
        };
        self.supervisor_ref.tell(msg).await.unwrap();
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
        logger::log_message::<Self, OnBeginBatch>(&msg);

        // Reset batch of shares to be processed.
        self.target = Vec::with_capacity(msg.iris_serial_ids.len());
        self.batch_idx = msg.batch_idx;
    }
}

impl Message<OnBeginGraphIndexation> for GraphIndexer {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnBeginGraphIndexation,
        _: Ctx<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginGraphIndexation>(&msg);

        self.do_index_batch().await;
    }
}

impl Message<OnFetchIrisShares> for GraphIndexer {
    type Reply = Result<(), IndexationError>;

    async fn handle(
        &mut self,
        msg: OnFetchIrisShares,
        _: Ctx<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnFetchIrisShares>(&msg);

        // Extend set of shares to be indexed.
        self.target.push(msg.iris_shares);

        // When batch is complete then signal readiness for indexing.
        if self.target.len() == self.target.capacity() {
            let msg = OnBeginGraphIndexation {
                batch_idx: msg.batch_idx,
                batch_size: self.target.len(),
            };
            self.supervisor_ref.tell(msg).await.unwrap();
        }

        Ok(())
    }
}
