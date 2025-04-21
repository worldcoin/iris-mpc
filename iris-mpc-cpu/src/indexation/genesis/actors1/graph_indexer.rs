use super::super::{
    errors::IndexationError,
    messages::{
        OnBeginGraphIndexation, OnBeginIndexationOfBatch, OnEndIndexationOfBatch, OnFetchIrisShares,
    },
    types::IrisGaloisShares,
    utils::logger,
};
use crate::hawkers::aby3::aby3_store::SharedIrisesRef;
use iris_mpc_common::config::Config;
use kameo::{
    actor::ActorRef,
    message::{Context, Message},
    Actor,
};
use kameo_actors::message_bus as mbus;

// ------------------------------------------------------------------------
// Component state.
// ------------------------------------------------------------------------

// Actor: Issues query/insert operations over in-memory HNSW graph.
#[allow(dead_code)]
pub struct GraphIndexer {
    // Indexation target, i.e. set of Iris Galois secret shares.
    batch: Vec<IrisGaloisShares>,

    // Batch ordinal identifier.
    batch_idx: usize,

    // System configuration information.
    config: Config,

    iris_store: Option<[SharedIrisesRef; 2]>,

    // Reference to message bus mediating intra-actor communications.
    mbus_ref: ActorRef<mbus::MessageBus>,
}

// Constructor.
impl GraphIndexer {
    pub fn new(config: Config, mbus_ref: ActorRef<mbus::MessageBus>) -> Self {
        Self {
            batch: Vec::new(),
            batch_idx: 0,
            iris_store: None,
            config,
            mbus_ref,
        }
    }
}

// ------------------------------------------------------------------------
// Component methods.
// ------------------------------------------------------------------------

impl GraphIndexer {
    async fn do_index_batch(&self) {
        logger::log_todo::<Self>(
            format!(
                "Index graph for Iris batch {} of size {}",
                self.batch_idx,
                self.batch.len()
            )
            .as_str(),
        );

        // TODO: remove temporary signal emission.
        let msg = OnEndIndexationOfBatch {
            batch_idx: self.batch_idx,
            batch_size: self.batch.len(),
        };
        self.mbus_ref.tell(mbus::Publish(msg)).await.unwrap();
    }
}

// ------------------------------------------------------------------------
// Component message handlers.
// ------------------------------------------------------------------------

impl Message<OnBeginIndexationOfBatch> for GraphIndexer {
    // Reply type.
    type Reply = ();

    // Handler.
    async fn handle(
        &mut self,
        msg: OnBeginIndexationOfBatch,
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnBeginIndexationOfBatch>(&msg);

        // Reset batch of shares to be processed.
        self.batch = Vec::with_capacity(msg.iris_serial_ids.len());
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
        _: &mut Context<Self, Self::Reply>,
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
        _: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        logger::log_message::<Self, OnFetchIrisShares>(&msg);

        // Extend set of shares to be indexed.
        self.batch.push(msg.iris_shares);

        // When batch is complete then signal readiness for indexing.
        if self.batch.len() == self.batch.capacity() {
            let msg = OnBeginGraphIndexation {
                batch_idx: msg.batch_idx,
                batch_size: self.batch.len(),
            };
            self.mbus_ref.tell(mbus::Publish(msg)).await.unwrap();
        }

        Ok(())
    }
}

// ------------------------------------------------------------------------
// Component lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for GraphIndexer {
    // Internal error type.
    type Error = IndexationError;

    /// Name - overrides auto-derived name.
    fn name() -> &'static str {
        "GraphIndexer"
    }

    /// Lifecycle event handler: on_start.
    ///
    /// State initialisation hook.
    async fn on_start(&mut self, actor_ref: ActorRef<Self>) -> Result<(), Self::Error> {
        logger::log_lifecycle::<Self>("on_start", None);

        // Register message handlers.
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnBeginIndexationOfBatch>(),
            ))
            .await
            .unwrap();
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnBeginGraphIndexation>(),
            ))
            .await
            .unwrap();
        self.mbus_ref
            .tell(mbus::Register(
                actor_ref.clone().recipient::<OnFetchIrisShares>(),
            ))
            .await
            .unwrap();

        // let iris_store = [(); 2].map(|_| SharedIrisesRef::default());
        // let d = IrisLoader {
        //     party_id: self.config.party_id.clone(),
        //     db_size: &mut 0_usize,
        //     irises: [iris_store[0].write().await, iris_store[1].write().await],
        // };

        Ok(())
    }
}
