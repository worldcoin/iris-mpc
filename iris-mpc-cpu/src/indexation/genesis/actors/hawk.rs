use super::super::{errors::IndexationError, utils::logger, Supervisor};
use crate::execution::hawk_main::{HawkActor, HawkArgs};
use iris_mpc_common::config::Config;
use kameo::{actor::ActorRef, mailbox::bounded::BoundedMailbox, Actor};

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Actor: Manages interaction with HNSW graph store/data.
#[allow(dead_code)]
pub struct HawkManager {
    // System configuration information.
    config: Config,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl HawkManager {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        assert!(config.database.is_some());

        Self {
            config,
            supervisor_ref,
        }
    }
}

// ------------------------------------------------------------------------
// Actor lifecycle handlers.
// ------------------------------------------------------------------------

impl Actor for HawkManager {
    // By default mailbox is limited to 1000 messages.
    type Mailbox = BoundedMailbox<Self>;
    type Error = IndexationError;

    /// Actor name - overrides auto-derived name.
    fn name() -> &'static str {
        "Supervisor"
    }

    /// Lifecycle event handler: on_start.
    ///
    /// # Arguments
    ///
    /// * `ref_to_self` - Self referential kameo actor pointer.
    ///
    async fn on_start(&mut self, _: ActorRef<Self>) -> Result<(), Self::Error> {
        logger::log_lifecycle::<Self>("on_start", None);

        let node_addresses: Vec<String> = self
            .config
            .node_hostnames
            .iter()
            .zip(self.config.service_ports.iter())
            .map(|(host, port)| format!("{}:{}", host, port))
            .collect();
        let hawk_args = HawkArgs {
            party_index: self.config.party_id,
            addresses: node_addresses.clone(),
            connection_parallelism: self.config.hawk_connection_parallelism,
            request_parallelism: self.config.hawk_request_parallelism,
            disable_persistence: self.config.disable_persistence,
        };
        let _ = HawkActor::from_cli(&hawk_args)
            .await
            .map_err(|_| IndexationError::HawkActorError)
            .unwrap();

        Ok(())
    }
}
