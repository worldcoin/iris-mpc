use super::super::Supervisor;
use iris_mpc_common::config::Config;
use kameo::{actor::ActorRef, Actor};

// ------------------------------------------------------------------------
// Actor name + state + ctor + methods.
// ------------------------------------------------------------------------

// Name for logging purposes.
const _: &str = "GraphDataWriter";

// Actor: Writes HNSW graph data to store.
#[derive(Actor)]
#[allow(dead_code)]
pub struct GraphDataWriter {
    // System configuration information.
    config: Config,

    // Reference to supervisor.
    supervisor_ref: ActorRef<Supervisor>,
}

// Constructors.
impl GraphDataWriter {
    pub fn new(config: Config, supervisor_ref: ActorRef<Supervisor>) -> Self {
        assert!(config.database.is_some());

        Self {
            config,
            supervisor_ref,
        }
    }
}
