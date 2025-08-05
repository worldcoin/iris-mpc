//. Enum over set of node types.
#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
pub enum NodeType {
    CPU,
    GPU,
}

/// Type alias: Ordinal identifier of an MPC participant.
pub type PartyIdx = usize;
