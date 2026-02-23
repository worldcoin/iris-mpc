use std::fmt;

use serde::{Deserialize, Serialize};

use super::ResponsePayload;
use crate::constants::N_PARTIES;

/// Encapsulates common data pertinent to a system processing request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestInfo {
    /// Associated request batch ordinal identifier.
    batch_idx: usize,

    /// Associated request batch item ordinal identifier.
    batch_item_idx: usize,

    /// System responses returned by MPC nodes (one per party).
    responses: [Option<ResponsePayload>; N_PARTIES],

    /// User assigned label ... used to associate child/parent requests.
    label: Option<String>,

    /// optional validation logic
    expected: Option<serde_json::Value>,

    /// Associated unique identifier.
    uid: uuid::Uuid,
}

impl RequestInfo {
    pub fn with_indices(
        batch_idx: usize,
        batch_item_idx: usize,
        label: Option<String>,
        expected: Option<serde_json::Value>,
    ) -> Self {
        Self {
            batch_idx,
            batch_item_idx,
            responses: [const { None }; N_PARTIES],
            label,
            expected,
            uid: uuid::Uuid::new_v4(),
        }
    }

    pub fn label(&self) -> &Option<String> {
        &self.label
    }

    pub fn uid(&self) -> &uuid::Uuid {
        &self.uid
    }

    pub fn is_complete(&self) -> bool {
        self.responses.iter().all(|c| c.is_some())
    }

    /// Records a response from a node. Returns true if all parties have now responded.
    /// Returns Err if validation failed (error is still logged).
    pub fn record_response(&mut self, response: &ResponsePayload) -> Result<bool, Vec<String>> {
        let node_id = response.node_id();
        if node_id >= N_PARTIES {
            tracing::warn!(
                "Ignoring response with out-of-range node_id {} (max {})",
                node_id,
                N_PARTIES - 1
            );
            return Ok(false);
        }
        if self.responses[node_id].is_some() {
            tracing::warn!("Duplicate response for node_id {}", node_id);
        }

        self.responses[node_id] = Some(response.clone());

        if let Err(err_msg) = self
            .expected
            .as_ref()
            .map(|expected| response.matches_expected(expected))
            .unwrap_or(Ok(()))
        {
            tracing::error!("validation failed for response: {:#?}", err_msg);
            Err(err_msg)
        } else {
            Ok(self.is_complete())
        }
    }

    pub fn responses(&self) -> &[Option<ResponsePayload>; N_PARTIES] {
        &self.responses
    }

    pub fn has_error_response(&self) -> bool {
        self.responses.iter().flatten().any(|r| r.is_error())
    }

    pub fn get_error_msgs(&self) -> String {
        self.responses
            .iter()
            .flatten()
            .filter(|r| r.is_error())
            .map(|r| {
                format!(
                    "node {}: {}",
                    r.node_id(),
                    r.error_reason().unwrap_or("no reason given")
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.label {
            Some(label) => write!(f, "{}", label),
            None => write!(f, "{}.{}", self.batch_idx, self.batch_item_idx),
        }
    }
}
