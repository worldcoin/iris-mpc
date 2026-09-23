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

    pub(crate) fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub(crate) fn batch_item_idx(&self) -> usize {
        self.batch_item_idx
    }

    pub fn uid(&self) -> &uuid::Uuid {
        &self.uid
    }

    pub fn is_complete(&self) -> bool {
        self.responses.iter().all(|c| c.is_some())
    }

    /// Records a response from a node. Returns true if all parties have now responded.
    /// Logs errors for out-of-range or duplicate responses but continues tracking.
    pub fn record_response(&mut self, response: &ResponsePayload) -> bool {
        let node_id = response.node_id();
        if node_id >= N_PARTIES {
            tracing::error!(
                "Received response with out-of-range node_id {} (max {})",
                node_id,
                N_PARTIES - 1
            );
            return false;
        }
        if self.responses[node_id].is_some() {
            tracing::error!("Duplicate response for node_id {}", node_id);
            return false;
        }

        self.responses[node_id] = Some(response.clone());
        self.is_complete()
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

    /// Validates all responses against expected values, if provided.
    /// Should only be called after all responses are complete.
    pub fn validate_expected(&self) -> Result<(), String> {
        if let Some(ref expected) = self.expected {
            for response in self.responses.iter().flatten() {
                response
                    .matches_expected(expected)
                    .map_err(|v| v.join("\n"))?;
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use iris_mpc_common::helpers::smpc_response::UniquenessResult;

    use super::{RequestInfo, ResponsePayload};

    #[test]
    fn responses_are_stored_in_party_order_not_arrival_order() {
        let mut info = RequestInfo::with_indices(7, 3, Some("ordered".to_string()), None);
        for node_id in [2, 0, 1] {
            let response = ResponsePayload::Uniqueness(UniquenessResult::new_error_result(
                node_id,
                "request-id".to_string(),
                "probe",
            ));
            info.record_response(&response);
        }

        let node_ids = info
            .responses()
            .iter()
            .map(|response| match response.as_ref().unwrap() {
                ResponsePayload::Uniqueness(result) => result.node_id,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        assert_eq!(node_ids, vec![0, 1, 2]);
    }
}
