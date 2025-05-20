use crate::hawkers::aby3::aby3_store::IrisRef;
pub use crate::hawkers::aby3::aby3_store::VectorId;
use eyre::Result;

use super::{BothEyes, HawkActor, HawkRequest, LEFT, RIGHT};

pub struct ResetUpdateRequest {
    pub vector_id: VectorId,
    pub irises: BothEyes<IrisRef>,
}

pub async fn handle_reset_updates(hawk_actor: &mut HawkActor, request: &HawkRequest) -> Result<()> {
    let mut left_store = hawk_actor.iris_store[LEFT].write().await;
    let mut right_store = hawk_actor.iris_store[RIGHT].write().await;

    // Get the reset updates from the request.
    let either_store = &left_store;
    let updates = request.reset_updates(either_store);

    // Update the iris store with the new irises.
    for update in updates {
        let [l, r] = update.irises;
        left_store.update(update.vector_id, l);
        right_store.update(update.vector_id, r);
    }

    Ok(())
}
