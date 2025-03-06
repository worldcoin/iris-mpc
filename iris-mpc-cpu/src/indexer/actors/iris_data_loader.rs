use crate::indexer::messages;
use kameo::{
    message::{Context, Message},
    Actor,
};
use tracing::info;

// Actor: Reads Iris data from remote store.
#[derive(Actor, Default)]
pub struct IrisDataLoader {}

// Message handler.
impl Message<messages::OnIrisIdPulledFromStore> for IrisDataLoader {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: messages::OnIrisIdPulledFromStore,
        _: Context<'_, Self, Self::Reply>,
    ) -> Self::Reply {
        info!("OnIrisIdPulledFromStore :: {}", msg.id_of_iris);

        info!("TODO: pull iris data from store");
    }
}
