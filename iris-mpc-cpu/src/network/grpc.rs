use std::sync::Arc;

use dashmap::DashMap;

use crate::execution::player::Identity;

use super::value::NetworkValue;

type MessageQueueStore = DashMap<
    Identity,
    (
        Arc<async_channel::Sender<NetworkValue>>,
        Arc<async_channel::Receiver<NetworkValue>>,
    ),
>;