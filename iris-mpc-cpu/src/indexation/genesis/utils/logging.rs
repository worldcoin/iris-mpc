pub(crate) fn log_signal(actor_name: &str, event: &str) {
    tracing::info!("{} :: SIGNAL :: {}", actor_name, event);
}

pub(crate) fn log_lifecycle(actor_name: &str, episode: &str) {
    tracing::info!("{} :: LIFECYCLE :: {}", actor_name, episode);
}
