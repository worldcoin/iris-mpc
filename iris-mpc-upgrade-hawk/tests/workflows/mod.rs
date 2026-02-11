pub mod genesis_100;
pub mod genesis_101;
pub mod genesis_102;
pub mod genesis_103;
pub mod genesis_104;
pub mod genesis_105;
pub mod genesis_106;
pub mod genesis_200;

#[macro_export]
macro_rules! join_runners {
    ($join_set:expr) => {{
        let res: Result<Vec<_>, eyre::Report> = $join_set.join_all().await.into_iter().collect();
        if res.is_err() {
            tracing::error!("join failed at line: {}", line!());
        } else {
            tracing::info!("join succeeded at line: {}", line!());
        }
        let _ = res?;
        // allow time to clean up
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }};
}
