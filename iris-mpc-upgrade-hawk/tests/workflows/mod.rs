pub mod genesis_100;
pub mod genesis_101;
pub mod genesis_102;
pub mod genesis_103;
pub mod genesis_104;
pub mod genesis_105;
pub mod genesis_106;
pub mod genesis_107;
pub mod genesis_108;
pub mod genesis_109;

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

#[macro_export]
macro_rules! run_genesis {
    ($self:expr, $args:expr) => {{
        use ::tracing::Instrument as _;
        let _args = $args;
        let mut join_set = ::tokio::task::JoinSet::new();
        for (idx, span, config) in $self
            .configs
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, config)| (idx, ::tracing::info_span!("genesis", idx = idx), config))
        {
            let args = _args.clone();
            join_set.spawn(async move {
                let r = ::iris_mpc_upgrade_hawk::genesis::exec(
                    ::iris_mpc_upgrade_hawk::genesis::ExecutionArgs::from_plaintext_args(
                        args, false,
                    ),
                    config,
                )
                .instrument(span.clone())
                .await;
                ::tracing::info!(genesis_id = idx, "exec_genesis returned {:?}", r);
                r
            });
        }
        $crate::join_runners!(join_set);
    }};
}
