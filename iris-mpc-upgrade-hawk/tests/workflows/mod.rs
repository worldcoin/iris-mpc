use tokio::task::JoinSet;

pub mod genesis_100;
pub mod genesis_101;
pub mod genesis_102;
pub mod genesis_103;
pub mod genesis_104;
pub mod genesis_105;
pub mod genesis_106;
pub mod genesis_107;
pub mod genesis_108;

pub async fn join_runners(join_set: JoinSet<eyre::Result<()>>) -> eyre::Result<()> {
    let results = join_set.join_all().await;
    if results.iter().any(|x| x.is_err()) {
        eyre::bail!("at least one peer returned an error");
    }
    Ok(())
}
