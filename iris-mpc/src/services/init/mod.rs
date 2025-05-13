use eyre::Result;
use iris_mpc_common::{config::Config, helpers::kms_dh::derive_shared_secret};

pub async fn initialize_chacha_seeds(config: Config) -> Result<([u32; 8], [u32; 8])> {
    // Init RNGs
    let own_key_arn = config
        .kms_key_arns
        .0
        .get(config.party_id)
        .expect("Expected value not found in kms_key_arns");
    let dh_pairs = match config.party_id {
        0 => (1usize, 2usize),
        1 => (2usize, 0usize),
        2 => (0usize, 1usize),
        _ => unimplemented!(),
    };

    let dh_pair_0: &str = config
        .kms_key_arns
        .0
        .get(dh_pairs.0)
        .expect("Expected value not found in kms_key_arns");
    let dh_pair_1: &str = config
        .kms_key_arns
        .0
        .get(dh_pairs.1)
        .expect("Expected value not found in kms_key_arns");

    // To be used only for e2e testing where we use localstack. There's a bug in
    // localstack's implementation of `derive_shared_secret`. See: https://github.com/localstack/localstack/pull/12071
    let chacha_seeds: ([u32; 8], [u32; 8]) = if config.fixed_shared_secrets {
        ([0u32; 8], [0u32; 8])
    } else {
        (
            bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_0).await?),
            bytemuck::cast(derive_shared_secret(own_key_arn, dh_pair_1).await?),
        )
    };

    Ok(chacha_seeds)
}
