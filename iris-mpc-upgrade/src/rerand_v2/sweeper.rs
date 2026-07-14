use std::collections::HashMap;
use std::time::Duration;

use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsClient;
use eyre::{ensure, Result};
use iris_mpc_common::rerand_offsets::{retarget_shares, EpochKey, EpochSeed};
use iris_mpc_store::rerand::{
    build_epoch_zero_rerand_context, get_rerand_epoch_inventory, RerandPass, RerandRowUpdate,
    RERAND_CHECK_PROTOCOL_VERSION,
};
use iris_mpc_store::Store;
use tokio::time::Instant;

use super::coordination::{self, Completion, EpochCheckPassed, StoreRegistry, OFFSET_GENERATION};
use super::epoch;
use super::epoch_check::{self, EpochCheckBinding};
use super::RerandSweeperConfig;

#[derive(Debug, Default)]
pub struct SweepOutcome {
    pub epoch: u32,
    pub retargeted: u64,
    pub semantic_write_races: u64,
}

async fn publish_completion(
    s3: &S3Client,
    secrets: &SecretsClient,
    config: &RerandSweeperConfig,
    epoch_number: u32,
    writer_role: &str,
    expected_commitment: [u8; 32],
) -> Result<()> {
    let seed = epoch::load_verified_epoch_seed(
        secrets,
        s3,
        &config.s3_bucket,
        &config.environment,
        &config.coordination_id,
        epoch_number,
        config.party_id,
    )
    .await?;
    ensure!(
        iris_mpc_common::rerand_offsets::seed_commitment(&seed) == expected_commitment,
        "persisted completion seed commitment does not match the verified seed"
    );
    coordination::publish_completion(
        s3,
        &config.s3_bucket,
        &config.environment,
        &config.coordination_id,
        &Completion {
            environment: config.environment.clone(),
            coordination_id: config.coordination_id.clone(),
            offset_generation: OFFSET_GENERATION,
            epoch: epoch_number,
            party_id: config.party_id,
            store_kind: config.store_kind,
            store_id: config.store_id.clone(),
            writer_role: writer_role.to_owned(),
            seed_commitment: hex::encode(expected_commitment),
        },
    )
    .await
}

pub async fn run_single_pass(
    store: &Store,
    s3: &S3Client,
    secrets: &SecretsClient,
    config: &RerandSweeperConfig,
) -> Result<SweepOutcome> {
    ensure!(config.rerand_enabled, "rerandomization is disabled");
    ensure!(config.party_id < 3, "party id must be 0, 1, or 2");
    ensure!(config.chunk_size > 0, "chunk size must be positive");
    epoch_check::validate_epoch_check_config(config).await?;
    coordination::validate_environment(&config.environment)?;
    coordination::validate_coordination_id(&config.coordination_id)?;
    ensure!(
        !config.store_id.is_empty() && !config.s3_bucket.is_empty(),
        "store id and S3 bucket must be non-empty"
    );
    let registry = StoreRegistry::parse(&config.expected_store_registry)?;
    let expected_local = registry.expected(config.party_id, config.store_kind)?;
    ensure!(
        expected_local.store_id == config.store_id,
        "local store config differs from the exact registry"
    );
    let chunk_size = i64::try_from(config.chunk_size)?;
    let poll = Duration::from_millis(config.poll_interval_ms.max(1));
    let mut pass = RerandPass::acquire(
        &store.pool,
        &config.store_id,
        &config.environment,
        &config.coordination_id,
        config.party_id,
        &config.store_kind.to_string(),
    )
    .await?;
    ensure!(
        pass.state.writer_role == expected_local.writer_role,
        "database writer role differs from the exact registry"
    );

    // Close the crash window between durable local completion and S3 publish.
    // Re-publishing an immutable equal marker is idempotent.
    if pass.state.last_completed_epoch > 0 {
        let completed_commitment = pass
            .state
            .last_seed_commitment
            .ok_or_else(|| eyre::eyre!("completed epoch has no seed commitment"))?;
        ensure!(
            coordination::epoch_check_passed(
                s3,
                &config.s3_bucket,
                &config.environment,
                &config.coordination_id,
                pass.state.last_completed_epoch,
                &completed_commitment,
                &registry,
                config.store_kind,
            )
            .await?,
            "completed rerandomization epoch has no validated check marker"
        );
        if pass.state.last_completed_epoch > 1 {
            let previous_seed = epoch::load_verified_epoch_seed(
                secrets,
                s3,
                &config.s3_bucket,
                &config.environment,
                &config.coordination_id,
                pass.state.last_completed_epoch - 1,
                config.party_id,
            )
            .await?;
            let previous_commitment =
                iris_mpc_common::rerand_offsets::seed_commitment(&previous_seed);
            coordination::wait_for_epoch_completion(
                s3,
                &config.s3_bucket,
                &config.environment,
                &config.coordination_id,
                pass.state.last_completed_epoch - 1,
                &previous_commitment,
                &registry,
                poll,
            )
            .await?;
        }
        publish_completion(
            s3,
            secrets,
            config,
            pass.state.last_completed_epoch,
            &pass.state.writer_role,
            completed_commitment,
        )
        .await?;
    }

    let pass_epoch = match pass.state.active_epoch {
        Some(epoch) => epoch,
        None => pass
            .state
            .last_completed_epoch
            .checked_add(1)
            .ok_or_else(|| eyre::eyre!("rerandomization epoch overflow"))?,
    };
    if pass_epoch > 1 {
        let previous_commitment = pass
            .state
            .last_seed_commitment
            .ok_or_else(|| eyre::eyre!("previous epoch has no seed commitment"))?;
        coordination::wait_for_epoch_completion(
            s3,
            &config.s3_bucket,
            &config.environment,
            &config.coordination_id,
            pass_epoch - 1,
            &previous_commitment,
            &registry,
            poll,
        )
        .await?;
    }
    let (target_seed, target_commitment) = epoch::ensure_epoch_seed(
        secrets,
        s3,
        &config.s3_bucket,
        &config.environment,
        &config.coordination_id,
        pass_epoch,
        config.party_id,
        poll,
    )
    .await?;

    let (mut next_id, max_id) = pass.begin_or_resume(pass_epoch, target_commitment).await?;
    let store_registry_commitment = registry.store_kind_commitment(config.store_kind)?;
    let epoch_check_already_passed = coordination::epoch_check_passed(
        s3,
        &config.s3_bucket,
        &config.environment,
        &config.coordination_id,
        pass_epoch,
        &target_commitment,
        &registry,
        config.store_kind,
    )
    .await?;
    ensure!(
        !epoch_check_already_passed
            || next_id
                == max_id.checked_add(1).ok_or_else(|| eyre::eyre!(
                    "rerandomization cursor overflow while validating epoch-check marker"
                ))?,
        "epoch-check marker exists for a pass whose durable cursor is not complete"
    );
    let mut seeds = HashMap::<u32, EpochSeed>::from([(pass_epoch, target_seed)]);
    let mut outcome = SweepOutcome {
        epoch: pass_epoch,
        ..Default::default()
    };

    while next_id <= max_id {
        let max_exclusive = max_id
            .checked_add(1)
            .ok_or_else(|| eyre::eyre!("rerandomization cursor overflow"))?;
        let end = next_id
            .checked_add(chunk_size)
            .ok_or_else(|| eyre::eyre!("rerandomization chunk overflow"))?
            .min(max_exclusive);
        let started = Instant::now();
        let rows = pass.fetch_rows(next_id, end).await?;
        let mut updates = Vec::with_capacity(rows.len());
        for source in rows {
            let row = source.iris;
            let from_epoch = row.rerand_epoch();
            ensure!(
                from_epoch >= 0 && from_epoch <= pass_epoch as i32,
                "row {} has impossible future epoch {}",
                row.id(),
                from_epoch
            );
            if from_epoch == pass_epoch as i32 {
                continue;
            }
            if from_epoch > 0 && !seeds.contains_key(&(from_epoch as u32)) {
                let seed = epoch::load_verified_epoch_seed(
                    secrets,
                    s3,
                    &config.s3_bucket,
                    &config.environment,
                    &config.coordination_id,
                    from_epoch as u32,
                    config.party_id,
                )
                .await?;
                seeds.insert(from_epoch as u32, seed);
            }

            let mut left_code = row.left_code().to_vec();
            let mut left_mask = row.left_mask().to_vec();
            let mut right_code = row.right_code().to_vec();
            let mut right_mask = row.right_mask().to_vec();
            let from_seed =
                (from_epoch > 0).then(|| seeds.get(&(from_epoch as u32)).expect("loaded"));
            retarget_shares(
                config.party_id as usize,
                row.id(),
                EpochKey::new(from_epoch, from_seed),
                EpochKey::new(pass_epoch as i32, Some(&target_seed)),
                &mut left_code,
                &mut left_mask,
                &mut right_code,
                &mut right_mask,
            )?;
            updates.push(RerandRowUpdate {
                id: row.id(),
                expected_version_id: row.version_id(),
                expected_semantic_id: source.semantic_id,
                from_epoch,
                left_code,
                left_mask,
                right_code,
                right_mask,
            });
        }

        let attempted = updates.len() as u64;
        let applied = pass.apply(&updates, pass_epoch).await?;
        outcome.retargeted += applied;
        outcome.semantic_write_races += attempted - applied;
        pass.advance(end).await?;

        if config.rows_per_second > 0 {
            let budget =
                Duration::from_secs_f64((end - next_id) as f64 / config.rows_per_second as f64);
            if let Some(remaining) = budget.checked_sub(started.elapsed()) {
                tokio::time::sleep(remaining).await;
            }
        }
        next_id = end;
    }

    if !epoch_check_already_passed {
        let inventory = get_rerand_epoch_inventory(&store.pool).await?;
        let rerand = build_epoch_zero_rerand_context(config.party_id as usize, seeds, &inventory)?;
        let binding = EpochCheckBinding::new(
            config,
            store_registry_commitment,
            pass_epoch,
            target_commitment,
        );
        epoch_check::run_store_persisted_epoch_check(config, &binding, store, &rerand).await?;
        coordination::publish_epoch_check_passed(
            s3,
            &config.s3_bucket,
            &config.environment,
            &config.coordination_id,
            &EpochCheckPassed {
                environment: config.environment.clone(),
                coordination_id: config.coordination_id.clone(),
                offset_generation: OFFSET_GENERATION,
                check_protocol: RERAND_CHECK_PROTOCOL_VERSION,
                epoch: pass_epoch,
                store_kind: config.store_kind,
                store_registry_commitment: hex::encode(store_registry_commitment),
                seed_commitment: hex::encode(target_commitment),
            },
        )
        .await?;
    }

    ensure!(
        coordination::epoch_check_passed(
            s3,
            &config.s3_bucket,
            &config.environment,
            &config.coordination_id,
            pass_epoch,
            &target_commitment,
            &registry,
            config.store_kind,
        )
        .await?,
        "rerandomization epoch cannot complete without its validated check marker"
    );
    let completed = pass.complete(RERAND_CHECK_PROTOCOL_VERSION).await?;
    ensure!(
        completed.last_completed_epoch == pass_epoch
            && completed.last_seed_commitment == Some(target_commitment),
        "completed pass binding changed"
    );
    coordination::publish_completion(
        s3,
        &config.s3_bucket,
        &config.environment,
        &config.coordination_id,
        &Completion {
            environment: config.environment.clone(),
            coordination_id: config.coordination_id.clone(),
            offset_generation: OFFSET_GENERATION,
            epoch: pass_epoch,
            party_id: config.party_id,
            store_kind: config.store_kind,
            store_id: config.store_id.clone(),
            writer_role: completed.writer_role,
            seed_commitment: hex::encode(target_commitment),
        },
    )
    .await?;
    Ok(outcome)
}
