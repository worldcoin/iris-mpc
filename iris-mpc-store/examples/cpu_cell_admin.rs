//! Separately credentialed installation/deployment/recovery entrypoint.
//! Set CPU_CELL_DATABASE_URL locally; never pass credentials in command arguments.
use eyre::{bail, Result, WrapErr};
use iris_mpc_store::cpu_cell::{bootstrap, BootstrapConfig, CpuCellAuthority, GenerationIdentity};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use std::{str::FromStr, time::Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let values: Vec<&str> = args.iter().map(String::as_str).collect();
    let url = std::env::var("CPU_CELL_DATABASE_URL").wrap_err(
        "CPU_CELL_DATABASE_URL must identify the separately credentialed administration connection",
    )?;
    let options = PgConnectOptions::from_str(&url)?
        .options([("statement_timeout", "30000"), ("lock_timeout", "5000")]);
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect_with(options)
        .await?;
    match values.as_slice() {
        ["bootstrap", schema, owner, incarnation, party, intent_login, recovery_login] => {
            bootstrap(&pool, &BootstrapConfig {
                schema, authority_owner: owner, incarnation, party_id: party.parse()?,
                intent_login, recovery_login,
            }).await?;
        }
        ["register", schema, incarnation, generation, login, slot, image, schema_version, protocol, configuration] => {
            CpuCellAuthority::new(pool.clone(), schema).register_generation(incarnation,generation,login,&GenerationIdentity {
                slot:(*slot).into(), image:(*image).into(), schema_version:(*schema_version).into(),
                protocol:(*protocol).into(), configuration:(*configuration).into(),
            }).await?;
        }
        ["intent", schema, incarnation, intent, source, destination, automatic] => {
            CpuCellAuthority::new(pool.clone(),schema).write_intent(incarnation,intent,source,destination,automatic.parse()?).await?;
        }
        ["status", schema] => {
            println!("{:#?}", CpuCellAuthority::new(pool.clone(),schema).status().await?);
        }
        ["begin-recovery", schema, incarnation, recovery, expected_revision] => {
            CpuCellAuthority::new(pool.clone(),schema).begin_recovery(incarnation,recovery,expected_revision.parse()?).await?;
        }
        ["finish-recovery", schema, incarnation, recovery, fresh_generation, fencing_ack, new_incarnation] => {
            CpuCellAuthority::new(pool.clone(),schema).finish_recovery(incarnation,recovery,fresh_generation,fencing_ack,new_incarnation).await?;
        }
        _ => bail!("Expected one of:\nbootstrap SCHEMA OWNER INCARNATION PARTY INTENT_LOGIN RECOVERY_LOGIN\nregister SCHEMA INCARNATION GENERATION LOGIN SLOT IMAGE_DIGEST SCHEMA_VERSION PROTOCOL CONFIGURATION\nintent SCHEMA INCARNATION INTENT SOURCE DESTINATION AUTOMATIC_BOOL\nstatus SCHEMA\nbegin-recovery SCHEMA INCARNATION RECOVERY EXPECTED_REVISION\nfinish-recovery SCHEMA INCARNATION RECOVERY FRESH_GENERATION EXTERNAL_FENCING_ACK NEW_INCARNATION"),
    }
    pool.close().await;
    println!("CPU authority operation completed");
    Ok(())
}
