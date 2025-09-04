use eyre::Result;
use futures::TryStreamExt;
use iris_mpc_common::galois;
use iris_mpc_common::galois::degree4::basis::Monomial;
use iris_mpc_common::galois::degree4::GaloisRingElement;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_common::id::PartyID;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::Store;
use iris_mpc_upgrade::utils::install_tracing;
use itertools::Itertools;

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();

    let postgres_client_old_0 = PostgresClient::new(
        "postgres://postgres:postgres@localhost:6200",
        "SMPC_testing_0",
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_old_1 = PostgresClient::new(
        "postgres://postgres:postgres@localhost:6201",
        "SMPC_testing_1",
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_old_2 = PostgresClient::new(
        "postgres://postgres:postgres@localhost:6202",
        "SMPC_testing_2",
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_new_0 = PostgresClient::new(
        "postgres://postgres:postgres@localhost:6203",
        "SMPC_testing_0",
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_new_1 = PostgresClient::new(
        "postgres://postgres:postgres@localhost:6204",
        "SMPC_testing_1",
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_new_2 = PostgresClient::new(
        "postgres://postgres:postgres@localhost:6205",
        "SMPC_testing_2",
        AccessMode::ReadOnly,
    )
    .await?;
    let store_old_0 = Store::new(&postgres_client_old_0).await?;
    let store_old_1 = Store::new(&postgres_client_old_1).await?;
    let store_old_2 = Store::new(&postgres_client_old_2).await?;
    let store_new_0 = Store::new(&postgres_client_new_0).await?;
    let store_new_1 = Store::new(&postgres_client_new_1).await?;
    let store_new_2 = Store::new(&postgres_client_new_2).await?;

    let max_id = store_old_0.get_max_serial_id().await?;
    for store in [
        &store_old_1,
        &store_old_2,
        &store_new_0,
        &store_new_1,
        &store_new_2,
    ] {
        let store_max_id = store.get_max_serial_id().await?;
        if store_max_id != max_id {
            eyre::bail!("Mismatched max serial IDs between databases.");
        }
    }

    // Iterate through the stores
    let mut stream_old_0 = store_old_0.stream_irises_in_range(1..(max_id + 1) as u64);
    let mut stream_old_1 = store_old_1.stream_irises_in_range(1..(max_id + 1) as u64);
    let mut stream_old_2 = store_old_2.stream_irises_in_range(1..(max_id + 1) as u64);
    let mut stream_new_0 = store_new_0.stream_irises_in_range(1..(max_id + 1) as u64);
    let mut stream_new_1 = store_new_1.stream_irises_in_range(1..(max_id + 1) as u64);
    let mut stream_new_2 = store_new_2.stream_irises_in_range(1..(max_id + 1) as u64);

    loop {
        let iris_old_0 = stream_old_0.try_next().await?;
        let iris_old_1 = stream_old_1.try_next().await?;
        let iris_old_2 = stream_old_2.try_next().await?;
        let iris_new_0 = stream_new_0.try_next().await?;
        let iris_new_1 = stream_new_1.try_next().await?;
        let iris_new_2 = stream_new_2.try_next().await?;

        // if all are none, we are done
        if [
            &iris_old_0,
            &iris_old_1,
            &iris_old_2,
            &iris_new_0,
            &iris_new_1,
            &iris_new_2,
        ]
        .iter()
        .all(|x| x.is_none())
        {
            break;
        }

        // unwrap all, panic if any is none, since this is an error
        let iris_old_0 = iris_old_0.expect("Mismatched number of irises between databases");
        let iris_old_1 = iris_old_1.expect("Mismatched number of irises between databases");
        let iris_old_2 = iris_old_2.expect("Mismatched number of irises between databases");
        let iris_new_0 = iris_new_0.expect("Mismatched number of irises between databases");
        let iris_new_1 = iris_new_1.expect("Mismatched number of irises between databases");
        let iris_new_2 = iris_new_2.expect("Mismatched number of irises between databases");

        // basic check, re-randomized irises should have the same ID as the old ones
        if iris_new_0.id() != iris_old_0.id()
            || iris_new_1.id() != iris_old_1.id()
            || iris_new_2.id() != iris_old_2.id()
            || iris_old_0.id() != iris_old_1.id()
            || iris_old_0.id() != iris_old_2.id()
        {
            eyre::bail!(
                "Mismatched iris IDs between databases: {}, {}, {}, {}, {}, {}",
                iris_old_0.id(),
                iris_old_1.id(),
                iris_old_2.id(),
                iris_new_0.id(),
                iris_new_1.id(),
                iris_new_2.id(),
            );
        }
        let id = iris_old_0.id();

        // basic check, re-randomized irises should have different shares than the old ones
        for (iris_new, iris_old) in [
            (&iris_new_0, &iris_old_0),
            (&iris_new_1, &iris_old_1),
            (&iris_new_2, &iris_old_2),
        ] {
            if iris_new.left_code() == iris_old.left_code() {
                eyre::bail!("Left code not changed for iris ID {id} between databases");
            }
            if iris_new.left_mask() == iris_old.left_mask() {
                eyre::bail!("Left mask not changed for iris ID {id} between databases");
            }
            if iris_new.right_code() == iris_old.right_code() {
                eyre::bail!("Right code not changed for iris ID {id} between databases");
            }
            if iris_new.right_mask() == iris_old.right_mask() {
                eyre::bail!("Right mask not changed for iris ID {id} between databases");
            }
        }

        // reconstruct the iris from the old shares
        let (iris_0, iris_1, iris_2) = [iris_old_0, iris_old_1, iris_old_2]
            .iter()
            .enumerate()
            .map(|(party_id, iris)| {
                (
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_mask().try_into().unwrap(),
                    },
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_mask().try_into().unwrap(),
                    },
                )
            })
            .collect_tuple()
            .unwrap();
        let old_left_code = reconstruct_shares(&iris_0.0.coefs, &iris_1.0.coefs, &iris_2.0.coefs);
        let old_left_mask = reconstruct_shares(&iris_0.1.coefs, &iris_1.1.coefs, &iris_2.1.coefs);
        let old_right_code = reconstruct_shares(&iris_0.2.coefs, &iris_1.2.coefs, &iris_2.2.coefs);
        let old_right_mask = reconstruct_shares(&iris_0.3.coefs, &iris_1.3.coefs, &iris_2.3.coefs);

        // reconstruct the iris from the new shares
        let (iris_3, iris_4, iris_5) = [iris_new_0, iris_new_1, iris_new_2]
            .iter()
            .enumerate()
            .map(|(party_id, iris)| {
                (
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_mask().try_into().unwrap(),
                    },
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_mask().try_into().unwrap(),
                    },
                )
            })
            .collect_tuple()
            .unwrap();
        let new_left_code = reconstruct_shares(&iris_3.0.coefs, &iris_4.0.coefs, &iris_5.0.coefs);
        let new_left_mask = reconstruct_shares(&iris_3.1.coefs, &iris_4.1.coefs, &iris_5.1.coefs);
        let new_right_code = reconstruct_shares(&iris_3.2.coefs, &iris_4.2.coefs, &iris_5.2.coefs);
        let new_right_mask = reconstruct_shares(&iris_3.3.coefs, &iris_4.3.coefs, &iris_5.3.coefs);

        if old_left_code != new_left_code {
            eyre::bail!("Mismatched left code for iris ID {id} between databases",);
        }
        if old_left_mask != new_left_mask {
            eyre::bail!("Mismatched left mask for iris ID {id} between databases",);
        }
        if old_right_code != new_right_code {
            eyre::bail!("Mismatched right code for iris ID {id} between databases",);
        }
        if old_right_mask != new_right_mask {
            eyre::bail!("Mismatched right mask for iris ID {id} between databases",);
        }
    }

    tracing::info!("All irises match between databases.");

    Ok(())
}

fn reconstruct_shares(share0: &[u16], share1: &[u16], share2: &[u16]) -> Vec<u16> {
    let lag_01 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID0,
        PartyID::ID1,
    );
    let lag_10 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID1,
        PartyID::ID0,
    );
    let lag_02 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID0,
        PartyID::ID2,
    );
    let lag_20 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID2,
        PartyID::ID0,
    );
    let lag_12 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID1,
        PartyID::ID2,
    );
    let lag_21 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID2,
        PartyID::ID1,
    );

    assert!(share0.len() == share1.len() && share1.len() == share2.len());

    let recon01 = share0
        .chunks_exact(4)
        .zip_eq(share1.chunks_exact(4))
        .flat_map(|(a, b)| {
            let a = GaloisRingElement::<Monomial>::from_coefs(a.try_into().unwrap());
            let b = GaloisRingElement::<Monomial>::from_coefs(b.try_into().unwrap());
            let c = a * lag_01 + b * lag_10;
            c.coefs
        })
        .collect_vec();
    let recon12 = share1
        .chunks_exact(4)
        .zip_eq(share2.chunks_exact(4))
        .flat_map(|(a, b)| {
            let a = GaloisRingElement::<Monomial>::from_coefs(a.try_into().unwrap());
            let b = GaloisRingElement::<Monomial>::from_coefs(b.try_into().unwrap());
            let c = a * lag_12 + b * lag_21;
            c.coefs
        })
        .collect_vec();
    let recon02 = share0
        .chunks_exact(4)
        .zip_eq(share2.chunks_exact(4))
        .flat_map(|(a, b)| {
            let a = GaloisRingElement::<Monomial>::from_coefs(a.try_into().unwrap());
            let b = GaloisRingElement::<Monomial>::from_coefs(b.try_into().unwrap());
            let c = a * lag_02 + b * lag_20;
            c.coefs
        })
        .collect_vec();

    assert_eq!(recon01, recon12);
    assert_eq!(recon01, recon02);
    recon01
}
