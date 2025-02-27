use clap::Parser;
use futures::StreamExt;
use iris_mpc_common::{
    galois::degree4::{basis::Monomial, GaloisRingElement, ShamirGaloisRingShare},
    galois_engine::degree4::GaloisRingIrisCodeShare,
    id::PartyID,
    IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
};
use iris_mpc_store::Store;
use iris_mpc_upgrade::{db::V1Db, utils::install_tracing};
use itertools::izip;
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use std::collections::HashMap;

const APP_NAME: &str = "SMPC";

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long)]
    db_urls: Vec<String>,

    #[clap(long)]
    from: u64,

    #[clap(long)]
    to: u64,

    #[clap(long)]
    environment: String,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    install_tracing();
    let args = Args::parse();

    if args.db_urls.len() != 6 {
        return Err(eyre::eyre!(
            "Expect 5 db urls to be provided: old_participant_1, old_participant_2, \
             old_coordinator_1, new_db0, new_db1, new_db2"
        ));
    }

    let old_left_shares_db0 =
        V1Db::new(format!("{}/{}", args.db_urls[0], "participant1_left").as_str()).await?;
    let old_left_shares_db1 =
        V1Db::new(format!("{}/{}", args.db_urls[1], "participant2_left").as_str()).await?;
    let old_left_masks_db =
        V1Db::new(format!("{}/{}", args.db_urls[2], "coordinator_left").as_str()).await?;

    let old_right_shares_db0 =
        V1Db::new(format!("{}/{}", args.db_urls[0], "participant1_right").as_str()).await?;
    let old_right_shares_db1 =
        V1Db::new(format!("{}/{}", args.db_urls[1], "participant2_right").as_str()).await?;
    let old_right_masks_db1 =
        V1Db::new(format!("{}/{}", args.db_urls[2], "coordinator_right").as_str()).await?;

    let base_schema_name = format!("{}_{}", APP_NAME, args.environment);

    let new_db0 = Store::new(
        &args.db_urls[3],
        format!("{}_{}", base_schema_name, "0").as_str(),
    )
    .await?;
    let new_db1 = Store::new(
        &args.db_urls[4],
        format!("{}_{}", base_schema_name, "1").as_str(),
    )
    .await?;
    let new_db2 = Store::new(
        &args.db_urls[5],
        format!("{}_{}", base_schema_name, "2").as_str(),
    )
    .await?;

    // grab the old shares from the db and reconstruct them
    let old_left_shares0 = old_left_shares_db0
        .stream_shares(args.from..args.to)
        .collect::<Vec<_>>()
        .await;
    let old_left_shares1 = old_left_shares_db1
        .stream_shares(args.from..args.to)
        .collect::<Vec<_>>()
        .await;
    let old_left_masks = old_left_masks_db
        .stream_masks(args.from..args.to)
        .collect::<Vec<_>>()
        .await;

    let old_left_db: HashMap<i64, (EncodedBits, Bits)> = izip!(
        old_left_shares0.into_iter(),
        old_left_shares1.into_iter(),
        old_left_masks.into_iter()
    )
    .map(|(a, b, m)| {
        let (idx0, mut share0) = a.unwrap();
        let (idx1, share1) = b.unwrap();
        let (idx2, mask) = m.unwrap();
        assert_eq!(idx0, idx1);
        assert_eq!(idx1, idx2);
        for (a, b) in share0.0.iter_mut().zip(share1.0) {
            *a = a.wrapping_add(b);
        }
        (idx0, (share0, mask))
    })
    .collect();

    let old_right_shares0 = old_right_shares_db0
        .stream_shares(args.from..args.to)
        .collect::<Vec<_>>()
        .await;
    let old_right_shares1 = old_right_shares_db1
        .stream_shares(args.from..args.to)
        .collect::<Vec<_>>()
        .await;
    let old_right_masks = old_right_masks_db1
        .stream_masks(args.from..args.to)
        .collect::<Vec<_>>()
        .await;

    let old_right_db: HashMap<i64, (EncodedBits, Bits)> = izip!(
        old_right_shares0.into_iter(),
        old_right_shares1.into_iter(),
        old_right_masks.into_iter()
    )
    .map(|(a, b, m)| {
        let (idx0, mut share0) = a.unwrap();
        let (idx1, share1) = b.unwrap();
        let (idx2, mask) = m.unwrap();
        assert_eq!(idx0, idx1);
        assert_eq!(idx1, idx2);
        for (a, b) in share0.0.iter_mut().zip(share1.0) {
            *a = a.wrapping_add(b);
        }
        (idx0, (share0, mask))
    })
    .collect();

    // grab shares from new DB and recombine
    let new_shares0 = new_db0
        .stream_irises_in_range(args.from..args.to)
        .collect::<Vec<_>>()
        .await;

    let new_shares1 = new_db1
        .stream_irises_in_range(args.from..args.to)
        .collect::<Vec<_>>()
        .await;
    let new_shares2 = new_db2
        .stream_irises_in_range(args.from..args.to)
        .collect::<Vec<_>>()
        .await;

    let mut new_left_db = HashMap::new();
    let mut new_right_db = HashMap::new();

    for (iris0, iris1, iris2) in izip!(new_shares0, new_shares1, new_shares2) {
        let iris0 = iris0?;
        let iris1 = iris1?;
        let iris2 = iris2?;

        assert_eq!(iris0.id(), iris1.id());
        assert_eq!(iris1.id(), iris2.id());

        let poly01 =
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID0, PartyID::ID1);
        let poly10 =
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID1, PartyID::ID0);

        let enc_left_bits =
            recombine_enc_bits(iris0.left_code(), iris1.left_code(), poly01, poly10);
        let enc_left_masks = Bits::from(&recombine_enc_masks(
            iris0.left_mask(),
            iris1.left_mask(),
            poly01,
            poly10,
        ));

        let poly12 =
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID1, PartyID::ID2);
        let poly21 =
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID2, PartyID::ID1);

        let enc_left_bits1 =
            recombine_enc_bits(iris1.left_code(), iris2.left_code(), poly12, poly21);
        let enc_left_masks1 = Bits::from(&recombine_enc_masks(
            iris1.left_mask(),
            iris2.left_mask(),
            poly12,
            poly21,
        ));

        assert_eq!(enc_left_bits, enc_left_bits1);
        assert_eq!(enc_left_masks, enc_left_masks1);

        let poly20 =
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID2, PartyID::ID0);
        let poly02 =
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID0, PartyID::ID2);

        let enc_left_bits2 =
            recombine_enc_bits(iris2.left_code(), iris0.left_code(), poly20, poly02);
        let enc_left_masks2 = Bits::from(&recombine_enc_masks(
            iris2.left_mask(),
            iris0.left_mask(),
            poly20,
            poly02,
        ));

        assert_eq!(enc_left_bits, enc_left_bits2);
        assert_eq!(enc_left_masks, enc_left_masks2);

        new_left_db.insert(iris0.id(), (enc_left_bits, enc_left_masks));

        // right
        let enc_right_bits =
            recombine_enc_bits(iris0.right_code(), iris1.right_code(), poly01, poly10);
        let enc_right_masks = Bits::from(&recombine_enc_masks(
            iris0.right_mask(),
            iris1.right_mask(),
            poly01,
            poly10,
        ));

        let enc_right_bits1 =
            recombine_enc_bits(iris1.right_code(), iris2.right_code(), poly12, poly21);
        let enc_right_masks1 = Bits::from(&recombine_enc_masks(
            iris1.right_mask(),
            iris2.right_mask(),
            poly12,
            poly21,
        ));

        assert_eq!(enc_right_bits, enc_right_bits1);
        assert_eq!(enc_right_masks, enc_right_masks1);

        let enc_right_bits2 =
            recombine_enc_bits(iris2.right_code(), iris0.right_code(), poly20, poly02);
        let enc_right_masks2 = Bits::from(&recombine_enc_masks(
            iris2.right_mask(),
            iris0.right_mask(),
            poly20,
            poly02,
        ));

        assert_eq!(enc_right_bits, enc_right_bits2);
        assert_eq!(enc_right_masks, enc_right_masks2);

        new_right_db.insert(iris0.id(), (enc_right_bits, enc_right_masks));
        if iris0.id() % 1000 == 0 {
            tracing::info!("Processed {} shares", iris0.id());
        }
    }
    tracing::info!("Processed {} shares", args.to - args.from);

    assert_eq!(old_left_db.len(), new_left_db.len());
    assert_eq!(old_right_db.len(), new_right_db.len());
    assert_eq!(old_left_db.len(), old_right_db.len());

    tracing::info!("Left / right db lengths match");

    for (idx, (code, mask)) in old_left_db {
        let (new_code, new_mask) = new_left_db.get(&idx).expect("old id is present in new db");
        if code != *new_code {
            tracing::error!("Code for id {} left does not match", idx);
            tracing::error!("Old: {:?}", code);
            tracing::error!("New: {:?}", new_code);
        } else if idx % 250 == 0 {
            tracing::info!("Code for id {} left matches", idx);
        }
        if mask != *new_mask {
            tracing::error!("Mask for id {} left does not match", idx);
            tracing::error!("Old: {:?}", mask);
            tracing::error!("New: {:?}", new_mask);
        } else if idx % 250 == 0 {
            tracing::info!("Mask for id {} left matches", idx);
        }
    }
    for (idx, (code, mask)) in old_right_db {
        let (new_code, new_mask) = new_right_db.get(&idx).expect("old id is present in new db");
        if code != *new_code {
            tracing::error!("Code for id {} right does not match", idx);
            tracing::error!("Old: {:?}", code);
            tracing::error!("New: {:?}", new_code);
        } else if idx % 250 == 0 {
            tracing::info!("Code for id {} right matches", idx);
        }
        if mask != *new_mask {
            tracing::error!("Mask for id {} right does not match", idx);
            tracing::error!("Old: {:?}", mask);
            tracing::error!("New: {:?}", new_mask);
        } else if idx % 250 == 0 {
            tracing::info!("Mask for id {} right matches", idx);
        }
    }

    Ok(())
}

fn recombine_enc_bits(
    a: &[u16],
    b: &[u16],
    lag_point_ab: GaloisRingElement<Monomial>,
    lag_point_ba: GaloisRingElement<Monomial>,
) -> EncodedBits {
    let res = _encode_mask_shares(IRIS_CODE_LENGTH, a, b, lag_point_ab, lag_point_ba);
    assert_eq!(res.len(), IRIS_CODE_LENGTH);
    // reorder the bits according to new encoding
    let mut reordered = [0u16; IRIS_CODE_LENGTH];
    for (i, bit) in res.into_iter().enumerate() {
        reordered[GaloisRingIrisCodeShare::remap_index(i)] = bit;
    }

    EncodedBits(reordered)
}
fn recombine_enc_masks(
    a: &[u16],
    b: &[u16],
    lag_point_ab: GaloisRingElement<Monomial>,
    lag_point_ba: GaloisRingElement<Monomial>,
) -> EncodedBits {
    let res = _encode_mask_shares(MASK_CODE_LENGTH, a, b, lag_point_ab, lag_point_ba);
    assert_eq!(res.len(), MASK_CODE_LENGTH);
    // reorder the bits according to new encoding
    let mut reordered = [0u16; IRIS_CODE_LENGTH];
    for (i, bit) in res.into_iter().enumerate() {
        reordered[GaloisRingIrisCodeShare::remap_index(i)] = bit;
        // we cut the mask in half, so we need to duplicate the bits to get a "full"
        // mask
        reordered[GaloisRingIrisCodeShare::remap_index(i + MASK_CODE_LENGTH)] = bit;
    }

    EncodedBits(reordered)
}

fn _encode_mask_shares(
    length: usize,
    a: &[u16],
    b: &[u16],
    lag_point_ab: GaloisRingElement<Monomial>,
    lag_point_ba: GaloisRingElement<Monomial>,
) -> Vec<u16> {
    let mut res = Vec::with_capacity(length);
    for (a, b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let share0 = GaloisRingElement::from_coefs(a.to_owned().try_into().unwrap());
        let share1 = GaloisRingElement::from_coefs(b.to_owned().try_into().unwrap());

        let share = share0 * lag_point_ab + share1 * lag_point_ba;
        let share = share.to_basis_A();
        res.extend_from_slice(&share.coefs);
    }
    res
}
