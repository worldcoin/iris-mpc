use clap::Parser;
use futures::StreamExt;
use gpu_iris_mpc::{
    setup::{
        galois::degree4::{basis::Monomial, GaloisRingElement, ShamirGaloisRingShare},
        id::PartyID,
    },
    store,
    upgrade::db::V1Db,
    IRIS_CODE_LENGTH,
};
use itertools::izip;
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use std::collections::HashMap;

// quick checking script that recombines the shamir shares for a local server
// setup and prints the iris code share

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long)]
    db_urls:      Vec<String>,
    #[clap(long)]
    num_elements: u64,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = Args::parse();

    if args.db_urls.len() != 7 {
        return Err(eyre::eyre!(
            "Expect 5 db urls to be provided: old_left_db0, old_left_db1, old_right_db0, \
             old_right_db1, new_db0, new_db1, new_db2"
        ));
    }

    let old_left_db0 = V1Db::new(&args.db_urls[0]).await?;
    let old_left_db1 = V1Db::new(&args.db_urls[1]).await?;
    let old_right_db0 = V1Db::new(&args.db_urls[2]).await?;
    let old_right_db1 = V1Db::new(&args.db_urls[3]).await?;

    let new_db0 = store::Store::new(&args.db_urls[4], "upgrade").await?;
    let new_db1 = store::Store::new(&args.db_urls[5], "upgrade").await?;
    let new_db2 = store::Store::new(&args.db_urls[6], "upgrade").await?;

    // grab the old shares from the db and reconstruct them
    let old_left_shares0 = old_left_db0
        .stream_shares(0..args.num_elements)
        .collect::<Vec<_>>()
        .await;
    let old_left_shares1 = old_left_db1
        .stream_shares(0..args.num_elements)
        .collect::<Vec<_>>()
        .await;
    let old_left_masks = old_left_db0
        .stream_masks(0..args.num_elements)
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

    let old_right_shares0 = old_right_db0
        .stream_shares(0..args.num_elements)
        .collect::<Vec<_>>()
        .await;
    let old_right_shares1 = old_right_db1
        .stream_shares(0..args.num_elements)
        .collect::<Vec<_>>()
        .await;
    let old_right_masks = old_right_db0
        .stream_masks(0..args.num_elements)
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
    let new_shares0 = new_db0.stream_irises().await.collect::<Vec<_>>().await;
    let new_shares1 = new_db1.stream_irises().await.collect::<Vec<_>>().await;
    let new_shares2 = new_db2.stream_irises().await.collect::<Vec<_>>().await;

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
        let enc_left_masks = Bits::from(&recombine_enc_bits(
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
        let enc_left_masks1 = Bits::from(&recombine_enc_bits(
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
        let enc_left_masks2 = Bits::from(&recombine_enc_bits(
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
        let enc_right_masks = Bits::from(&recombine_enc_bits(
            iris0.right_mask(),
            iris1.right_mask(),
            poly01,
            poly10,
        ));

        let enc_right_bits1 =
            recombine_enc_bits(iris1.right_code(), iris2.right_code(), poly12, poly21);
        let enc_right_masks1 = Bits::from(&recombine_enc_bits(
            iris1.right_mask(),
            iris2.right_mask(),
            poly12,
            poly21,
        ));

        assert_eq!(enc_right_bits, enc_right_bits1);
        assert_eq!(enc_right_masks, enc_right_masks1);

        let enc_right_bits2 =
            recombine_enc_bits(iris2.right_code(), iris0.right_code(), poly20, poly02);
        let enc_right_masks2 = Bits::from(&recombine_enc_bits(
            iris2.right_mask(),
            iris0.right_mask(),
            poly20,
            poly02,
        ));

        assert_eq!(enc_right_bits, enc_right_bits2);
        assert_eq!(enc_right_masks, enc_right_masks2);

        new_right_db.insert(iris0.id(), (enc_right_bits, enc_right_masks));
    }

    assert_eq!(old_left_db.len(), new_left_db.len());
    assert_eq!(old_right_db.len(), new_right_db.len());
    assert_eq!(old_left_db.len(), old_right_db.len());

    for (idx, (code, mask)) in old_left_db {
        let (new_code, new_mask) = new_left_db.get(&idx).expect("old id is present in new db");
        assert_eq!(code, *new_code);
        assert_eq!(mask, *new_mask);
    }
    for (idx, (code, mask)) in old_right_db {
        let (new_code, new_mask) = new_right_db.get(&idx).expect("old id is present in new db");
        assert_eq!(code, *new_code);
        assert_eq!(mask, *new_mask);
    }

    Ok(())
}

fn recombine_enc_bits(
    a: &[u16],
    b: &[u16],
    lag_point_ab: GaloisRingElement<Monomial>,
    lag_point_ba: GaloisRingElement<Monomial>,
) -> EncodedBits {
    let mut res = Vec::with_capacity(IRIS_CODE_LENGTH);
    for (a, b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let share0 = GaloisRingElement::from_coefs(a.to_owned().try_into().unwrap());
        let share1 = GaloisRingElement::from_coefs(b.to_owned().try_into().unwrap());

        let share = share0 * lag_point_ab + share1 * lag_point_ba;
        let share = share.to_basis_A();
        res.extend_from_slice(&share.coefs);
    }
    EncodedBits(res.try_into().expect("iris codes have correct size"))
}
