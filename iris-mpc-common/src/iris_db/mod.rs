use crate::galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingMaskCodeShare};
use iris::IrisCode;
use rand::{rngs::StdRng, SeedableRng};

pub mod db;
pub mod iris;
pub mod shamir_db;
pub mod shamir_iris;

pub fn get_dummy_shares_for_deletion(
    party_id: usize,
) -> (GaloisRingIrisCodeShare, GaloisRingMaskCodeShare) {
    let mut rng: StdRng = StdRng::seed_from_u64(0);
    let dummy: IrisCode = IrisCode::default();
    let iris_share: GaloisRingIrisCodeShare =
        GaloisRingIrisCodeShare::encode_iris_code(&dummy.code, &dummy.mask, &mut rng)[party_id]
            .clone();
    let mask_share: GaloisRingMaskCodeShare =
        GaloisRingIrisCodeShare::encode_mask_code(&dummy.mask, &mut rng)[party_id]
            .clone()
            .into();
    (iris_share, mask_share)
}
