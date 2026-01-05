use rand::{CryptoRng, Rng};

use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
};
use iris_mpc_cpu::{execution::hawk_main::BothEyes, protocol::shared_iris::GaloisRingSharedIris};

use crate::constants::N_PARTIES;

/// Returns generated iris code shares over both eyes.
pub fn generate_iris_code_and_mask_party_shares_for_both_eyes<R: Rng + CryptoRng>(
    rng: &mut R,
) -> BothEyes<[[GaloisRingIrisCodeShare; N_PARTIES]; 2]> {
    [
        // Left.
        generate_iris_code_and_mask_party_shares(rng, None),
        // Right.
        generate_iris_code_and_mask_party_shares(rng, None),
    ]
}

/// Returns generated iris code/mask shares.
pub fn generate_iris_code_and_mask_party_shares<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [[GaloisRingIrisCodeShare; N_PARTIES]; 2] {
    let iris_code = iris_code.unwrap_or_else(|| IrisCode::random_rng(rng));

    [
        // Code.
        GaloisRingIrisCodeShare::encode_iris_code(&iris_code.code, &iris_code.mask, rng),
        // Mask.
        GaloisRingIrisCodeShare::encode_mask_code(&iris_code.mask, rng),
    ]
}

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
/// TODO: rationalize this with the source in shared_iris.rs in iris-mpc-cpu
#[allow(dead_code)]
pub fn generate_iris_party_shares_locally<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [GaloisRingSharedIris; N_PARTIES] {
    let iris_code = iris_code.unwrap_or_else(|| IrisCode::random_rng(rng));
    let [code_shares, mask_shares] = generate_iris_code_and_mask_party_shares(rng, Some(iris_code));

    [
        // Party 1.
        GaloisRingSharedIris {
            code: code_shares[0].to_owned(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[0]),
        },
        // Party 2.
        GaloisRingSharedIris {
            code: code_shares[1].to_owned(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[1]),
        },
        // Party 3.
        GaloisRingSharedIris {
            code: code_shares[2].to_owned(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[2]),
        },
    ]
}

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
/// TODO: rationalize this with the source in shared_iris.rs in iris-mpc-cpu
#[allow(dead_code)]
pub fn generate_iris_party_shares_locally_mirrored<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [GaloisRingSharedIris; N_PARTIES] {
    let iris_code = iris_code.unwrap_or_else(|| IrisCode::random_rng(rng));
    let [code_shares, mask_shares] = generate_iris_code_and_mask_party_shares(rng, Some(iris_code));
    let code_shares_mirrored = code_shares
        .iter()
        .map(|code| code.mirrored_code())
        .collect::<Vec<_>>();
    let mask_shares_mirrored = mask_shares
        .iter()
        .map(|mask| mask.mirrored_mask())
        .collect::<Vec<_>>();

    [
        // Party 1.
        GaloisRingSharedIris {
            code: code_shares_mirrored[0].to_owned(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[0]),
        },
        // Party 2.
        GaloisRingSharedIris {
            code: code_shares_mirrored[1].to_owned(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[1]),
        },
        // Party 3.
        GaloisRingSharedIris {
            code: code_shares_mirrored[2].to_owned(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[2]),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        generate_iris_code_and_mask_party_shares,
        generate_iris_code_and_mask_party_shares_for_both_eyes, generate_iris_party_shares_locally,
        generate_iris_party_shares_locally_mirrored, IrisCode,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn create_iris_code() -> IrisCode {
        IrisCode::default()
    }

    fn create_rng() -> StdRng {
        StdRng::from_entropy()
    }

    #[test]
    fn test_can_generate_iris_code_shares() {
        let mut rng = create_rng();
        let _ = generate_iris_code_and_mask_party_shares_for_both_eyes(&mut rng);
    }

    #[test]
    fn test_can_generate_iris_code_share() {
        let mut rng = create_rng();
        let _ = generate_iris_code_and_mask_party_shares(&mut rng, None);
        let _ = generate_iris_code_and_mask_party_shares(&mut rng, Some(create_iris_code()));
    }

    #[test]
    fn test_can_generate_shared_iris_locally() {
        let mut rng = create_rng();
        let _ = generate_iris_party_shares_locally(&mut rng, None);
        let _ = generate_iris_party_shares_locally(&mut rng, Some(create_iris_code()));
    }

    #[test]
    fn test_can_generate_shared_iris_locally_mirrored() {
        let mut rng = create_rng();
        let _ = generate_iris_party_shares_locally_mirrored(&mut rng, None);
        let _ = generate_iris_party_shares_locally_mirrored(&mut rng, Some(create_iris_code()));
    }
}
