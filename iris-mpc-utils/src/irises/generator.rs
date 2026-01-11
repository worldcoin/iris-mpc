use rand::{CryptoRng, Rng};

use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

use crate::constants::N_PARTIES;

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
/// TODO: rationalize this with the source in shared_iris.rs in iris-mpc-cpu
#[allow(dead_code)]
pub fn generate_iris_shares_locally<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [GaloisRingSharedIris; N_PARTIES] {
    let iris_code = iris_code.unwrap_or_else(|| IrisCode::random_rng(rng));
    let [code_shares, mask_shares] = generate_iris_code_and_mask_shares(rng, Some(iris_code));

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
pub fn generate_iris_shares_locally_mirrored<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [GaloisRingSharedIris; N_PARTIES] {
    let iris_code = iris_code.unwrap_or_else(|| IrisCode::random_rng(rng));
    let [code_shares, mask_shares] = generate_iris_code_and_mask_shares(rng, Some(iris_code));
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

/// Returns generated iris code/mask shares.
pub fn generate_iris_code_and_mask_shares<R: Rng + CryptoRng>(
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

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::{
        generate_iris_code_and_mask_shares, generate_iris_shares_locally,
        generate_iris_shares_locally_mirrored, IrisCode,
    };
    use crate::constants::N_PARTIES;

    #[test]
    fn test_can_generate_iris_code_share_1() {
        let mut rng = StdRng::seed_from_u64(42);
        let shares = generate_iris_code_and_mask_shares(&mut rng, None);
        assert_eq!(shares.len(), 2);
        assert_eq!(shares[0].len(), N_PARTIES);
        assert_eq!(shares[1].len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_iris_code_share_2() {
        let mut rng = StdRng::seed_from_u64(43);
        let iris_code = IrisCode::default();
        let shares = generate_iris_code_and_mask_shares(&mut rng, Some(iris_code));
        assert_eq!(shares.len(), 2);
        assert_eq!(shares[0].len(), N_PARTIES);
        assert_eq!(shares[1].len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_shared_iris_locally_1() {
        let mut rng = StdRng::seed_from_u64(44);
        let shares = generate_iris_shares_locally(&mut rng, None);
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_shared_iris_locally_2() {
        let mut rng = StdRng::seed_from_u64(45);
        let iris_code = IrisCode::default();
        let shares = generate_iris_shares_locally(&mut rng, Some(iris_code));
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_shared_iris_locally_mirrored_1() {
        let mut rng = StdRng::seed_from_u64(46);
        let shares = generate_iris_shares_locally_mirrored(&mut rng, None);
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_shared_iris_locally_mirrored_2() {
        let mut rng = StdRng::seed_from_u64(47);
        let iris_code = IrisCode::default();
        let shares = generate_iris_shares_locally_mirrored(&mut rng, Some(iris_code));
        assert_eq!(shares.len(), N_PARTIES);
    }
}
