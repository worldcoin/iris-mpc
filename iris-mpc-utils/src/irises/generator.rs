use rand::{CryptoRng, Rng};

use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
};
use iris_mpc_cpu::{execution::hawk_main::BothEyes, protocol::shared_iris::GaloisRingSharedIris};

use crate::constants::N_PARTIES;

/// Iris shares for upload to the MPC server.
/// This struct stores the full-size mask share (GaloisRingIrisCodeShare) instead of
/// the trimmed version, because the server expects full-size mask shares when decoding.
#[derive(Debug, Clone)]
pub struct GaloisRingSharedIrisUpload {
    pub code: GaloisRingIrisCodeShare,
    pub mask: GaloisRingIrisCodeShare,
}

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
pub fn generate_iris_shares<R: Rng + CryptoRng>(
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

/// Convenience function that returns 2 sets of Iris shares.
pub fn generate_iris_shares_both_eyes<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code_l: Option<IrisCode>,
    iris_code_r: Option<IrisCode>,
) -> BothEyes<[GaloisRingSharedIris; N_PARTIES]> {
    [
        generate_iris_shares(rng, iris_code_l),
        generate_iris_shares(rng, iris_code_r),
    ]
}

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
/// TODO: rationalize this with the source in shared_iris.rs in iris-mpc-cpu
pub fn generate_iris_shares_mirrored<R: Rng + CryptoRng>(
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

/// Convenience function that returns 2 sets of mirrored Iris shares.
#[allow(dead_code)]
pub fn generate_iris_shares_mirrored_both_eyes<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code_l: Option<IrisCode>,
    iris_code_r: Option<IrisCode>,
) -> BothEyes<[GaloisRingSharedIris; N_PARTIES]> {
    [
        generate_iris_shares_mirrored(rng, iris_code_l),
        generate_iris_shares_mirrored(rng, iris_code_r),
    ]
}

/// Returns generated iris code/mask shares.
fn generate_iris_code_and_mask_shares<R: Rng + CryptoRng>(
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

/// Returns generated iris shares for upload (with full-size mask shares).
/// Use this when generating shares to be sent to the MPC server.
pub fn generate_iris_shares_for_upload<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [GaloisRingSharedIrisUpload; N_PARTIES] {
    let iris_code = iris_code.unwrap_or_else(|| IrisCode::random_rng(rng));
    let [code_shares, mask_shares] = generate_iris_code_and_mask_shares(rng, Some(iris_code));

    [
        // Party 1.
        GaloisRingSharedIrisUpload {
            code: code_shares[0].to_owned(),
            mask: mask_shares[0].to_owned(),
        },
        // Party 2.
        GaloisRingSharedIrisUpload {
            code: code_shares[1].to_owned(),
            mask: mask_shares[1].to_owned(),
        },
        // Party 3.
        GaloisRingSharedIrisUpload {
            code: code_shares[2].to_owned(),
            mask: mask_shares[2].to_owned(),
        },
    ]
}

/// Convenience function that returns 2 sets of Iris shares for upload.
/// Use this when generating shares to be sent to the MPC server.
pub fn generate_iris_shares_for_upload_both_eyes<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code_l: Option<IrisCode>,
    iris_code_r: Option<IrisCode>,
) -> BothEyes<[GaloisRingSharedIrisUpload; N_PARTIES]> {
    [
        generate_iris_shares_for_upload(rng, iris_code_l),
        generate_iris_shares_for_upload(rng, iris_code_r),
    ]
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::{
        generate_iris_code_and_mask_shares, generate_iris_shares, generate_iris_shares_both_eyes,
        generate_iris_shares_mirrored, generate_iris_shares_mirrored_both_eyes, IrisCode,
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
    fn test_can_generate_iris_shares_1() {
        let mut rng = StdRng::seed_from_u64(44);
        let shares = generate_iris_shares(&mut rng, None);
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_iris_shares_2() {
        let mut rng = StdRng::seed_from_u64(45);
        let iris_code = IrisCode::default();
        let shares = generate_iris_shares(&mut rng, Some(iris_code));
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_iris_shares_both_eyes_1() {
        let mut rng = StdRng::seed_from_u64(44);
        let shares = generate_iris_shares_both_eyes(&mut rng, None, None);
        assert_eq!(shares.len(), 2);
    }

    #[test]
    fn test_can_generate_iris_shares_both_eyes_2() {
        let mut rng = StdRng::seed_from_u64(44);
        let iris_code_l = IrisCode::default();
        let iris_code_r = IrisCode::default();
        let shares = generate_iris_shares_both_eyes(&mut rng, Some(iris_code_l), Some(iris_code_r));
        assert_eq!(shares.len(), 2);
    }

    #[test]
    fn test_can_generate_iris_shares_mirrored_1() {
        let mut rng = StdRng::seed_from_u64(46);
        let shares = generate_iris_shares_mirrored(&mut rng, None);
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_iris_shares_mirrored_2() {
        let mut rng = StdRng::seed_from_u64(47);
        let iris_code = IrisCode::default();
        let shares = generate_iris_shares_mirrored(&mut rng, Some(iris_code));
        assert_eq!(shares.len(), N_PARTIES);
    }

    #[test]
    fn test_can_generate_iris_shares_mirrored_both_eyes_1() {
        let mut rng = StdRng::seed_from_u64(46);
        let shares = generate_iris_shares_mirrored_both_eyes(&mut rng, None, None);
        assert_eq!(shares.len(), 2);
    }

    #[test]
    fn test_can_generate_iris_shares_mirrored_both_eyes_2() {
        let mut rng = StdRng::seed_from_u64(46);
        let iris_code_l = IrisCode::default();
        let iris_code_r = IrisCode::default();
        let shares =
            generate_iris_shares_mirrored_both_eyes(&mut rng, Some(iris_code_l), Some(iris_code_r));
        assert_eq!(shares.len(), 2);
    }
}
