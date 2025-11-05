use rand::{CryptoRng, Rng};

use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
};
use iris_mpc_cpu::{execution::hawk_main::BothEyes, protocol::shared_iris::GaloisRingSharedIris};

use crate::types::{NetGaloisRingIrisCodeShare, NetGaloisRingSharedIris};

/// Returns generated iris code shares over both eyes.
pub fn generate_iris_code_shares<R: Rng + CryptoRng>(
    rng: &mut R,
) -> BothEyes<[NetGaloisRingIrisCodeShare; 2]> {
    [
        // Left.
        generate_iris_code_share(rng, None),
        // Right.
        generate_iris_code_share(rng, None),
    ]
}

/// Returns generated iris code/mask shares.
pub fn generate_iris_code_share<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: Option<IrisCode>,
) -> [NetGaloisRingIrisCodeShare; 2] {
    let iris_code = match iris_code {
        Some(iris_code) => iris_code,
        None => IrisCode::random_rng(rng),
    };

    [
        // Code.
        GaloisRingIrisCodeShare::encode_iris_code(&iris_code.code, &iris_code.mask, rng),
        // Mask.
        GaloisRingIrisCodeShare::encode_mask_code(&iris_code.mask, rng),
    ]
}

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
pub fn generate_shared_iris_locally<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: IrisCode,
) -> NetGaloisRingSharedIris {
    let [code_shares, mask_shares] = generate_iris_code_share(rng, Some(iris_code));

    [
        // Party 1.
        GaloisRingSharedIris {
            code: code_shares[0].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[0]),
        },
        // Party 2.
        GaloisRingSharedIris {
            code: code_shares[1].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[1]),
        },
        // Party 3.
        GaloisRingSharedIris {
            code: code_shares[2].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[2]),
        },
    ]
}

/// Returns generated iris shares from an iris code using local randomness alongside its mirrored component.
pub fn generate_shared_iris_locally_mirrored<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: IrisCode,
) -> NetGaloisRingSharedIris {
    let [code_shares, mask_shares] = generate_iris_code_share(rng, Some(iris_code));
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
            code: code_shares_mirrored[0].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[0]),
        },
        // Party 2.
        GaloisRingSharedIris {
            code: code_shares_mirrored[1].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[1]),
        },
        // Party 3.
        GaloisRingSharedIris {
            code: code_shares_mirrored[2].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[2]),
        },
    ]
}
