use uuid::Uuid;

use crate::{
    constants::N_PARTIES,
    types::{NetGaloisRingIrisCodeShare, NetIrisCodeSharesJSON},
};
use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, helpers::smpc_request::IrisCodeSharesJSON,
};

/// TODO: review use of these constants.
const IRIS_VERSION: &str = "1.0";
const IRIS_SHARES_VERSION: &str = "1.3";

/// Converts iris code shares into a JSON representation.
#[allow(dead_code)]
fn to_iris_code_party_shares(
    left_shared_code: NetGaloisRingIrisCodeShare,
    left_shared_mask: NetGaloisRingIrisCodeShare,
    right_shared_code: NetGaloisRingIrisCodeShare,
    right_shared_mask: NetGaloisRingIrisCodeShare,
) -> IrisCodePartyShares {
    IrisCodePartyShares::new(
        Uuid::new_v4().to_string(),
        to_iris_code_shares_json(
            left_shared_code,
            left_shared_mask,
            right_shared_code,
            right_shared_mask,
        )
        .to_vec(),
    )
}

/// Converts iris code shares into a JSON representation.
fn to_iris_code_shares_json(
    left_shared_code: [GaloisRingIrisCodeShare; N_PARTIES],
    left_shared_mask: [GaloisRingIrisCodeShare; N_PARTIES],
    right_shared_code: [GaloisRingIrisCodeShare; N_PARTIES],
    right_shared_mask: [GaloisRingIrisCodeShare; N_PARTIES],
) -> NetIrisCodeSharesJSON {
    std::array::from_fn(|i| IrisCodeSharesJSON {
        iris_version: IRIS_VERSION.to_string(),
        iris_shares_version: IRIS_SHARES_VERSION.to_string(),
        right_iris_code_shares: right_shared_code[i].to_base64(),
        right_mask_code_shares: right_shared_mask[i].to_base64(),
        left_iris_code_shares: left_shared_code[i].to_base64(),
        left_mask_code_shares: left_shared_mask[i].to_base64(),
    })
}

#[cfg(test)]
mod tests {
    use super::{to_iris_code_party_shares, to_iris_code_shares_json};
    use crate::{
        irises::generate_iris_code_and_mask_shares_for_both_eyes, types::IrisCodeAndMaskShares,
    };
    use iris_mpc_cpu::execution::hawk_main::BothEyes;
    use rand::{rngs::StdRng, SeedableRng};

    fn create_rng() -> StdRng {
        StdRng::from_entropy()
    }

    fn create_iris_code_and_mask_shares_for_both_eyes() -> BothEyes<IrisCodeAndMaskShares> {
        let mut rng = create_rng();
        generate_iris_code_and_mask_shares_for_both_eyes(&mut rng)
    }

    #[test]
    fn test_convert_to_iris_code_party_shares() {
        let [l, r] = create_iris_code_and_mask_shares_for_both_eyes();
        let [l_code, l_mask] = l;
        let [r_code, r_mask] = r;
        let _ = to_iris_code_party_shares(l_code, l_mask, r_code, r_mask);
    }

    #[test]
    fn test_convert_to_iris_code_shares_json() {
        let [l, r] = create_iris_code_and_mask_shares_for_both_eyes();
        let [l_code, l_mask] = l;
        let [r_code, r_mask] = r;
        let _ = to_iris_code_shares_json(l_code, l_mask, r_code, r_mask);
    }
}
