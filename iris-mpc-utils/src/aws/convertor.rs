use base64::{engine::general_purpose::STANDARD as b64, Engine};
use serde_json;
use sodiumoxide::crypto::{box_::PublicKey, sealedbox};
use uuid::Uuid;

use crate::constants::N_PARTIES;
use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare,
    helpers::{
        sha256::sha256_as_hex_string,
        smpc_request::{IrisCodeSharesJSON, SharesS3Object},
    },
};

/// TODO: review use of these constants.
const IRIS_VERSION: &str = "1.0";
const IRIS_SHARES_VERSION: &str = "1.3";

/// Converts iris code shares into a JSON representation.
#[allow(dead_code)]
pub fn to_iris_code_party_shares(
    l_code: [GaloisRingIrisCodeShare; N_PARTIES],
    l_mask: [GaloisRingIrisCodeShare; N_PARTIES],
    r_code: [GaloisRingIrisCodeShare; N_PARTIES],
    r_mask: [GaloisRingIrisCodeShare; N_PARTIES],
    signup_id: Option<String>,
) -> IrisCodePartyShares {
    let signup_id = signup_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    IrisCodePartyShares::new(
        signup_id,
        to_iris_code_shares_json(l_code, l_mask, r_code, r_mask).to_vec(),
    )
}

/// Converts iris code shares into a JSON representation.
pub fn to_iris_code_shares_json(
    l_code: [GaloisRingIrisCodeShare; N_PARTIES],
    l_mask: [GaloisRingIrisCodeShare; N_PARTIES],
    r_code: [GaloisRingIrisCodeShare; N_PARTIES],
    r_mask: [GaloisRingIrisCodeShare; N_PARTIES],
) -> [IrisCodeSharesJSON; N_PARTIES] {
    std::array::from_fn(|i| IrisCodeSharesJSON {
        iris_version: IRIS_VERSION.to_string(),
        iris_shares_version: IRIS_SHARES_VERSION.to_string(),
        right_iris_code_shares: r_code[i].to_base64(),
        right_mask_code_shares: r_mask[i].to_base64(),
        left_iris_code_shares: l_code[i].to_base64(),
        left_mask_code_shares: l_mask[i].to_base64(),
    })
}

/// Converts iris code shares into a JSON representation.
#[allow(dead_code)]
pub fn to_iris_party_shares_for_s3(
    shares: &IrisCodePartyShares,
    encryption_public_keys: &[PublicKey; N_PARTIES],
) -> SharesS3Object {
    let mut hash_set: [String; N_PARTIES] = Default::default();
    let mut content_set: [String; N_PARTIES] = Default::default();

    for i in 0..N_PARTIES {
        let as_obj = shares.party(i);
        let as_json = serde_json::to_string(as_obj)
            .expect("Serialization failed")
            .clone();
        let as_bytes = sealedbox::seal(as_json.as_bytes(), &encryption_public_keys[i]);

        content_set[i] = b64.encode(&as_bytes);
        hash_set[i] = sha256_as_hex_string(&as_json);
    }

    SharesS3Object {
        iris_share_0: content_set[0].clone(),
        iris_share_1: content_set[1].clone(),
        iris_share_2: content_set[2].clone(),
        iris_hashes_0: hash_set[0].clone(),
        iris_hashes_1: hash_set[1].clone(),
        iris_hashes_2: hash_set[2].clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{to_iris_code_party_shares, to_iris_code_shares_json, to_iris_party_shares_for_s3};
    use crate::{
        constants::N_PARTIES, irises::generate_iris_code_and_mask_shares_for_both_eyes,
        types::IrisCodeAndMaskShares,
    };
    use iris_mpc_cpu::execution::hawk_main::BothEyes;
    use rand::{rngs::StdRng, SeedableRng};
    use sodiumoxide::crypto::box_::{gen_keypair, PublicKey};

    fn create_iris_code_and_mask_shares_for_both_eyes() -> BothEyes<IrisCodeAndMaskShares> {
        let mut rng = create_rng();
        generate_iris_code_and_mask_shares_for_both_eyes(&mut rng)
    }

    fn create_public_keys_for_encryption() -> [PublicKey; N_PARTIES] {
        std::array::from_fn(|_| gen_keypair().0)
    }

    fn create_rng() -> StdRng {
        StdRng::from_entropy()
    }

    #[test]
    fn test_convert_to_iris_code_party_shares() {
        let [l, r] = create_iris_code_and_mask_shares_for_both_eyes();
        let [l_code, l_mask] = l;
        let [r_code, r_mask] = r;
        let _ = to_iris_code_party_shares(l_code, l_mask, r_code, r_mask, None);
    }

    #[test]
    fn test_convert_to_iris_code_shares_json() {
        let [l, r] = create_iris_code_and_mask_shares_for_both_eyes();
        let [l_code, l_mask] = l;
        let [r_code, r_mask] = r;
        let _ = to_iris_code_shares_json(l_code, l_mask, r_code, r_mask);
    }

    #[test]
    fn test_to_iris_party_shares_for_s3() {
        let [l, r] = create_iris_code_and_mask_shares_for_both_eyes();
        let [l_code, l_mask] = l;
        let [r_code, r_mask] = r;
        let shares = to_iris_code_party_shares(l_code, l_mask, r_code, r_mask, None);
        let keys = create_public_keys_for_encryption();

        let _ = to_iris_party_shares_for_s3(&shares, &keys);
    }
}
