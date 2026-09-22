use alkali::asymmetric::seal::curve25519xsalsa20poly1305 as sealedbox;
use base64::{engine::general_purpose::STANDARD as b64, Engine};
use eyre::{Result, WrapErr};
use uuid::Uuid;

use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::helpers::{
    sha256::sha256_as_hex_string,
    smpc_request::{IrisCodeSharesJSON, SharesS3Object},
};
use iris_mpc_cpu::execution::hawk_main::{BothEyes, LEFT as LEFT_EYE, RIGHT as RIGHT_EYE};

use crate::{
    constants::N_PARTIES, irises::GaloisRingSharedIrisForUpload, misc::encode_b64,
    types::PublicKeyset,
};

/// TODO: review use of these constants.
const IRIS_VERSION: &str = "1.0";
const IRIS_SHARES_VERSION: &str = "1.3";

/// Converts iris code shares into a representation to be dispatched to an S3 bucket.
pub fn create_iris_code_shares(
    signup_id: &Uuid,
    shares: &BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>,
) -> IrisCodePartyShares {
    IrisCodePartyShares::new(
        signup_id.to_string(),
        create_iris_code_shares_json(shares).to_vec(),
    )
}

/// Converts iris code shares into a JSON representation.
fn create_iris_code_shares_json(
    shares: &BothEyes<[GaloisRingSharedIrisForUpload; N_PARTIES]>,
) -> [IrisCodeSharesJSON; N_PARTIES] {
    std::array::from_fn(|i| IrisCodeSharesJSON {
        iris_version: IRIS_VERSION.to_string(),
        iris_shares_version: IRIS_SHARES_VERSION.to_string(),
        left_iris_code_shares: encode_b64(&shares[LEFT_EYE][i].code),
        left_mask_code_shares: encode_b64(&shares[LEFT_EYE][i].mask),
        right_iris_code_shares: encode_b64(&shares[RIGHT_EYE][i].code),
        right_mask_code_shares: encode_b64(&shares[RIGHT_EYE][i].mask),
    })
}

/// Serializes and encrypts each party's shares for upload to S3.
///
/// Returns an error if serialization or encryption fails, including when a recipient
/// public key is rejected by libsodium. No partial S3 object is returned.
pub fn create_iris_code_shares_s3(
    shares: &IrisCodePartyShares,
    encryption_keys: &PublicKeyset,
) -> Result<SharesS3Object> {
    let mut hash_set: [String; N_PARTIES] = Default::default();
    let mut content_set: [String; N_PARTIES] = Default::default();
    for i in 0..N_PARTIES {
        let as_json =
            serde_json::to_string(shares.party(i)).wrap_err("Failed to serialize iris shares")?;
        let mut as_bytes = vec![0; as_json.len() + sealedbox::OVERHEAD_LENGTH];
        sealedbox::encrypt(as_json.as_bytes(), &encryption_keys[i], &mut as_bytes)
            .wrap_err_with(|| format!("Failed to encrypt iris shares for party {i}"))?;
        content_set[i] = b64.encode(&as_bytes);
        hash_set[i] = sha256_as_hex_string(&as_json);
    }

    Ok(SharesS3Object {
        iris_share_0: content_set[0].clone(),
        iris_share_1: content_set[1].clone(),
        iris_share_2: content_set[2].clone(),
        iris_hashes_0: hash_set[0].clone(),
        iris_hashes_1: hash_set[1].clone(),
        iris_hashes_2: hash_set[2].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        create_iris_code_shares, create_iris_code_shares_json, create_iris_code_shares_s3,
    };
    use crate::{constants::N_PARTIES, irises::generate_iris_shares_for_upload_both_eyes};
    use alkali::{
        asymmetric::seal::{curve25519xsalsa20poly1305 as sealedbox, SealError},
        AlkaliError,
    };
    use base64::{engine::general_purpose::STANDARD as b64, Engine};
    use iris_mpc_common::helpers::sha256::sha256_as_hex_string;
    use rand::{rngs::StdRng, SeedableRng};
    use uuid::Uuid;

    #[test]
    fn test_create_iris_code_shares() {
        let mut rng = StdRng::from_entropy();
        let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
        let signup_id = Uuid::new_v4();
        let _ = create_iris_code_shares(&signup_id, &shares);
    }

    #[test]
    fn test_create_iris_code_shares_json() {
        let mut rng = StdRng::from_entropy();
        let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
        let _ = create_iris_code_shares_json(&shares);
    }

    #[test]
    fn test_create_iris_code_shares_s3() {
        let mut rng = StdRng::seed_from_u64(42);
        let keypairs: [_; N_PARTIES] =
            std::array::from_fn(|_| sealedbox::Keypair::generate().unwrap());
        let keys = std::array::from_fn(|i| keypairs[i].public_key);
        let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
        let signup_id = Uuid::new_v4();
        let shares_1 = create_iris_code_shares(&signup_id, &shares);
        let result = create_iris_code_shares_s3(&shares_1, &keys).unwrap();
        let contents = [
            result.iris_share_0,
            result.iris_share_1,
            result.iris_share_2,
        ];
        let hashes = [
            result.iris_hashes_0,
            result.iris_hashes_1,
            result.iris_hashes_2,
        ];

        for i in 0..N_PARTIES {
            let expected = serde_json::to_string(shares_1.party(i)).unwrap();
            let encrypted = b64.decode(&contents[i]).unwrap();
            assert_eq!(encrypted.len(), expected.len() + sealedbox::OVERHEAD_LENGTH);
            let mut decrypted = vec![0; expected.len()];
            let written = sealedbox::decrypt(&encrypted, &keypairs[i], &mut decrypted).unwrap();
            assert_eq!(written, expected.len());
            assert_eq!(decrypted, expected.as_bytes());
            assert_eq!(hashes[i], sha256_as_hex_string(&expected));
        }
    }

    #[test]
    fn test_create_iris_code_shares_s3_rejects_invalid_recipient() {
        let mut rng = StdRng::seed_from_u64(42);
        let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
        let shares = create_iris_code_shares(&Uuid::new_v4(), &shares);
        let valid_key = sealedbox::Keypair::generate().unwrap().public_key;

        for party in 0..N_PARTIES {
            let mut keys = [valid_key; N_PARTIES];
            keys[party] = [0; 32];
            let error = create_iris_code_shares_s3(&shares, &keys).unwrap_err();
            assert_eq!(
                error.downcast_ref::<AlkaliError>(),
                Some(&AlkaliError::SealError(SealError::PublicKeyUnacceptable))
            );
            assert_eq!(
                error.to_string(),
                format!("Failed to encrypt iris shares for party {party}")
            );
        }
    }
}
