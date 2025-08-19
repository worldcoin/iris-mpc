use base64::prelude::{Engine, BASE64_STANDARD};
use data_encoding::HEXLOWER;
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};
use rand::{prelude::StdRng, SeedableRng};
use ring::digest::{digest, SHA256};
use serde::{ser::Error, Serialize, Serializer};
use serde_json::Value;
use std::{collections::BTreeMap, env};

const RNG_SEED: u64 = 42; // Replace with your seed value
const IRIS_VERSION: &str = "1.1";
const IRIS_MPC_VERSION: &str = "1.0";

#[derive(Serialize)]
pub struct SerializeWithSortedKeys<T: Serialize>(#[serde(serialize_with = "sorted_keys")] pub T);

fn sorted_keys<T: Serialize, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    let value = serde_json::to_value(value).map_err(Error::custom)?;

    if let Value::Object(map) = value {
        // Create a BTreeMap which automatically sorts the keys
        let sorted_map: BTreeMap<_, _> = map.into_iter().collect();
        sorted_map.serialize(serializer)
    } else {
        value.serialize(serializer)
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct IrisCodeSharesJson {
    #[serde(rename = "IRIS_version")]
    iris_version: String,
    #[serde(rename = "IRIS_shares_version")]
    iris_shares_version: String,
    left_iris_code_shares: String,
    left_mask_code_shares: String,
    right_iris_code_shares: String,
    right_mask_code_shares: String,
}

/// Iris code shares.
#[allow(clippy::module_name_repetitions)]
pub type IrisCodeShares = [String; 3];
/// Iris mask code shares.
pub type MaskCodeShares = [String; 3];

fn main() {
    let mut rng = if let Ok(seed_rng) = env::var("SEED_RNG") {
        // env variable passed, use passed seed
        StdRng::seed_from_u64(seed_rng.parse().unwrap())
    } else {
        // no env variable passed, use default seed
        StdRng::seed_from_u64(RNG_SEED)
    };

    let iris_code_left = if let Ok(iris_base_64) = env::var("IRIS_B64_LEFT") {
        // env variable passed, use passed iris code
        IrisCodeArray::from_base64(&iris_base_64).unwrap()
    } else {
        // no env variable passed, generate random iris code
        IrisCodeArray::random_rng(&mut rng)
    };

    let mask_code_left = if let Ok(mask_base_64) = env::var("MASK_B64_LEFT") {
        // env variable passed, use passed iris mask
        IrisCodeArray::from_base64(&mask_base_64).unwrap()
    } else {
        // no env variable passed, use default iris mask
        IrisCodeArray::default()
    };

    let iris_code_right = if let Ok(iris_base_64) = env::var("IRIS_B64_RIGHT") {
        // env variable passed, use passed iris code
        IrisCodeArray::from_base64(&iris_base_64).unwrap()
    } else {
        // no env variable passed, generate random iris code
        IrisCodeArray::random_rng(&mut rng)
    };

    let mask_code_right = if let Ok(mask_base_64) = env::var("MASK_B64_RIGHT") {
        // env variable passed, use passed iris mask
        IrisCodeArray::from_base64(&mask_base_64).unwrap()
    } else {
        // no env variable passed, use default iris mask
        IrisCodeArray::default()
    };

    let shares_left =
        GaloisRingIrisCodeShare::encode_iris_code(&iris_code_left, &mask_code_left, &mut rng)
            .map(|x| x.to_base64());

    let masks_left =
        GaloisRingIrisCodeShare::encode_mask_code(&mask_code_left, &mut rng).map(|x| x.to_base64());
    let shares_right =
        GaloisRingIrisCodeShare::encode_iris_code(&iris_code_right, &mask_code_right, &mut rng)
            .map(|x| x.to_base64());
    let masks_right = GaloisRingIrisCodeShare::encode_mask_code(&mask_code_right, &mut rng)
        .map(|x| x.to_base64());

    let mut iris_code_shares_jsons = Vec::new();
    let mut iris_code_shares_file_output = BTreeMap::new();

    for (i, ((li, lm), (ri, rm))) in shares_left
        .iter()
        .zip(masks_left.iter())
        .zip(shares_right.iter().zip(masks_right.iter()))
        .enumerate()
    {
        let iris_code_shares = IrisCodeSharesJson {
            iris_version: IRIS_VERSION.to_string(),
            iris_shares_version: IRIS_MPC_VERSION.to_string(),
            left_iris_code_shares: li.clone(),
            left_mask_code_shares: lm.clone(),
            right_iris_code_shares: ri.clone(),
            right_mask_code_shares: rm.clone(),
        };
        let json_u8 = serde_json::to_string(&SerializeWithSortedKeys(&iris_code_shares))
            .unwrap()
            .into_bytes();

        iris_code_shares_file_output.insert(
            format!("iris_code_shares_hash_{i}"),
            HEXLOWER.encode(digest(&SHA256, &json_u8).as_ref()),
        );
        iris_code_shares_file_output.insert(
            format!("iris_code_shares_{i}"),
            BASE64_STANDARD.encode(&json_u8),
        );
        iris_code_shares_jsons.push(json_u8);
    }

    // write iris_code_shares_file_output to file
    let json_data =
        serde_json::to_string(&iris_code_shares_file_output).expect("Serialization failed");
    println!("{}", json_data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iris_code_shares_json() {
        let iris_code_shares = IrisCodeSharesJson {
            iris_version: IRIS_VERSION.to_string(),
            iris_shares_version: IRIS_MPC_VERSION.to_string(),
            left_iris_code_shares: "left_iris_code_shares".to_string(),
            left_mask_code_shares: "left_mask_code_shares".to_string(),
            right_iris_code_shares: "right_iris_code_shares".to_string(),
            right_mask_code_shares: "right_mask_code_shares".to_string(),
        };

        let expected = r#"{"IRIS_shares_version":"1.0","IRIS_version":"1.1","left_iris_code_shares":"left_iris_code_shares","left_mask_code_shares":"left_mask_code_shares","right_iris_code_shares":"right_iris_code_shares","right_mask_code_shares":"right_mask_code_shares"}"#;
        assert_eq!(
            serde_json::to_string(&SerializeWithSortedKeys(&iris_code_shares)).unwrap(),
            expected
        );
    }
}
