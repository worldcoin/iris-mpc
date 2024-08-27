use base64::prelude::{Engine, BASE64_STANDARD};
use data_encoding::HEXLOWER;
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};
use rand::{prelude::StdRng, SeedableRng};
use ring::digest::{digest, SHA256};
use serde::{ser::Error, Serialize, Serializer};
use serde_big_array::BigArray;
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
    iris_version:           String,
    #[serde(rename = "IRIS_shares_version")]
    iris_shares_version:    String,
    left_iris_code_shares:  String,
    left_iris_mask_shares:  String,
    right_iris_code_shares: String,
    right_iris_mask_shares: String,
}

/// Iris code shares.
#[allow(clippy::module_name_repetitions)]
pub type IrisCodeShares = [IrisCodeShare; 3];
/// Iris mask code shares.
pub type MaskCodeShares = [IrisCodeShare; 3];

/// Iris code share.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Serialize)]
pub struct IrisCodeShare {
    /// The ID.
    pub id:    usize,
    /// The coefficients.
    #[serde(with = "BigArray")]
    pub coefs: [u16; 12800],
}

impl Default for IrisCodeShare {
    fn default() -> Self {
        Self {
            id:    0,
            coefs: [0; 12800],
        }
    }
}

impl From<GaloisRingIrisCodeShare> for IrisCodeShare {
    fn from(share: GaloisRingIrisCodeShare) -> Self {
        Self {
            id:    share.id,
            coefs: share.coefs,
        }
    }
}

impl From<&IrisCodeShare> for Vec<u8> {
    fn from(val: &IrisCodeShare) -> Self {
        bincode::serialize(&val).expect("to serialize")
    }
}

impl From<&IrisCodeShare> for String {
    fn from(val: &IrisCodeShare) -> Self {
        BASE64_STANDARD.encode::<Vec<u8>>(val.into())
    }
}

fn to_array(input: [GaloisRingIrisCodeShare; 3]) -> [IrisCodeShare; 3] {
    input
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Expected exactly 3 elements")
}

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

    let shares_left = to_array(GaloisRingIrisCodeShare::encode_iris_code(
        &iris_code_left,
        &mask_code_left,
        &mut rng,
    ));
    let masks_left = to_array(GaloisRingIrisCodeShare::encode_mask_code(
        &mask_code_left,
        &mut rng,
    ));
    let shares_right = to_array(GaloisRingIrisCodeShare::encode_iris_code(
        &iris_code_right,
        &mask_code_right,
        &mut rng,
    ));
    let masks_right = to_array(GaloisRingIrisCodeShare::encode_mask_code(
        &mask_code_right,
        &mut rng,
    ));

    let mut iris_code_shares_jsons = Vec::new();
    let mut iris_code_shares_file_output = BTreeMap::new();

    for (i, ((li, lm), (ri, rm))) in shares_left
        .iter()
        .zip(masks_left.iter())
        .zip(shares_right.iter().zip(masks_right.iter()))
        .enumerate()
    {
        let iris_code_shares = IrisCodeSharesJson {
            iris_version:           IRIS_VERSION.to_string(),
            iris_shares_version:    IRIS_MPC_VERSION.to_string(),
            left_iris_code_shares:  li.into(),
            left_iris_mask_shares:  lm.into(),
            right_iris_code_shares: ri.into(),
            right_iris_mask_shares: rm.into(),
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
            iris_version:           IRIS_VERSION.to_string(),
            iris_shares_version:    IRIS_MPC_VERSION.to_string(),
            left_iris_code_shares:  "left_iris_code_shares".to_string(),
            left_iris_mask_shares:  "left_iris_mask_shares".to_string(),
            right_iris_code_shares: "right_iris_code_shares".to_string(),
            right_iris_mask_shares: "right_iris_mask_shares".to_string(),
        };

        let expected = r#"{"IRIS_shares_version":"1.0","IRIS_version":"1.1","left_iris_code_shares":"left_iris_code_shares","left_iris_mask_shares":"left_iris_mask_shares","right_iris_code_shares":"right_iris_code_shares","right_iris_mask_shares":"right_iris_mask_shares"}"#;
        assert_eq!(
            serde_json::to_string(&SerializeWithSortedKeys(&iris_code_shares)).unwrap(),
            expected
        );
    }
}
