use base64::prelude::{Engine, BASE64_STANDARD};
use clap::Parser;
use data_encoding::HEXLOWER;
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};
use rand::{prelude::StdRng, SeedableRng};
use ring::digest::{digest, SHA256};
use serde::{ser::Error, Serialize, Serializer};
use serde_big_array::BigArray;
use std::collections::BTreeMap;

const RNG_SEED: u64 = 42; // Replace with your seed value
const IRIS_VERSION: &str = "1.0";
const IRIS_MPC_VERSION: &str = "1.0";

#[derive(Serialize)]
pub struct SerializeWithSortedKeys<T: Serialize>(#[serde(serialize_with = "sorted_keys")] pub T);

fn sorted_keys<T: Serialize, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serde_json::to_value(value)
        .map_err(Error::custom)?
        .serialize(serializer)
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct IrisCodeSharesJson {
    iris_version:           String,
    iris_shares_version:    String,
    left_iris_code_shares:  String,
    left_mask_code_shares:  String,
    right_iris_code_shares: String,
    right_mask_code_shares: String,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct IrisCodeSharesFileOutput {
    iris_code_shares_1:      String,
    iris_code_shares_2:      String,
    iris_code_shares_3:      String,
    iris_code_shares_1_hash: String,
    iris_code_shares_2_hash: String,
    iris_code_shares_3_hash: String,
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    iris_b64_left: Option<String>,

    #[arg(long)]
    mask_b64_left: Option<String>,

    #[arg(long)]
    iris_b64_right: Option<String>,

    #[arg(long)]
    mask_b64_right: Option<String>,

    #[arg(short, long, env)]
    rng_seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let mut rng = if let Some(seed_rng) = args.rng_seed {
        StdRng::seed_from_u64(seed_rng)
    } else {
        StdRng::seed_from_u64(RNG_SEED)
    };

    let iris_code_left = if let Some(iris_base_64) = args.iris_b64_right {
        IrisCodeArray::from_base64(&iris_base_64).unwrap()
    } else {
        IrisCodeArray::random_rng(&mut rng)
    };

    let mask_code_left = if let Some(mask_base_64) = args.mask_b64_right {
        IrisCodeArray::from_base64(&mask_base_64).unwrap()
    } else {
        IrisCodeArray::default()
    };

    let iris_code_right = if let Some(iris_base_64) = args.iris_b64_left {
        IrisCodeArray::from_base64(&iris_base_64).unwrap()
    } else {
        IrisCodeArray::random_rng(&mut rng)
    };

    let mask_code_right = if let Some(mask_base_64) = args.mask_b64_left {
        IrisCodeArray::from_base64(&mask_base_64).unwrap()
    } else {
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
            left_mask_code_shares:  lm.into(),
            right_iris_code_shares: ri.into(),
            right_mask_code_shares: rm.into(),
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
