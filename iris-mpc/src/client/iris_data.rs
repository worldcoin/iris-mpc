#![allow(clippy::needless_range_loop)]
use eyre::Context;
use eyre::Result;
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, helpers::smpc_request::IrisCodeSharesJSON,
    iris_db::iris::IrisCode,
};
use rand::rngs::StdRng;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct EnrollmentIrisData {
    #[serde(rename = "signup_id")]
    pub signup_id: String,
    #[serde(rename = "iris_code_shares_left")]
    pub iris_codes_shares_left: Vec<String>,
    #[serde(rename = "iris_code_shares_right")]
    pub iris_codes_shares_right: Vec<String>,
    #[serde(rename = "mask_code_shares_left")]
    pub iris_mask_shares_left: Vec<String>,
    #[serde(rename = "mask_code_shares_right")]
    pub iris_mask_shares_right: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IrisCodePartyShares {
    pub signup_id: String,
    pub parties: Vec<IrisCodeSharesJSON>,
}

impl IrisCodePartyShares {
    pub fn new(signup_id: String, parties: Vec<IrisCodeSharesJSON>) -> Self {
        assert_eq!(
            parties.len(),
            3,
            "IrisCodePartyShares must have exactly 3 parties"
        );
        Self { signup_id, parties }
    }

    pub fn party(&self, index: usize) -> &IrisCodeSharesJSON {
        &self.parties[index]
    }

    pub fn create_duplicate_party_shares(&self, signup_id: String) -> IrisCodePartyShares {
        IrisCodePartyShares::new(signup_id, self.parties.clone())
    }
}

pub async fn read_iris_data_from_file(file_name: &str) -> Result<Vec<IrisCodePartyShares>> {
    let mut file = File::open(file_name).context(format!("Failed to open file: {}", file_name))?;

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .context(format!("Failed to read file: {}", file_name))?;

    // Parse the array of enrollment data
    let enrollment_data: Vec<EnrollmentIrisData> =
        serde_json::from_slice(&bytes).context("Failed to parse enrollment iris data array")?;

    let batch_size = enrollment_data.len();
    let mut result = Vec::with_capacity(batch_size);

    // Process each entry in the batch
    for entry_index in 0..batch_size {
        let shares_entry = &enrollment_data[entry_index];
        let mut shares = Vec::new();
        for i in 0..3 {
            shares.push(IrisCodeSharesJSON {
                iris_version: "1.0".to_string(),
                iris_shares_version: "1.3".to_string(),
                right_iris_code_shares: shares_entry.iris_codes_shares_right[i].clone(),
                right_mask_code_shares: shares_entry.iris_mask_shares_right[i].clone(),
                left_iris_code_shares: shares_entry.iris_codes_shares_left[i].clone(),
                left_mask_code_shares: shares_entry.iris_mask_shares_left[i].clone(),
            });
        }
        result.push(IrisCodePartyShares::new(
            shares_entry.signup_id.clone(),
            shares,
        ));
    }

    Ok(result)
}

pub fn generate_party_shares(mut rng: StdRng) -> IrisCodePartyShares {
    let template_left = IrisCode::random_rng(&mut rng);
    let template_right = IrisCode::random_rng(&mut rng);

    let left_shared_code = GaloisRingIrisCodeShare::encode_iris_code(
        &template_left.code,
        &template_left.mask,
        &mut rng,
    );
    let left_shared_mask = GaloisRingIrisCodeShare::encode_mask_code(&template_left.mask, &mut rng);
    let right_shared_code = GaloisRingIrisCodeShare::encode_iris_code(
        &template_right.code,
        &template_right.mask,
        &mut rng,
    );
    let right_shared_mask =
        GaloisRingIrisCodeShare::encode_mask_code(&template_right.mask, &mut rng);

    // make a vector of IrisCodeSharesJSON that is empty and then fill it with the IrisCodeSharesJSON
    let mut shares = Vec::new();
    for i in 0..3 {
        shares.push(IrisCodeSharesJSON {
            iris_version: "1.0".to_string(),
            iris_shares_version: "1.3".to_string(),
            right_iris_code_shares: right_shared_code[i].to_base64(),
            right_mask_code_shares: right_shared_mask[i].to_base64(),
            left_iris_code_shares: left_shared_code[i].to_base64(),
            left_mask_code_shares: left_shared_mask[i].to_base64(),
        });
    }
    let request_id = Uuid::new_v4().to_string();
    IrisCodePartyShares::new(request_id, shares)
}
