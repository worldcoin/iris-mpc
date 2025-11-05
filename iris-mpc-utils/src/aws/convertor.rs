use uuid::Uuid;

use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::helpers::smpc_request::IrisCodeSharesJSON;

use crate::types::{NetGaloisRingIrisCodeShare, NetIrisCodeSharesJSON};

/// TODO: review use of these constants.
const IRIS_VERSION: &str = "1.0";
const IRIS_SHARES_VERSION: &str = "1.3";

/// Converts iris code shares into a JSON representation.
fn to_iris_code_shares_json(
    left_shared_code: NetGaloisRingIrisCodeShare,
    left_shared_mask: NetGaloisRingIrisCodeShare,
    right_shared_code: NetGaloisRingIrisCodeShare,
    right_shared_mask: NetGaloisRingIrisCodeShare,
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

/// Converts iris code shares into a JSON representation.
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
