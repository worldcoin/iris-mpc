use axum::extract::Path;
use cudarc::nccl::Id;
use std::str::FromStr;

pub struct IdWrapper(pub Id);

impl FromStr for IdWrapper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s)
            .unwrap()
            .iter()
            .map(|&c| c as i8)
            .collect::<Vec<_>>();

        let mut id = [0i8; 128];
        id.copy_from_slice(&bytes);

        Ok(IdWrapper(Id::uninit(id)))
    }
}

impl std::fmt::Display for IdWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode(
            self.0
                .internal()
                .iter()
                .map(|&c| c as u8)
                .collect::<Vec<_>>(),
        ))
    }
}

pub async fn http_root(ids: Vec<Id>, Path(device_id): Path<String>) -> String {
    // let device_id: usize = device_id.parse().unwrap();
    // IdWrapper(ids[device_id]).to_string()
    format!("2f671c856cc744b00200aa{:02x}0a0f201b0000000000000000000000000000000000000000000000000400000000000000906b0385fe7f0000434ed33384550000a8640000000000000000000000000000b022d83a845500000002000000000000b8600385fe7f000000000000000000000700000000000000405e0385fe7f0000", device_id)
}
