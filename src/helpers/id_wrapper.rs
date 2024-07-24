use axum::extract::Path;
use cudarc::nccl::Id;
use std::str::FromStr;

pub struct IdWrapper(pub Id);

impl FromStr for IdWrapper {
    type Err = hex::FromHexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut id: [std::ffi::c_char; 128] = [0; 128];
        hex::decode_to_slice(s, bytemuck::cast_slice_mut(&mut id))?;
        Ok(IdWrapper(Id::uninit(id)))
    }
}

impl std::fmt::Display for IdWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[allow(clippy::unnecessary_cast)]
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
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(ids[device_id]).to_string()
}
