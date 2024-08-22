use serde::{
    ser::{Error, Serializer},
    Serialize,
};

#[derive(Serialize)]
pub struct SerializeWithSortedKeys<T: Serialize>(#[serde(serialize_with = "sorted_keys")] pub T);

fn sorted_keys<T: Serialize, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serde_json::to_value(value)
        .map_err(Error::custom)?
        .serialize(serializer)
}
