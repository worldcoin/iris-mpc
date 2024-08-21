//! Helpers to serialize a Rust `struct` with all keys sorted alphabetically.
//!
//! ```
//! use iris_mpc_common::helpers::serialize_with_sorted_keys::SerializeWithSortedKeys;
//!
//! #[derive(serde::Serialize)]
//! struct Foo {
//!     d: usize,
//!     c: usize,
//!     a: usize,
//!     b: usize,
//! }
//!
//! let foo = Foo {
//!     c: 3,
//!     b: 2,
//!     a: 1,
//!     d: 4,
//! };
//!
//! // By default serde serialized the keys in the order in which they were defined.
//! assert_eq!(
//!     serde_json::to_string(&foo).unwrap(),
//!     r#"{"d":4,"c":3,"a":1,"b":2}"#
//! );
//!
//! // We can sort the keys alphabetically with this little helper.
//! assert_eq!(
//!     serde_json::to_string(&SerializeWithSortedKeys(&foo)).unwrap(),
//!     r#"{"a":1,"b":2,"c":3,"d":4}"#
//! );
//! ```

use serde::{
    ser::{Error, Serializer},
    Serialize,
};

/// Wrapper type for serializing a `struct` with all keys sorted alhpabetically.
///
/// See [the module-level documentation](self) for details.
#[derive(Serialize)]
pub struct SerializeWithSortedKeys<T: Serialize>(#[serde(serialize_with = "sorted_keys")] pub T);

fn sorted_keys<T: Serialize, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serde_json::to_value(value)
        .map_err(Error::custom)?
        .serialize(serializer)
}
