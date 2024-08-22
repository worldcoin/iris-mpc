mod tests {
    use iris_mpc_common::helpers::serialize_with_sorted_keys::SerializeWithSortedKeys;
    use serde_json;

    #[derive(serde::Serialize)]
    struct Foo {
        d: usize,
        c: usize,
        a: usize,
        b: usize,
    }

    #[test]
    fn test_foo_serialization() {
        let foo = Foo {
            c: 3,
            b: 2,
            a: 1,
            d: 4,
        };

        // By default, serde serializes the keys in the order in which they were
        // defined.
        assert_eq!(
            serde_json::to_string(&foo).unwrap(),
            r#"{"d":4,"c":3,"a":1,"b":2}"#
        );

        // We can sort the keys alphabetically with this little helper.
        assert_eq!(
            serde_json::to_string(&SerializeWithSortedKeys(&foo)).unwrap(),
            r#"{"a":1,"b":2,"c":3,"d":4}"#
        );
    }
}
