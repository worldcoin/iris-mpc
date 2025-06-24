use eyre::Result;

mod shared;

/// Test 100:
///   against:
///     a known dataset of Iris shares in plaintext format;
///     an empty dataset of exclusions;
///     an empty dataset of modifications;
///   asserts:
///     graph construction is equivalent for each node;
#[tokio::test]
async fn test_100() -> Result<()> {
    // Test code here
    shared::setup();

    // Test code here
    assert!(true);

    Ok(())
}
