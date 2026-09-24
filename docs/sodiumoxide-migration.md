# Sodiumoxide replacement: prepared, blocked on ampc-common

Status as of 2026-09-24: the direct consumers in this workspace have been
migrated to alkali 0.3. This is a **dependency-blocked draft**, not a validated
release. Do not merge or deploy it until the dependency and validation steps
below are complete. The ampc-common migration is owned separately.

## Prepared changes

| Package | Change |
| --- | --- |
| `iris-mpc-common` | Import existing private keys into retained hardened keypairs; borrow them for decryption and preserve current/previous-key fallback. |
| `iris-mpc` | Migrate client encryption and key-rotation test helpers; replace sodiumoxide hex formatting with the existing hex crate. |
| `iris-mpc-utils` | Migrate public-key parsing and share encryption; propagate encryption failures before any upload. |
| `iris-mpc-bins` | Migrate key-manager generation/import/validation and its tests; replace sodiumoxide hex formatting in the anonymous-statistics server. |

Other workspace packages have no direct sodiumoxide consumers. The obsolete
RUSTSEC-2021-0137 exception is removed. No replacement advisory exception,
local dependency patch, vendored fork, or ampc-common edit is included.

Alkali uses the same native Curve25519/XSalsa20-Poly1305 sealed-box construction:
32-byte public/private keys, unchanged base64 representation, 48-byte ciphertext
overhead, and unchanged native seed-to-key derivation. Fresh ephemeral keys make
new ciphertext random; compatibility means successful cross-decryption, not
identical ciphertext from two independently generated boxes. Existing keys do not
need rotation merely because the binding changes.

`create_iris_code_shares_s3` now returns `eyre::Result<SharesS3Object>`. Its caller
propagates failure before uploading. Low-order recipient keys produce a native
error rather than unchecked ciphertext. Authentication failure may try the
previous key; initialization/allocation failures are propagated instead of being
misreported as a wrong key. Key-manager malformed-input/mismatch paths return
errors without displaying private-key bytes.

Common's direct alkali dependency is optional under the existing `helpers`
feature. This does not promise a crypto-free transitive dependency graph: other
dependencies may still use a binding. No global `use-pkg-config` feature is added;
native-library provisioning continues to use the binding's default build behavior.

Key storage holds an alkali `Keypair` directly; decryption no longer copies the
private bytes into a new native allocation on each call. Key types no longer
implement `Clone`. GPU/HNSW servers, batch processing and modification-sync tasks
share an immutable `Arc<SharesEncryptionKeyPairs>` instead, cloning only ownership
handles. `decrypt_iris_share` now borrows `&SharesEncryptionKeyPairs`.

The keys are wiped when the final owning handle is dropped, including task
cancellation. This does not implement immediate key revocation: dropping one
handle cannot invalidate keys still in use by another task. Explicit `Zeroize`
requires exclusive access to the key set (for shared ownership, a successful
`Arc::get_mut`). No lock or mutable shared key storage is introduced.

Production key-manager rotation uses `Keypair::generate()`; deterministic
`from_seed()` remains only in tests. Public-key-only derivation uses
`PrivateKey::public_key()` without cloning the secret into a temporary keypair.
All crypto calls keep explicit `curve25519xsalsa20poly1305` imports.

Debug output omits the secret; decoded private-key buffers and failed plaintext
output buffers in the share-decryption helper are wiped on drop.
This is not a guarantee of complete memory erasure: caller copies, allocator
reallocations, AWS SDK buffers, returned plaintext and process aborts remain
outside that guarantee.

## Tests prepared

- Frozen native-libsodium synthetic ciphertext opens with an imported private key.
- Native deterministic seed-derived public/private bytes and base64 round trips.
- Empty plaintext, malformed base64 and private-key lengths.
- Wrong recipient, tampering at every ciphertext byte and every truncated prefix.
- Shared hardened storage across tasks, last-owner release, cancellation cleanup,
  explicit zeroization and secret-free Debug output.
- Generated production-style keys survive base64 export/import and decrypt the
  original ciphertext; seeded native compatibility vectors remain covered.
- Current-key decryption without a previous key; actual previous-key fallback.
- Encryption/decryption and JSON/hash equality for all three share recipients.
- Invalid recipient rejection at each party index, including the native error.

Fixtures added by this migration use public synthetic seeds/messages, not real
biometric payloads or operational keys. The ignored key-manager test that needed
an external biometric file is replaced by self-contained synthetic coverage.

## Validation status and blocker

Formatting (`cargo +1.95.0 fmt --all -- --check`) and `git diff --check` pass.
An offline focused test attempt also encountered an uncached `relative-path`
dependency before reaching the native-link conflict; this is not a test failure
or evidence that the migration compiles.
Compilation, Clippy and Rust test execution are **not validated**: Cargo stops at
dependency resolution, before compiling the changed code:

```text
iris-mpc-bins -> ampc-server-utils
  (ampc-common rev 867671cff92a82f0208afc1f41e619cf360596ff)
  -> sodiumoxide 0.2.7 -> libsodium-sys (links = "sodium")

alkali 0.3 -> libsodium-sys-stable (links = "sodium")

only one package in the dependency graph may specify the same links value
```

The pinned ampc-server-utils dependency is unconditional; disabling default
features does not remove it. `Cargo.lock` is intentionally unchanged, still
contains sodiumoxide, and does **not** represent the new manifests yet. Editing
the lockfile by hand would hide the blocker, not fix it.

## Resume after ampc-common is fixed

1. Obtain the **merged full commit SHA** from the ampc-common owner. Verify its
   resolved graph no longer contains sodiumoxide or the old libsodium-sys, and
   that its binding is compatible with alkali 0.3 / libsodium-sys-stable. A fix on
   an unmerged branch is insufficient: this repository enforces main ancestry.
2. Update all four ampc-common workspace dependencies together in `Cargo.toml`
   (`ampc-anon-stats`, `ampc-actor-utils`, `ampc-secret-sharing`,
   `ampc-server-utils`). Regenerate the lockfile with Cargo and review the diff:

   ```sh
   cargo metadata --format-version 1 > /dev/null
   bash scripts/check-ampc-common-pin.sh
   cargo tree --locked --workspace --all-features -i alkali
   rg -n 'name = "(sodiumoxide|libsodium-sys)"' Cargo.lock
   ```

   The final search must have no matches (rg exit status 1). Do not accept a
   manifest-only migration or retain the deprecated advisory exception.
3. Run the focused checks below in the normal Linux development/CI environment
   with its required native dependencies. Fix compilation failures before claiming
   that the prepared tests pass:

   ```sh
   cargo fmt --all -- --check
   cargo check --locked -p iris-mpc-common --lib --no-default-features
   cargo test --locked -p iris-mpc-common --lib helpers::key_pair::tests
   cargo test --locked -p iris-mpc-common --test smpc_request
   cargo test --locked -p iris-mpc-utils --lib aws::factory::tests
   cargo test --locked -p iris-mpc-bins --bin key-manager
   cargo check --locked -p iris-mpc -p iris-mpc-utils -p iris-mpc-bins --all-targets
   cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
   cargo deny --locked check advisories licenses
   ```

4. Run the existing full workspace build/tests and CI jobs, including configured
   database/localstack, CPU key-rotation E2E and CUDA-dependent checks. Do not run
   key-manager rotation against production to validate this change. Confirm that
   a legacy-produced ciphertext decrypts under the new service and new client
   ciphertext decrypts under a legacy service, including rotation overlap.
   Measure share-decryption throughput and allocation cost under representative
   load, confirming that retained hardened keys avoid per-decryption key imports.
   Compare current-key success and previous-key fallback against the old service.
5. Re-review the final dependency and source diff for secrets, update this status
   with actual validation results, and only then submit/merge the migration.
   Roll out through the existing staging path; keep the previous image available
   for rollback. This migration does not change stored key material or wire format.

Consumers pinned to older iris-mpc revisions must separately assess intervening
API/dependency changes. This work does not establish that upgrading an old pin
directly to current main is safe.
