# Key Manager CLI

The Key Manager CLI is a command line interface to rotate public and private keys used to encode shares.
The initial private key is generated using `smpc-setup`, and it is empty.

Key manager must be run from each of the participant accounts at least once before initiating the protocol.

Keys can be rotated at any time using the `rotate` command.

## Usage

```bash
>>> key-manager --node-id 2 --env prod rotate --public-key-bucket-name wf-env-stage-public-keys
```

This will:

1. Update the public key in the bucket `wf-env-stage-public-keys` for node 2.
2. Generate a new private key and store aws secrets manager under the secret name: `prod/iris-mpc/ecdh-private-key-2`

This key will be immediately valid, though the previous key will retain a validity of 24 hours (dictated by the cloudfront caching behavior, 
and by application logic that checks against AWSCURRENT and AWSPREVIOUS version of the secret).

# Migrator

Minimal wrapper on top of the `sqlx` to migrate the database in dev mode. Was implemented because of issues with running `sqlx migrate run` 
into a custom schema.
