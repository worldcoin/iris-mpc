---
name: update-genesis-height
description: Increase GENESIS_MAX_HEIGHT across all three HNSW node dev deploy configs
argument-hint: [increment]
---

Increase the `GENESIS_MAX_HEIGHT` environment variable value in all three dev HNSW deploy config files.

If an argument is provided (`$ARGUMENTS`), use it as the increment. Otherwise, default to an increment of **1000**.

## Steps

1. Read the current value of `GENESIS_MAX_HEIGHT` from each file:
   - `deploy/dev/ampc-hnsw-0-dev/values-ampc-hnsw.yaml`
   - `deploy/dev/ampc-hnsw-1-dev/values-ampc-hnsw.yaml`
   - `deploy/dev/ampc-hnsw-2-dev/values-ampc-hnsw.yaml`

2. Confirm the current value is the same across all three files. If not, warn the user and stop.

3. Compute the new value: `current + increment`.

4. Update the `GENESIS_MAX_HEIGHT` value in all three files. The entry looks like:
   ```yaml
     - name: GENESIS_MAX_HEIGHT
       value: "<height>"
   ```

5. Verify by grepping for `GENESIS_MAX_HEIGHT` across all three files and showing the old and new values.
