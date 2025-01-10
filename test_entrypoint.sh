#!/usr/bin/env bash
set -euo pipefail

# Function that prints the date on exit
on_exit() {
  echo "[Container End] $(date)  :: Container is stopping..."
}

# Register the function above to run when the script exits
trap on_exit EXIT

# Print a timestamp when container starts
echo "[Container Start] $(date)  :: Container is starting..."

# Execute the main process
exec "$@"
