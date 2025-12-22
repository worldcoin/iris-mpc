#!/usr/bin/env bash

HOSTS=("aws0" "aws1" "aws2")

echo "Killing local processes..."
pkill -9 -f "run-all-minimal-servers" 2>/dev/null || true
pkill -9 -f "run-minimal-server" 2>/dev/null || true

echo "Killing remote servers..."
for host in "${HOSTS[@]}"; do
  echo "  $host..."
  ssh "ec2-user@${host}" "pkill -9 -f minimal-hawk-server" 2>/dev/null || true
done

echo "Done."

