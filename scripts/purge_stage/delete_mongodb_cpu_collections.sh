#!/usr/bin/env bash

set -e

CONNECTION_STRING="$1"

DATABASE="iris"
COLLECTIONS="mpcv2.cpu.results mpcv2.cpu.results.partial mpcv2.cpu.results.deletion mpcv2.cpu.results.anonymized_statistics"

for COLLECTION in $COLLECTIONS; do
    echo "Deleting collection $COLLECTION from database $DATABASE..."
    mongosh "$CONNECTION_STRING" --eval "db.getCollection('$COLLECTION').deleteMany({})"
done

echo "MPC collections deleted successfully."
