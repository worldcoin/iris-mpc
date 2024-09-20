#!/usr/bin/env bash

set -e

CONNECTION_STRING="$1"

DATABASE="iris"
COLLECTIONS="mpcv2.results mpcv2.results.partial mpcv2.results.deletion deletion.requests"

for COLLECTION in $COLLECTIONS; do
    echo "Deleting collection $COLLECTION from database $DATABASE..."
    mongosh "$CONNECTION_STRING" --eval "db.getCollection('$COLLECTION').drop()"
done

echo "MPC collections deleted successfully."
