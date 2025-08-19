#!/usr/bin/env bash

set -e

CONNECTION_STRING="$1"

DATABASE="iris"
COLLECTIONS="mpcv2.hnsw.results mpcv2.hnsw.results.partial mpcv2.hnsw.results.deletion mpcv2.hnsw.results.anonymized_statistics mpcv2.hnsw.reauth.results mpcv2.hnsw.reauth.results.partial mpcv2.hnsw.reset.results mpcv2.hnsw.reset.results.partial mpcv2.hnsw.reset.results.update_acks"

for COLLECTION in $COLLECTIONS; do
    echo "Deleting collection $COLLECTION from database $DATABASE..."
    mongosh "$CONNECTION_STRING" --eval "db.getCollection('$COLLECTION').deleteMany({})"
done

echo "MPC collections deleted successfully."
