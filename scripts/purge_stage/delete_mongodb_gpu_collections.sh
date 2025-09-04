#!/usr/bin/env bash

set -e

CONNECTION_STRING="$1"

DATABASE="iris"
COLLECTIONS="mpcv2.results mpcv2.results.partial mpcv2.results.deletion mpcv2.results.anonymized_statistics mpcv2.results.anonymized_statistics_2d mpcv2.reauth.results mpcv2.reauth.results.partial mpcv2.reset.results mpcv2.reset.results.partial mpcv2.reset.results.update_acks"

for COLLECTION in $COLLECTIONS; do
    echo "Deleting collection $COLLECTION from database $DATABASE..."
    mongosh "$CONNECTION_STRING" --eval "db.getCollection('$COLLECTION').deleteMany({})"
done

echo "MPC collections deleted successfully."
