#!/usr/bin/env bash

printf "\n=============================="
printf "\nActivate teleport tunnel to mongo_db iris and delete collections\n"

tsh login --proxy=teleport.worldcoin.dev:443 --auth=okta
tsh db login --db-user developer-read-write mongo-atlas-iris-stage --db-name iris
tsh proxy db --tunnel mongo-atlas-iris-stage -p 60003 --db-user arn:aws:iam::510867353226:role/developer-read-write &

sleep 5

./delete_mongodb_mpcv2_collections.sh "mongodb://127.0.0.1:60003/iris?serverSelectionTimeoutMS=10000"
