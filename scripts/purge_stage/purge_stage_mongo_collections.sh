#!/usr/bin/env bash

if [ -z "$1" ]; then
  printf "\nError: Cleanup type parameter is required\n"
  printf "Usage: %s <cleanup_type>\n" "$0"
  printf "Available cleanup types: gpu, cpu\n"
  exit 1
fi

CLEANUP_TYPE=$1

printf "\n=============================="
printf "\nActivate teleport tunnel to mongo_db iris and delete collections (%s)\n" "$CLEANUP_TYPE"

# Login to teleport
tsh login --proxy=teleport.worldcoin.dev:443 --auth=okta
tsh db login --db-user developer-read-write mongo-atlas-iris-stage --db-name iris
tsh proxy db --tunnel mongo-atlas-iris-stage -p 60003 --db-user arn:aws:iam::510867353226:role/developer-read-write &

# Wait for proxy connection to establish
sleep 5

# MongoDB connection string
MONGO_URI="mongodb://127.0.0.1:60003/iris?serverSelectionTimeoutMS=10000"

# Select the appropriate cleanup script based on the cleanup type
case "$CLEANUP_TYPE" in
  "gpu")
    printf "\nRunning GPU collections cleanup script\n"
    ./delete_mongodb_gpu_collections.sh "$MONGO_URI"
    ;;
  "cpu")
    printf "\nRunning CPU collections cleanup script\n"
    ./delete_mongodb_cpu_collections.sh "$MONGO_URI"
    ;;
  *)
    printf "\nUnknown cleanup type: %s\n" "$CLEANUP_TYPE"
    printf "Available cleanup types: gpu, cpu\n"
    exit 1
    ;;
esac

printf "\nCleanup completed successfully\n"
