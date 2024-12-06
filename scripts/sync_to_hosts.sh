#!/usr/bin/env bash

# Check if username is provided as a parameter
if [ -z "$1" ]; then
    echo "No username provided."
    echo "Usage: $0 <username>"
    exit 1
fi

# Assign the provided username to a variable
USER_NAME=$1

# Define the target directory
TARGET_DIR="/home/ubuntu/${USER_NAME}"

# Define the hosts, order is important, indices are used to identify .env files
TARGET_HOSTS=(
"ubuntu@mpc1-c4b4574e50dcf9c1.elb.eu-north-1.amazonaws.com"
"ubuntu@mpc2-c2cf3f545a8c3ae3.elb.eu-north-1.amazonaws.com"
"ubuntu@mpc3-9aa224d36eb4357f.elb.eu-north-1.amazonaws.com"
)

# Define the directories to exclude
EXCLUDE_DIRS=(".idea" ".git" "target")

# Create the exclude options for rsync
EXCLUDE_OPTIONS=()
for EXCLUDE_DIR in "${EXCLUDE_DIRS[@]}"; do
    EXCLUDE_OPTIONS+=("--exclude=$EXCLUDE_DIR")
done

# Get the name of the current working directory
CURRENT_DIR_NAME=$(basename "$PWD")

## Loop through each host using an index, starting with 1
for i in "${!TARGET_HOSTS[@]}"; do
   INDEX=$((i + 1))
   HOST=${TARGET_HOSTS[$i]}
   echo "Syncing $CURRENT_DIR_NAME to $HOST:$TARGET_DIR (Target Host $INDEX)"

   ## Make sure that the correct environment file is copied to the correct host
   cp .env.mpc$INDEX.dist .env

   ## Sync the current directory to the target host
   rsync -avz --delete "${EXCLUDE_OPTIONS[@]}" "$PWD" "$HOST:$TARGET_DIR"
   if [ $? -eq 0 ]; then
       echo "Sync to $HOST completed successfully."
   else
       echo "Sync to $HOST failed."
   fi
done
