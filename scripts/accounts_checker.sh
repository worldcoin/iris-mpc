#!/usr/bin/env bash

ORB_STAGE_ACCOUNT_ID="510867353226"
MPC_1_STAGE_ACCOUNT_ID="767397983205"
MPC_2_STAGE_ACCOUNT_ID="381492197851"
MPC_3_STAGE_ACCOUNT_ID="590184084615"

ACTUAL_ORB_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-stage --query Account --output text)
if [ "$ACTUAL_ORB_ACCOUNT_ID" != "$ORB_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-stage' profile."
    echo "$ACTUAL_ORB_ACCOUNT_ID != $ORB_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_1_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv2-1 --query Account --output text)
if [ "$ACTUAL_MPC_1_ACCOUNT_ID" != "$MPC_1_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv2-1' profile."
    echo "$ACTUAL_MPC_1_ACCOUNT_ID != $MPC_1_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_2_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv2-2 --query Account --output text)
if [ "$ACTUAL_MPC_2_ACCOUNT_ID" != "$MPC_2_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv2-2' profile."
    echo "$ACTUAL_MPC_2_ACCOUNT_ID != $MPC_2_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_3_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv2-3 --query Account --output text)
if [ "$ACTUAL_MPC_3_ACCOUNT_ID" != "$MPC_3_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv2-3' profile."
    echo "$ACTUAL_MPC_3_ACCOUNT_ID != $MPC_3_STAGE_ACCOUNT_ID"
    exit 1
fi

echo "Accounts check succeeded, we are running on staging accounts!"
