#!/usr/bin/env bash

export ORB_STAGE_ACCOUNT_ID="510867353226"
export MPC_0_STAGE_ACCOUNT_ID="024848486749"
export MPC_1_STAGE_ACCOUNT_ID="024848486818"
export MPC_2_STAGE_ACCOUNT_ID="024848486770"
export MPC_VPC_STAGE_ACCOUNT_ID="302263054573"

ACTUAL_ORB_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-stage --query Account --output text)
if [ "$ACTUAL_ORB_ACCOUNT_ID" != "$ORB_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-stage' profile."
    echo "$ACTUAL_ORB_ACCOUNT_ID != $ORB_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_VPC_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv-io-vpc --query Account --output text)
if [ "$ACTUAL_MPC_VPC_ACCOUNT_ID" != "$MPC_VPC_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv-io-vpc' profile."
    echo "$ACTUAL_MPC_VPC_ACCOUNT_ID != $MPC_VPC_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_0_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv-io-0 --query Account --output text)
if [ "$ACTUAL_MPC_0_ACCOUNT_ID" != "$MPC_0_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv-io-0' profile."
    echo "$ACTUAL_MPC_0_ACCOUNT_ID != $MPC_0_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_1_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv-io-1 --query Account --output text)
if [ "$ACTUAL_MPC_1_ACCOUNT_ID" != "$MPC_1_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv-io-1' profile."
    echo "$ACTUAL_MPC_1_ACCOUNT_ID != $MPC_1_STAGE_ACCOUNT_ID"
    exit 1
fi

ACTUAL_MPC_2_ACCOUNT_ID=$(aws sts get-caller-identity --profile worldcoin-smpcv-io-2 --query Account --output text)
if [ "$ACTUAL_MPC_2_ACCOUNT_ID" != "$MPC_2_STAGE_ACCOUNT_ID" ]; then
    echo "The actual account ID does not match the expected account ID for the 'worldcoin-smpcv-io-2' profile."
    echo "$ACTUAL_MPC_2_ACCOUNT_ID != $MPC_2_STAGE_ACCOUNT_ID"
    exit 1
fi

echo "Accounts check succeeded, we are running on staging accounts!"
