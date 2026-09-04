#!/usr/bin/env python3
"""Create the Moto resources needed by the real iris-mpc server benchmark."""

from __future__ import annotations

import argparse
import json

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "endpoint_url": args.endpoint,
        "region_name": args.region,
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
    }
    s3 = boto3.client("s3", **common)
    sns = boto3.client("sns", **common)
    sqs = boto3.client("sqs", **common)
    secrets = boto3.client("secretsmanager", **common)

    for bucket in (
        "wf-dev-public-keys",
        "wf-smpcv2-dev-sns-requests",
        "wf-smpcv2-dev-sync-protocol",
        "wf-smpcv2-dev-hnsw-performance-reports",
        "wf-smpcv2-dev-hnsw-checkpoint",
    ):
        s3.create_bucket(Bucket=bucket)
    s3.put_bucket_policy(
        Bucket="wf-dev-public-keys",
        Policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": "arn:aws:s3:::wf-dev-public-keys/*",
                    }
                ],
            }
        ),
    )
    s3.put_object(
        Bucket="wf-smpcv2-dev-sync-protocol",
        Key="dev_deleted_serial_ids.json",
        Body=json.dumps({"deleted_serial_ids": []}).encode(),
    )

    input_topic = sns.create_topic(
        Name="iris-mpc-input.fifo",
        Attributes={"FifoTopic": "true", "ContentBasedDeduplication": "true"},
    )["TopicArn"]
    result_topic = sns.create_topic(
        Name="iris-mpc-results.fifo",
        Attributes={"FifoTopic": "true", "ContentBasedDeduplication": "true"},
    )["TopicArn"]

    queues = []
    for name in (
        "smpcv2-0-dev.fifo",
        "smpcv2-1-dev.fifo",
        "smpcv2-2-dev.fifo",
        "iris-mpc-results-us-east-1.fifo",
    ):
        url = sqs.create_queue(
            QueueName=name,
            Attributes={
                "FifoQueue": "true",
                "ContentBasedDeduplication": "true",
                "VisibilityTimeout": "600",
            },
        )["QueueUrl"]
        arn = sqs.get_queue_attributes(
            QueueUrl=url, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        queues.append((url, arn))

    for _, arn in queues[:3]:
        sns.subscribe(TopicArn=input_topic, Protocol="sqs", Endpoint=arn)
    sns.subscribe(TopicArn=result_topic, Protocol="sqs", Endpoint=queues[3][1])

    for party in range(3):
        secrets.create_secret(
            Name=f"dev/iris-mpc/ecdh-private-key-{party}",
            SecretString='{"private-key":""}',
        )

    print(
        json.dumps(
            {
                "endpoint": args.endpoint,
                "input_topic": input_topic,
                "result_topic": result_topic,
                "request_queues": [url for url, _ in queues[:3]],
                "result_queue": queues[3][0],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
