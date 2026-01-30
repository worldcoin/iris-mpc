# Dev Environment Operational Guide

The dev environment is designed to facilitate development and testing of the application. It is configured to mimic the stage environment as closely as possible, having
the same deployment pattern, which enables us to quickly iterate on different sets of configurations and running artifacts. The main thing that differentiates it from stage is that
the I/O is not orchestrated by our backed, rather, it is completely isolated and can be interacted with directly by sending and
receiving SNS/SQS messages.

This guide provides instructions on how to set up, run, and manage the dev environment effectively.

## Tooling

1. [AWS CLI](https://aws.amazon.com/cli/) - for interacting with AWS services and configuring access to k8s clusters
2. k8s visibility tool. [k9s](https://k9scli.io/) or [Lens](https://k8slens.dev/)
3. `kubectl` cli [installation guide](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
4. Rust toolchain - for building and running the application, and for sending / receiving test messages

Optional

1. kubectx [installation guide](https://github.com/ahmetb/kubectx) for easier switching between k8s contexts


## Access

### AWS

To access the dev environment, you need to have the necessary permissions to interact with the AWS resources and k8s clusters.
This requires you to have an AWS account that has the permissions to assume the roles:

```text
arn::aws:iam::004304088310:role/ampc-hnsw-developer-role
arn::aws:iam::284038850594:role/ampc-hnsw-developer-role
arn::aws:iam::882222437714:role/ampc-hnsw-developer-role
```

These can currently assumed by the following AWS principals:

- Inversed Tech: `387760840988`
- TFH: `033662022620`
- Taceo: TBD

A user can be created on your account, with the following policy attached to it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": [
        "arn::aws:iam::004304088310:role/ampc-hnsw-developer-role",
        "arn::aws:iam::284038850594:role/ampc-hnsw-developer-role",
        "arn::aws:iam::882222437714:role/ampc-hnsw-developer-role",
        "arn::aws:iam::238407200320:role/ampc-hnsw-developer-role"
      ]
    }
  ]
}
```
Once you have an AWS account with the policy above, you can login to it on with the AWS CLI and setup a profile for it
in your aws config file (`~/.aws/config`), and add the -dev account related ones.

```text
[profile <you-account-profile>]
...

# add these 
[profile worldcoin-smpcv-io-0-dev]
source_profile=<you-account-profile>
role_arn=arn:aws:iam::004304088310:role/ampc-hnsw-developer-role

[profile worldcoin-smpcv-io-1-dev]
source_profile=<you-account-profile>
role_arn=arn:aws:iam::284038850594:role/ampc-hnsw-developer-role

[profile worldcoin-smpcv-io-2-dev]
source_profile=<you-account-profile>
role_arn=arn:aws:iam::882222437714:role/ampc-hnsw-developer-role

[profile worldcoin-smpcv-io-vpc-dev]
source_profile=<you-account-profile>
role_arn=arn:aws:iam::238407200320:role/ampc-hnsw-developer-role
```

These profiles can also be used on the AWS console to switch to the appropriate account. Use the "Switch Role" option on the top right corner of the console.

### k8s

To access the k8s clusters, you need the AWS CLI configured with the profiles above. You can then use the following commands to get the k8s config for each cluster:

```bash
AWS_PROFILE=worldcoin-smpcv-io-0-dev aws eks update-kubeconfig --name ampc-hnsw-0-dev --region eu-central-1 --alias ampc-hnsw-0-dev
AWS_PROFILE=worldcoin-smpcv-io-1-dev aws eks update-kubeconfig --name ampc-hnsw-1-dev --region eu-central-1 --alias ampc-hnsw-1-dev
AWS_PROFILE=worldcoin-smpcv-io-2-dev aws eks update-kubeconfig --name ampc-hnsw-2-dev --region eu-central-1 --alias ampc-hnsw-2-dev
```

This will add the k8s config for each cluster to your `~/.kube/config` file. You can then use `kubectl` or any k8s visibility tool to interact with the clusters.

You can switch between the clusters using the following commands:

```bash
kubectl config use-context ampc-hnsw-0-dev
# or use kubectx
kubectx ampc-hnsw-0-dev
# check if everything is working
kubectl -n ampc-hnsw get pods
```

Once you have the clusters in your config, you can then use either Lens or k9s to interact with them. In lens and k9s, you
can easily switch between the clusters and namespaces (the application is deployed on the `ampc-hnsw` namespace), you can also
perform actions like viewing logs, restarting pods, SSHing into nodes & pods, etc.

## Deploying a Custom Branch

Deployments are orchestrated using ArgoCD. ArgoCD periodically checks the git repository for changes and applies them to the clusters. In general,
ArgoCD is configured to track the `main` branch of the repository, meaning that whatever configs in the `deploy/dev` folder exist on the `main`
branch will be the ones deployed to the clusters.

To deploy a custom branch, you need to alter ArgoCD to point to a specific branch. We have a bash script to make it easy:

```shell
AWS_PROFILE=<your-account-profile> ./scripts/tools/argo_update_revision.sh <branch-name> cpu dev
```
Check out the contents of the script for more details.

One of the most common use cases for this is to deploy a branch that contains a new image. To do this, you need to create a PR that updates the image in the `deploy/dev/common-values-ampc-hnsw.yaml` file. Other changes that might be interesting are the
parameters that control the behavior of the system, like the batch size, parallelism, etc. These are node-specific, and live in `deploy/dev/ampc-hnsw-<node-id>-dev/values-ampc-hnsw.yaml`. Remember that most of these need to be exactly the same across all nodes, otherwise the nodes will not start up.

## Logs

Logs are available in DataDog. E.g.: [Logs for Party 0](https://app.datadoghq.com/logs/livetail?query=env:stage%20service:ampc-hnsw%20-OpenTelemetry%20-%22setting%20skip_persistence%22%20-%22Started%20processing%22%20-%22batch%20id%22%20-%22Response%20Status:%20200%20OK%22%20aws_eks_cluster-name:ampc-hnsw-0-stage&agg_m=count&agg_m_source=base&agg_t=count&clustering_pattern_field_path=message&cols=host,service&messageDisplay=inline&storage=driveline&stream_sort=desc&viz=stream&from_ts=1756817555305&to_ts=1756818455305&live=true)
In DataDog we also have a dashboard for monitoring the health of the clusters: [Dashboard](https://app.datadoghq.com/dashboard/mn9-thh-t66/hnsw?fromUser=false&overlay=changes&tpl_var_env%5B0%5D=dev&from_ts=1757258206009&to_ts=1757344606009&live=true)

## Sending Test Messages & Querying Results

To send test messages to the dev environment, you can use the `client` binary:

```bash
# Don't forge to unset the AWS_ENDPOINT_URL AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY from direnv!
export AWS_PROFILE=worldcoin-smpcv-io-vpc-dev
export AWS_REGION=eu-central-1
cargo run --release -p iris-mpc-bins --bin client -- \
   --request-topic-arn arn:aws:sns:eu-central-1:238407200320:iris-mpc-input-dev.fifo \
   --requests-bucket-name wf-smpcv2-dev-sns-requests-v2 \
   --public-key-base-url "https://pki-smpcv2-dev.worldcoin.org" \
   --response-queue-url https://sqs.eu-central-1.amazonaws.com/238407200320/hnsw-smpc-results.fifo \
   --n-batches 14 \
   --batch-size 32
```
