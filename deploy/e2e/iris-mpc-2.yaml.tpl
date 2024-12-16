iris-mpc-2:
  fullnameOverride: "iris-mpc-2"
  image: "ghcr.io/worldcoin/iris-mpc:$IRIS_MPC_IMAGE_TAG"

  environment: $ENV
  replicaCount: 1

  strategy:
    type: Recreate

  datadog:
    enabled: false

  ports:
    - containerPort: 3000
      name: health
      protocol: TCP

  livenessProbe:
    httpGet:
      path: /health
      port: health

  readinessProbe:
    periodSeconds: 30
    httpGet:
      path: /ready
      port: health

  startupProbe:
    initialDelaySeconds: 60
    failureThreshold: 40
    periodSeconds: 30
    httpGet:
      path: /ready
      port: health

  resources:
    limits:
      cpu: 31
      memory: 60Gi
      nvidia.com/gpu: 1
      vpc.amazonaws.com/efa: 1
    requests:
      cpu: 30
      memory: 55Gi
      nvidia.com/gpu: 1
      vpc.amazonaws.com/efa: 1

  imagePullSecrets:
    - name: github-secret

  nodeSelector:
    kubernetes.io/arch: amd64

  hostNetwork: false

  tolerations:
    - key: "dedicated"
      operator: "Equal"
      value: "gpuGroup"
      effect: "NoSchedule"

  keelPolling:
    # -- Specifies whether keel should poll for container updates
    enabled: true

  libsDir:
    enabled: true
    path: "/libs"
    size: 2Gi
    files:
      - path: "/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.12.2.5.6"
        file: "libcublasLt.so.12.2.5.6"
      - path: "/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so.12.2.5.6"
        file: "libcublas.so.12.2.5.6"

  preStop:
    # preStop.sleepPeriod specifies the time spent in Terminating state before SIGTERM is sent
    sleepPeriod: 10

  # terminationGracePeriodSeconds specifies the grace time between SIGTERM and SIGKILL
  terminationGracePeriodSeconds: 180 # 3x SMPC__PROCESSING_TIMEOUT_SECS

  env:
    - name: RUST_LOG
      value: "info"

    - name: RUST_BACKTRACE
      value: "full"

    - name: NCCL_SOCKET_IFNAME
      value: "eth0"

    - name: NCCL_COMM_ID
      value: "iris-mpc-node.1.$ENV.smpcv2.worldcoin.dev:4000"

    - name: SMPC__ENVIRONMENT
      value: "$ENV"

    - name: SMPC__SERVICE__SERVICE_NAME
      value: "smpcv2-server-$ENV"

    - name: SMPC__DATABASE__URL
      valueFrom:
        secretKeyRef:
          key: DATABASE_AURORA_URL
          name: application

    - name: SMPC__DATABASE__MIGRATE
      value: "true"

    - name: SMPC__DATABASE__CREATE
      value: "true"

    - name: SMPC__DATABASE__LOAD_PARALLELISM
      value: "8"

    - name: SMPC__AWS__REGION
      value: "eu-north-1"

    - name: SMPC__REQUESTS_QUEUE_URL
      value: "arn:aws:sns:eu-central-1:000000000000:iris-mpc-input"

    - name: SMPC__RESULTS_TOPIC_ARN
      value: "arn:aws:sns:eu-central-1:000000000000:iris-mpc-results"

    - name: SMPC__PROCESSING_TIMEOUT_SECS
      value: "60"

    - name: SMPC__PATH
      value: "/data/"

    - name: SMPC__KMS_KEY_ARNS
      value: '["arn:aws:kms:eu-north-1:000000000000:key/00000000-0000-0000-0000-000000000000","arn:aws:kms:eu-north-1:000000000000:key/00000000-0000-0000-0000-000000000001","arn:aws:kms:eu-north-1:000000000000:key/00000000-0000-0000-0000-000000000002"]'

    - name: SMPC__PARTY_ID
      value: "2"

    - name: SMPC__PUBLIC_KEY_BASE_URL
      value: "https://pki-smpcv2-stage.worldcoin.org"

    - name: SMPC__ENABLE_S3_IMPORTER
      value: "false"

    - name: SMPC__SHARES_BUCKET_NAME
      value: "wf-smpcv2-stage-sns-requests"

    - name: SMPC__CLEAR_DB_BEFORE_INIT
      value: "true"

    - name: SMPC__INIT_DB_SIZE
      value: "80000"

    - name: SMPC__MAX_DB_SIZE
      value: "110000"

    - name: SMPC__MAX_BATCH_SIZE
      value: "64"

    - name: SMPC__SERVICE__METRICS__HOST
      valueFrom:
        fieldRef:
          fieldPath: status.hostIP

    - name: SMPC__SERVICE__METRICS__PORT
      value: "8125"

    - name: SMPC__SERVICE__METRICS__QUEUE_SIZE
      value: "5000"

    - name: SMPC__SERVICE__METRICS__BUFFER_SIZE
      value: "256"

    - name: SMPC__SERVICE__METRICS__PREFIX
      value: "smpcv2-$ENV-2"

    - name: SMPC__RETURN_PARTIAL_RESULTS
      value: "true"

    - name: SMPC__NODE_HOSTNAMES
      value: '["iris-mpc-node.1.$ENV.smpcv2.worldcoin.dev","iris-mpc-node.2.$ENV.smpcv2.worldcoin.dev","iris-mpc-node.3.$ENV.smpcv2.worldcoin.dev"]'

    - name: SMPC__IMAGE_NAME
      value: "ghcr.io/worldcoin/iris-mpc:$IRIS_MPC_IMAGE_TAG"

  initContainer:
    enabled: true
    image: "amazon/aws-cli:2.17.62"
    name: "iris-mpc-2-copy-cuda-libs"
    env:
      - name: PARTY_ID
        value: "3"
      - name: MY_NODE_IP
        valueFrom:
          fieldRef:
            fieldPath: status.hostIP
    configMap:
      name: "iris-mpc-2-init"
      init.sh: |
        #!/usr/bin/env bash

        # Set up environment variables
        HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name "$PARTY_ID".$ENV.smpcv2.worldcoin.dev --query "HostedZones[].Id" --output text)

        # Generate the JSON content in memory
        BATCH_JSON=$(cat <<EOF
        {
          "Comment": "Upsert the A record for iris-mpc NCCL_COMM_ID",
          "Changes": [
            {
              "Action": "UPSERT",
              "ResourceRecordSet": {
                "Name": "iris-mpc-node.$PARTY_ID.$ENV.smpcv2.worldcoin.dev",
                "TTL": 5,
                "Type": "A",
                "ResourceRecords": [{
                  "Value": "$MY_NODE_IP"
                }]
              }
            }
          ]
        }
        EOF
        )

        # Execute AWS CLI command with the generated JSON
        aws route53 change-resource-record-sets --hosted-zone-id "$HOSTED_ZONE_ID" --change-batch "$BATCH_JSON"

        cd /libs
        aws s3 cp s3://wf-smpcv2-stage-libs/libcublas.so.12.2.5.6 .
        aws s3 cp s3://wf-smpcv2-stage-libs/libcublasLt.so.12.2.5.6 .
