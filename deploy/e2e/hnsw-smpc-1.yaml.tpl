hnsw-smpc-1:
  fullnameOverride: "hnsw-smpc-1"
  image: "$IMAGE_REGISTRY_IRIS_MPC/iris-mpc-cpu:$IRIS_MPC_CPU_IMAGE_TAG"

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
    - containerPort: 4000
      name: tcp-4000
      protocol: TCP
    - containerPort: 4001
      name: tcp-4001
      protocol: TCP
    - containerPort: 4002
      name: tcp-4002
      protocol: TCP
    - containerPort: 4100
      name: tcp-4100
      protocol: TCP
    - containerPort: 4101
      name: tcp-4101
      protocol: TCP
    - containerPort: 4102
      name: tcp-4102
      protocol: TCP

  service:
    additionalPorts:
    - name: health
      port: 3000
      targetPort: 3000

  livenessProbe:
    httpGet:
      path: /health
      port: health

  readinessProbe:
    periodSeconds: 30
    httpGet:
      path: /health
      port: health

  startupProbe:
    initialDelaySeconds: 60
    failureThreshold: 120
    periodSeconds: 30
    httpGet:
      path: /health
      port: health

  resources:
    limits:
      cpu: 4
      memory: 16Gi
    requests:
      cpu: 4
      memory: 16Gi

  imagePullSecrets:
    - name: github-secret

  nodeSelector:
    kubernetes.io/arch: amd64

  hostNetwork: false

  dnsPolicy: None
  dnsConfig:
    nameservers:
      - "172.20.0.10"
    searches:
      - "localstack"
      - "mongodb.$ENV.svc.cluster.local"
      - "$ENV.svc.cluster.local"
      - "svc.cluster.local"
      - "cluster.local"

  preStop:
    # preStop.sleepPeriod specifies the time spent in Terminating state before SIGTERM is sent
    sleepPeriod: 10

  # terminationGracePeriodSeconds specifies the grace time between SIGTERM and SIGKILL
  # long enough to allow for graceful shutdown to safely process 2 batches
  # single batch timeout in stage is 240 seconds
  terminationGracePeriodSeconds: 500

  # mountSSLCerts:
  #   enabled: true
  #   mountPath: /etc/ssl/private

  env:
    - name: RUST_LOG
      value: "info"

    - name: RUST_BACKTRACE
      value: "full"

    - name: AWS_REGION
      value: "$AWS_REGION"

    - name: AWS_ACCESS_KEY_ID
      value: "access_key"

    - name: AWS_SECRET_ACCESS_KEY
      value: "secret_key"

    - name: AWS_ENDPOINT_URL
      value: "http://localstack:4566"

    - name: SMPC__ENVIRONMENT
      value: "$ENV"

    - name: SMPC__SERVICE__SERVICE_NAME
      value: "hnsw-service-1"

    - name: SMPC__SERVER_COORDINATION__PARTY_ID
      value: "1"

    - name: SMPC__SERVER_COORDINATION__NODE_HOSTNAMES
      value: '["hnsw-smpc-0.orb.e2e.test","0.0.0.0","hnsw-smpc-2.orb.e2e.test"]'

    - name: SMPC__SERVER_COORDINATION__IMAGE_NAME
      value: $(IMAGE_NAME)

    - name: SMPC__CPU_DATABASE__URL
      valueFrom:
        secretKeyRef:
          key: DATABASE_AURORA_HNSW_URL
          name: application

    - name: SMPC__HNSW_SCHEMA_NAME_SUFFIX
      value: "_hnsw"

    - name: SMPC__MAX_DB_SIZE
      value: "100000"

    - name: SMPC__MAX_BATCH_SIZE
      value: "32"

    - name: SMPC__PROCESSING_TIMEOUT_SECS
      value: "60"

    - name: SMPC__HAWK_REQUEST_PARALLELISM
      value: "1024"

    - name: SMPC__HAWK_CONNECTION_PARALLELISM
      value: "8"

    - name: SMPC__HAWK_STREAM_PARALLELISM
      value: "32"

    - name: SMPC__OVERRIDE_MAX_BATCH_SIZE
      value: "false"

    - name: SMPC__DISABLE_PERSISTENCE
      value: "false"

    - name: SMPC__MATCH_DISTANCES_BUFFER_SIZE
      value: "64"

    - name: SMPC__N_BUCKETS
      value: "16"

    - name: SMPC__ANON_STATS_DATABASE__URL
      valueFrom:
        secretKeyRef:
          key: DATABASE_AURORA_URL
          name: application

    - name: SMPC__ANON_STATS_DATABASE__MIGRATE
      value: "false"

    - name: SMPC__ANON_STATS_DATABASE__CREATE
      value: "false"

    - name: SMPC__ANON_STATS_DATABASE__LOAD_PARALLELISM
      value: "8"

    - name: SMPC__ANON_STATS_DATABASE__DB_SCHEMA_NAME
      value: "anon_stats_hnsw_1"

    - name: SMPC__HAWK_SERVER_REAUTHS_ENABLED
      value: "false"

    - name: SMPC__HAWK_SERVER_RESETS_ENABLED
      value: "false"

    - name: SMPC__HAWK_SERVER_DELETIONS_ENABLED
      value: "true"

    - name: SMPC__LUC_ENABLED
      value: "true"

    - name: SMPC__LUC_LOOKBACK_RECORDS
      value: "0"

    - name: SMPC__LUC_SERIAL_IDS_FROM_SMPC_REQUEST
      value: "true"

    - name: SMPC__AWS__REGION
      value: "$AWS_REGION"

    - name: SMPC__SERVICE_PORTS
      value: '["4000","4001","4002"]'

    - name: SMPC__HAWK_SERVER_HEALTHCHECK_PORT
      value: '3000'

    - name: SMPC__SHARES_BUCKET_NAME
      value: "wf-smpcv2-stage-sns-requests"

    - name: SMPC__ENABLE_S3_IMPORTER
      value: "false"

    - name: SMPC__DB_CHUNKS_BUCKET_NAME
      value: "iris-mpc-db-exporter-store-node-1-stage--eun1-az3--x-s3"

    - name: SMPC__DB_CHUNKS_FOLDER_NAME
      value: "hnsw_even_odd_with_version_id_output_16k"

    - name: SMPC__LOAD_CHUNKS_PARALLELISM
      value: "64"

    - name: SMPC__LOAD_CHUNKS_BUFFER_SIZE
      value: "1024"

    - name: SMPC__REQUESTS_QUEUE_URL
      value: "http://sqs.$AWS_REGION.localhost.localstack.cloud:4566/000000000000/hnsw-smpc-request-1-e2e.fifo"

    - name: SMPC__RESULTS_TOPIC_ARN
      value: "arn:aws:sns:$AWS_REGION:000000000000:hnsw-smpc-results.fifo"

    - name: SMPC__KMS_KEY_ARNS
      value: '["arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000000","arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000001","arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000002"]'

    - name: SMPC__HEARTBEAT_INITIAL_RETRIES
      value: "65"

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
      value: "hnsw-0"

    - name: SMPC__IMAGE_NAME
      value: "$IMAGE_REGISTRY_IRIS_MPC/iris-mpc-cpu:$IRIS_MPC_CPU_IMAGE_TAG"

    - name: SMPC__ENABLE_MODIFICATIONS_SYNC
      value: "true"

    - name: SMPC__ENABLE_MODIFICATIONS_REPLAY
      value: "true"

  initContainer:
    enabled: true
    image: "$IMAGE_REGISTRY_INIT_CONTAINER/iris-mpc:$IRIS_MPC_KEY_MANAGER_IMAGE_TAG" # no-cuda image
    name: "hnsw-mpc-dns-records-updater-1"
    env:
      - name: PARTY_ID
        value: "1"
      - name: MY_POD_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
    configMap:
      name: "hnws-init-1"
      init.sh: |
        #!/usr/bin/env bash
        set -e

        # CPU and GPU versions use the same encryption keys in e2e. Commenting this out to avoid race conditions which end up with unseal errors in one of the systems.
        # Currently, we rely on GPU pod's init container to do the key rotation. Uncomment this once GPU is deprecated.
        # key-manager --node-id 1 --env $ENV --region $AWS_REGION --endpoint-url "http://localstack:4566" rotate --public-key-bucket-name wf-$ENV-public-keys
        # key-manager --node-id 1 --env $ENV --region $AWS_REGION --endpoint-url "http://localstack:4566" rotate --public-key-bucket-name wf-$ENV-public-keys

        # Wait for GPU pods to complete key generation
        echo "Waiting for GPU pods to complete key generation..."
        BUCKET_NAME="wf-$ENV-public-keys"
        EXPECTED_KEYS=3
        TIMEOUT=120  # 2 minutes timeout
        SLEEP_INTERVAL=5

        wait_for_keys() {
            local elapsed=0
            while [ $elapsed -lt $TIMEOUT ]; do
                echo "Checking for keys in bucket: $BUCKET_NAME"

                # Count objects in the bucket
                KEY_COUNT=$(aws s3 ls s3://$BUCKET_NAME --endpoint-url "http://localstack:4566" --region eu-central-1 | wc -l)

                echo "Found $KEY_COUNT keys, expecting $EXPECTED_KEYS"

                if [ "$KEY_COUNT" -ge "$EXPECTED_KEYS" ]; then
                    echo "All $EXPECTED_KEYS keys found! GPU key generation completed."
                    return 0
                fi

                echo "Waiting for GPU pods to generate keys... ($elapsed/$TIMEOUT seconds elapsed)"
                sleep $SLEEP_INTERVAL
                elapsed=$((elapsed + SLEEP_INTERVAL))
            done

            echo "Timeout waiting for GPU key generation after $TIMEOUT seconds"
            echo "Found $KEY_COUNT keys, but expected $EXPECTED_KEYS"
            exit 1
        }

        # Wait for keys before proceeding
        wait_for_keys

        # Set up environment variables
        HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name --region $AWS_REGION --dns-name orb.e2e.test --query "HostedZones[].Id" --output text)

        # Generate the JSON content in memory
        BATCH_JSON=$(cat <<EOF
        {
          "Comment": "Upsert the A record for HNSW pod",
          "Changes": [
            {
              "Action": "UPSERT",
              "ResourceRecordSet": {
                "Name": "hnsw-smpc-1.orb.e2e.test",
                "TTL": 5,
                "Type": "A",
                "ResourceRecords": [{
                  "Value": "$MY_POD_IP"
                }]
              }
            }
          ]
        }
        EOF
        )

        # Execute AWS CLI command with the generated JSON
        aws route53 change-resource-record-sets --hosted-zone-id "$HOSTED_ZONE_ID" --change-batch "$BATCH_JSON"
