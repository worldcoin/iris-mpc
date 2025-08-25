hnsw-smpc-2:
  fullnameOverride: "hnsw-smpc-2"
  image: "ghcr.io/worldcoin/iris-mpc-cpu:$IRIS_MPC_IMAGE_TAG"

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

  podSecurityContext:
    seccompProfile:
      type: RuntimeDefault

  resources:
    limits:
      cpu: 4
      memory: 4Gi
    requests:
      cpu: 4
      memory: 4Gi

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
      - "mongodb.e2e.svc.cluster.local"
      - "e2e.svc.cluster.local"
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

    - name: SMPC__ENVIRONMENT
      value: "$ENV"

    - name: SMPC__SERVICE__SERVICE_NAME
      value: "hnsw-service-2"

    - name: SMPC__PARTY_ID
      value: "2"

    - name: SMPC__CPU_DATABASE__URL
      valueFrom:
        secretKeyRef:
          key: DATABASE_AURORA_HNSW_URL
          name: application

    - name: SMPC__MAX_DB_SIZE
      value: "100000"

    - name: SMPC__MAX_BATCH_SIZE
      value: "1"

    - name: SMPC__PROCESSING_TIMEOUT_SECS
      value: "30"  # 2 minutes per batch in stage, bump to 4 in prod

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

    - name: SMPC__ENABLE_SENDING_ANONYMIZED_STATS_MESSAGE
      value: "false"

    - name: SMPC__HAWK_SERVER_REAUTHS_ENABLED
      value: "false"

    - name: SMPC__HAWK_SERVER_RESETS_ENABLED
      value: "false"

    - name: SMPC__HAWK_SERVER_DELETIONS_ENABLED
      value: "true"

    - name: SMPC__LUC_ENABLED
      value: "true"

    - name: SMPC__LUC_LOOKBACK_RECORDS
      value: "5"

    - name: SMPC__LUC_SERIAL_IDS_FROM_SMPC_REQUEST
      value: "false"

    - name: SMPC__AWS__REGION
      value: "$AWS_REGION"

    - name: SMPC__SERVICE_PORTS
      value: '["4000","4001","4002"]'

    - name: SMPC__HAWK_SERVER_HEALTHCHECK_PORT
      value: '3000'

    - name: SMPC__NODE_HOSTNAMES
      value: '["hnsw-smpc-0.$ENV.svc.cluster.local","hnsw-smpc-1.$ENV.svc.cluster.local","0.0.0.0"]'

    - name: SMPC__SHARES_BUCKET_NAME
      value: "wf-smpcv2-stage-sns-requests"

    - name: SMPC__ENABLE_S3_IMPORTER
      value: "false"

    - name: SMPC__DB_CHUNKS_BUCKET_NAME
      value: "iris-mpc-db-exporter-store-node-2-stage--eun1-az3--x-s3"

    - name: SMPC__DB_CHUNKS_FOLDER_NAME
      value: "hnsw_even_odd_with_version_id_output_16k"

    - name: SMPC__LOAD_CHUNKS_PARALLELISM
      value: "64"

    - name: SMPC__LOAD_CHUNKS_BUFFER_SIZE
      value: "1024"

    - name: SMPC__REQUESTS_QUEUE_URL
      value: "http://sqs.$AWS_REGION.localhost.localstack.cloud:4566/000000000000/hnsw-smpc-request-2-e2e.fifo"

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
      value: "hnsw-2"

    - name: SMPC__IMAGE_NAME
      value: "ghcr.io/worldcoin/iris-mpc-cpu:$IRIS_MPC_IMAGE_TAG"

    - name: SMPC__ENABLE_MODIFICATIONS_SYNC
      value: "true"

    - name: SMPC__ENABLE_MODIFICATIONS_REPLAY
      value: "true"

  initContainer:
    enabled: true
    image: "amazon/aws-cli:2.17.62"
    name: "hnsw-mpc-dns-records-updater-2"
    env:
      - name: PARTY_ID
        value: "2"
      - name: MY_POD_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
    configMap:
      name: "hnws-init-2"
      init.sh: |
        #!/usr/bin/env bash

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
                "Name": "hnsw-smpc-2.orb.e2e.test",
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
