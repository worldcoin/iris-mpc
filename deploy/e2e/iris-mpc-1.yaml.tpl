iris-mpc-1:
  fullnameOverride: "iris-mpc-1"
  image: "$IMAGE_REGISTRY_IRIS_MPC/iris-mpc:$IRIS_MPC_IMAGE_TAG"

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
    failureThreshold: 40
    periodSeconds: 30
    httpGet:
      path: /health
      port: health

  podSecurityContext:
    seccompProfile:
      type: RuntimeDefault

  resources:
    limits:
      cpu: 31
      memory: 60Gi
      nvidia.com/gpu: 1

    requests:
      cpu: 30
      memory: 55Gi
      nvidia.com/gpu: 1

  imagePullSecrets:
    - name: github-secret

  nodeSelector:
    kubernetes.io/arch: amd64

  hostNetwork: true

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

  tolerations:
    - key: "gpuGroup"
      operator: "Equal"
      value: "dedicated"
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
    - name: NCCL_SOCKET_IFNAME
      value: "eth0"

    - name: NCCL_IB_DISABLE
      value: "1"

    - name: NCCL_IBEXT_DISABLE
      value: "1"

    - name: NCCL_NET
      value: "Socket"

    - name: RUST_LOG
      value: "info"

    - name: AWS_REGION
      value: "$AWS_REGION"

    - name: AWS_ACCESS_KEY_ID
      value: "access_key"

    - name: AWS_SECRET_ACCESS_KEY
      value: "secret_key"

    - name: AWS_ENDPOINT_URL
      value: "http://localstack:4566"

    - name: RUST_BACKTRACE
      value: "full"

    - name: NCCL_COMM_ID
      value: "iris-mpc-0.orb.e2e.test:4000"

    - name: SMPC__ENVIRONMENT
      value: "$ENV"

    - name: SMPC__AWS__REGION
      value: "$AWS_REGION"

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

    - name: SMPC__REQUESTS_QUEUE_URL
      value: "http://sqs.$AWS_REGION.localhost.localstack.cloud:4566/000000000000/smpcv2-1-e2e.fifo"

    - name: SMPC__RESULTS_TOPIC_ARN
      value: "arn:aws:sns:$AWS_REGION:000000000000:iris-mpc-results.fifo"

    - name: SMPC__PROCESSING_TIMEOUT_SECS
      value: "600"

    - name: SMPC__PATH
      value: "/data/"

    - name: SMPC__KMS_KEY_ARNS
      value: '["arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000000","arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000001","arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000002"]'

    - name: SMPC__SERVER_COORDINATION__PARTY_ID
      value: "1"

    - name: SMPC__SERVER_COORDINATION__NODE_HOSTNAMES
      value: '["iris-mpc-0.$ENV.svc.cluster.local","iris-mpc-1.$ENV.svc.cluster.local","iris-mpc-2.$ENV.svc.cluster.local"]'

    - name: SMPC__SERVER_COORDINATION__IMAGE_NAME
      value: $(IMAGE_NAME)

    - name: SMPC__PUBLIC_KEY_BASE_URL
      value: "http://wf-$ENV-public-keys.s3.localhost.localstack.cloud:4566"

    - name: SMPC__ENABLE_S3_IMPORTER
      value: "false"

    - name: SMPC__SHARES_BUCKET_NAME
      value: "wf-smpcv2-stage-sns-requests"

    - name: SMPC__SNS_BUFFER_BUCKET_NAME
      value: "wf-smpcv2-stage-sns-buffer"

    - name: SMPC__CLEAR_DB_BEFORE_INIT
      value: "true"

    - name: SMPC__INIT_DB_SIZE
      value: "0"

    - name: SMPC__MAX_DB_SIZE
      value: "10000"

    - name: SMPC__MAX_BATCH_SIZE
      value: "64"

    - name: SMPC__MATCH_DISTANCES_BUFFER_SIZE
      value: "64"

    - name: SMPC__ENABLE_SENDING_ANONYMIZED_STATS_MESSAGE
      value: "true"

    - name: SMPC__ENABLE_REAUTH
      value: "true"

    - name: SMPC__ENABLE_RESET
      value: "true"

    - name: SMPC__LUC_ENABLED
      value: "true"

    - name: SMPC__LUC_LOOKBACK_RECORDS
      value: "0"

    - name: SMPC__LUC_SERIAL_IDS_FROM_SMPC_REQUEST
      value: "true"

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
      value: "smpcv2-$ENV-1"

    - name: SMPC__RETURN_PARTIAL_RESULTS
      value: "true"

    - name: SMPC__FIXED_SHARED_SECRETS
      value: "true"

    - name: SMPC__NODE_HOSTNAMES
      value: '["iris-mpc-0.$ENV.svc.cluster.local","iris-mpc-1.$ENV.svc.cluster.local","iris-mpc-2.$ENV.svc.cluster.local"]'

    - name: SMPC__IMAGE_NAME
      value: "$IMAGE_REGISTRY_IRIS_MPC/iris-mpc:$IRIS_MPC_IMAGE_TAG"

    - name: SMPC__HEARTBEAT_INITIAL_RETRIES
      value: "1000"

    - name: SMPC__ENABLE_MODIFICATIONS_SYNC
      value: "true"

    - name: SMPC__ENABLE_MODIFICATIONS_REPLAY
      value: "true"

    - name : SMPC__ENABLE_DEBUG_TIMING
      value: "true"

    - name : SMPC__FULL_SCAN_SIDE_SWITCHING_ENABLED
      value: "false"

  initContainer:
    enabled: true
    image: "$IMAGE_REGISTRY_INIT_CONTAINER/iris-mpc:$IRIS_MPC_KEY_MANAGER_IMAGE_TAG" # no-cuda image
    name: "iris-mpc-1-copy-cuda-libs"
    env:
      - name: AWS_REGION
        value: "$AWS_REGION"
      - name: PARTY_ID
        value: "2"
      - name: MY_NODE_IP
        valueFrom:
          fieldRef:
            fieldPath: status.hostIP
    configMap:
      name: "iris-mpc-1-init"
      init.sh: |
        #!/usr/bin/env bash
        set -e

        cd /libs

        aws s3 cp s3://wf-smpcv2-stage-libs/libcublas.so.12.2.5.6 .
        aws s3 cp s3://wf-smpcv2-stage-libs/libcublasLt.so.12.2.5.6 .

        key-manager --node-id 1 --env $ENV --region $AWS_REGION --endpoint-url "http://localstack:4566" rotate --public-key-bucket-name wf-$ENV-public-keys
        key-manager --node-id 1 --env $ENV --region $AWS_REGION --endpoint-url "http://localstack:4566" rotate --public-key-bucket-name wf-$ENV-public-keys
