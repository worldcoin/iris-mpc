iris-mpc-1:
  fullnameOverride: "iris-mpc-1"
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

  podSecurityContext:
    runAsNonRoot: false
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

  hostNetwork: false

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
    - name: RUST_LOG
      value: "info"

    - name: AWS_REGION
      value: "$AWS_REGION"

    - name: AWS_ENDPOINT_URL
      value: "http://localstack:4566"

    - name: RUST_BACKTRACE
      value: "full"

    - name: NCCL_SOCKET_IFNAME
      value: "eth0"

    - name: NCCL_COMM_ID
      value: "iris-mpc-1.svc.cluster.local:4000"

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

    - name: SMPC__REQUESTS_QUEUE_URL
      value: "arn:aws:sns:eu-central-1:000000000000:iris-mpc-input"

    - name: SMPC__RESULTS_TOPIC_ARN
      value: "arn:aws:sns:eu-central-1:000000000000:iris-mpc-results"

    - name: SMPC__PROCESSING_TIMEOUT_SECS
      value: "60"

    - name: SMPC__PATH
      value: "/data/"

    - name: SMPC__KMS_KEY_ARNS
      value: '["arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000000","arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000001","arn:aws:kms:$AWS_REGION:000000000000:key/00000000-0000-0000-0000-000000000002"]'

    - name: SMPC__PARTY_ID
      value: "1"

    - name: SMPC__PUBLIC_KEY_BASE_URL
      value: "http://wf-$ENV-public-keys.s3.localhost.localstack.cloud:4566"

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
      value: "smpcv2-$ENV-1"

    - name: SMPC__RETURN_PARTIAL_RESULTS
      value: "true"

    - name: SMPC__NODE_HOSTNAMES
      value: '["iris-mpc-0.svc.cluster.local","iris-mpc-1.svc.cluster.local","iris-mpc-2.svc.cluster.local"]'

    - name: SMPC__IMAGE_NAME
      value: "ghcr.io/worldcoin/iris-mpc:$IRIS_MPC_IMAGE_TAG"

  initContainer:
    enabled: true
    image: "ghcr.io/worldcoin/iris-mpc:2694d8cbb37c278ed84951ef9aac3af47b21f146" # no-cuda image
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
