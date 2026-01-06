anon-stats-server-smpc-1:
  fullnameOverride: "anon-stats-server-smpc-1"
  image: "$IMAGE_REGISTRY_IRIS_MPC/anon-stats-server:$IRIS_MPC_ANON_STATS_SERVER_IMAGE_TAG"
  environment: $ENV
  replicaCount: 1

  serviceAccount:
    create: false

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

  startupProbe:
    httpGet:
      path: /health
      port: health

  podSecurityContext:
    runAsUser: 65534
    runAsGroup: 65534

  livenessProbe:
    httpGet:
      path: /health
      port: health

  readinessProbe:
    periodSeconds: 20
    failureThreshold: 4
    httpGet:
      path: /health
      port: health

  resources:
    limits:
      cpu: "2"
      memory: 10Gi
    requests:
      cpu: "2"
      memory: 10Gi

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

  env:
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

    - name: SMPC__ENVIRONMENT
      value: "$ENV"

    - name: SMPC__SERVICE__SERVICE_NAME
      value: "anon-stats-server-smpcv2-1"

    - name: SMPC__SNS_BUFFER_BUCKET_NAME
      value: "wf-smpcv2-$ENV-sns-buffer"

    - name: SMPC__DB_URL
      valueFrom:
        secretKeyRef:
          key: DATABASE_AURORA_URL
          name: application

    - name: SMPC__DB_SCHEMA_NAME
      value: "anon_stats_smpcv2_1"

    - name: SMPC__SERVICE_PORTS
      value: '["4000","4001","4002"]'

    - name: SMPC__SERVER_COORDINATION__PARTY_ID
      value: "1"

    - name: SMPC__SERVER_COORDINATION__NODE_HOSTNAMES
      value: '["anon-stats-server-0.orb.e2e.test","0.0.0.0","anon-stats-server-2.orb.e2e.test"]'

    - name: SMPC__SERVER_COORDINATION__IMAGE_NAME
      value: $(IMAGE_NAME)

    - name: SMPC__RESULTS_TOPIC_ARN
      value: "arn:aws:sns:$AWS_REGION:000000000000:iris-mpc-results.fifo"

    - name: SMPC__AWS__REGION
      value: "$AWS_REGION"

    - name: SMPC__SERVER_COORDINATION__HEARTBEAT_INITIAL_RETRIES
      value: "5"

    - name: SMPC__SERVER_COORDINATION__HTTP_QUERY_RETRY_DELAY_MS
      value: "5000"

    - name: SMPC__SERVER_COORDINATION__HEARTBEAT_INTERVAL_SECS
      value: "3"

    - name: SMPC__SERVER_COORDINATION__SHUTDOWN_LAST_RESULTS_SYNC_TIMEOUT_SECS
      value: "10"

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
      value: "anon-stats-server-$ENV-smpcv2-1"

    - name: SMPC__N_BUCKETS_1D
      value: "375"

    - name: SMPC__N_BUCKETS_2D
      value: "375"

    - name: SMPC__MIN_1D_JOB_SIZE_REAUTH
      value: "10"

    - name: SMPC__MIN_1D_JOB_SIZE
      value: "1"

    - name: SMPC__MIN_2D_JOB_SIZE_REAUTH
      value: "10"

    - name: SMPC__MIN_2D_JOB_SIZE
      value: "64"

    - name: SMPC__POLL_INTERVAL_SECS
      value: "30"


  initContainer:
    enabled: true
    image: "amazon/aws-cli:2.17.62"
    name: "anon-stats-server-smpcv2-dns-records-updater"
    env:
      - name: MY_NODE_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
    configMap:
      name: "anon-stats-server-init-1"
      init.sh: |
        #!/usr/bin/env bash

        # Set up environment variables
        HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name --region $AWS_REGION --dns-name orb.e2e.test --query "HostedZones[0].Id" --output text)

        # Generate the JSON content in memory
        BATCH_JSON=$(cat <<EOF
        {
          "Comment": "Upsert the A record for anon-stats-server-smpcv2 pod",
          "Changes": [
            {
              "Action": "UPSERT",
              "ResourceRecordSet": {
                "Name": "anon-stats-server-1.orb.e2e.test",
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
        aws route53 change-resource-record-sets --region $AWS_REGION --hosted-zone-id "$HOSTED_ZONE_ID" --change-batch "$BATCH_JSON"
