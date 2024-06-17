# gpu-iris-mpc

## Setup
- Node PoC implementation in `src/bin/server.rs`
- Example client in `src/bin/client.rs`

```
AWS_REGION=eu-north-1 AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin server -- --party-id {0,1,2} --queue https://sqs.eu-north-1.amazonaws.com/xxx/mpc1.fifo
```

```
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin client -- --topic-arn arn:aws:sns:eu-north-1:xxx:mpc.fifo --db-index 2 --n-repeat 32
```

## Architecture
![architecture](mpc-architecture-v2.png)

## Streams and Synchronization in V2 (`src/bin/server.rs`)
> TODO: dedup between query and previous is not yet implemented
![streams](mpc-iris-streams-v2.png)

