# gpu-iris-mpc

## How to release

New releases are created automagically by [Release Drafter GH action](https://github.com/worldcoin//gpu-iris-mpc/actions/workflows/release.yaml).

Type of release bump is made of commits (tags feat/bugfix/etc...).

Release is created as draft, so you have to edit it manually and change it to final.


## Setup
- Node PoC implementation in `src/bin/server.rs`
- Example client in `src/bin/client.rs`

```
AWS_REGION=eu-north-1 AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin server -- --party-id {0,1,2} --queue https://sqs.eu-north-1.amazonaws.com/xxx/mpc1.fifo
```

```
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin client -- --topic-arn arn:aws:sns:eu-north-1:xxx:mpc.fifo --db-index 2 --n-repeat 32
```

### Dependencies

Requires a NVIDIA graphics card with recent drivers and CUDA libraries.

The following dependency versions have been confirmed to work:
- nvidia-x11-550.78-6.9.6
- cuda_nvrtc-12.2.140
- libcublas-12.2.5.6

Some Linux distributions have a (lib)cuda package 12.2 which depends on earlier versions of these packages.
It might not work.

## Testing

To run the tests:
```sh
docker-compose up -d
cargo test --release
# Requires a significant amount of GPU memory
cargo bench
```

If you are using `cargo test` with non-standard library paths, you might need [a workaround](https://github.com/worldcoin/gpu-iris-mpc/issues/25).

## Architecture
![architecture](mpc-architecture-v2.png)

## Streams and Synchronization in V2 (`src/bin/server.rs`)
> TODO: dedup between query and previous is not yet implemented
![streams](mpc-iris-streams-v2.png)

