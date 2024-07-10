# gpu-iris-mpc

## Setup
- Node PoC implementation in `src/bin/server.rs`
- Example client in `src/bin/client.rs`


#### Running the server without config override
```
AWS_REGION=eu-north-1 AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin server
```

#### Running the server with override
```
AWS_REGION=eu-north-1 AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin server -- --party-id {0,1,2} --queue https://sqs.eu-north-1.amazonaws.com/xxx/mpc1.fifo
```

#### Running the client
```
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx cargo run --release --bin client -- --topic-arn arn:aws:sns:eu-north-1:xxx:mpc.fifo --db-index 2 --n-repeat 32
```

### Server configuration

Please note that this mechanism does not apply to the client part.


#### Environment variables
Application can be completely configured via environment variables. To see the list of variables please check out contents of the `.env*` files.

**Important!**
Please note that there is a dist file per an instance of MPC node. Before running, please make sure to rename the correct dist file to `.env`.

For now the environment variables are read in via a `dotenvy` crate for the ease of development.

#### CLI arguments
Not to force developers into a change of approach to work that has been established, overloading the environment variables with CLI arguments is also possible. 
List of possible overrides is represented by the following `struct`:
```rust
pub struct Opt {
    #[structopt(long)]
    requests_queue_url: Option<String>,

    #[structopt(long)]
    results_topic_arn: Option<String>,

    #[structopt(long)]
    party_id: Option<usize>,

    #[structopt(long)]
    bootstrap_url: Option<String>,

    #[structopt(long)]
    path: Option<String>,
}
```
Please note that due to ambiguity, all the arguments need to be provided using their full names.

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
