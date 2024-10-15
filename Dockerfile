FROM --platform=linux/amd64 ubuntu:22.04 as build-image

WORKDIR /src
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    texinfo \
    libcap2-bin \
    pkg-config \
    git \
    devscripts \
    debhelper \
    ca-certificates \
    wget

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH "/root/.cargo/bin:${PATH}"
ENV RUSTUP_HOME "/root/.rustup"
ENV CARGO_HOME "/root/.cargo"
RUN rustup toolchain install nightly-2024-07-10
RUN rustup default nightly-2024-07-10
RUN rustup component add cargo
RUN cargo install cargo-build-deps \
    && cargo install cargo-edit

FROM --platform=linux/amd64 build-image as build-app
WORKDIR /src/gpu-iris-mpc
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-gnu --bin nccl --bin server --bin client --bin key-manager --bin upgrade-server --bin upgrade-client --bin upgrade-checker

FROM --platform=linux/amd64 ghcr.io/worldcoin/iris-mpc-base:cuda12_2-nccl2_22_3_1
ENV DEBIAN_FRONTEND=noninteractive

# Include client, server and key-manager, upgrade-client and upgrade-server binaries
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/nccl /bin/nccl
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/server /bin/server
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/client /bin/client
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/key-manager /bin/key-manager
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/upgrade-server /bin/upgrade-server
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/upgrade-client /bin/upgrade-client
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/upgrade-checker /bin/upgrade-checker

USER 65534
ENTRYPOINT ["/bin/server"]
