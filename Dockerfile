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
RUN cargo build --release --target x86_64-unknown-linux-gnu --bin server --bin client

FROM --platform=linux/amd64 ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/server /bin/server
# include client for testing
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/client /bin/client
RUN apt-get update && apt-get install -y pkg-config wget libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-2 libnccl2 libnccl-dev

USER 65534
ENTRYPOINT ["/bin/server"]
