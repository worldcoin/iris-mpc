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
    protobuf-compiler \
    wget

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH "/root/.cargo/bin:${PATH}"
ENV RUSTUP_HOME "/root/.rustup"
ENV CARGO_HOME "/root/.cargo"
RUN rustup toolchain install 1.89.0
RUN rustup default 1.89.0
RUN rustup component add cargo
RUN cargo install cargo-build-deps \
    && cargo install cargo-edit --version 0.13.6 --locked

RUN curl -o /tmp/ca_smpcv2_party_stage_0.crt https://wf-smpcv2-ca-stage-eu-north-1.s3.eu-north-1.amazonaws.com/ca_party_stage_1.stage.smpcv2.worldcoin.dev.pem && \
    curl -o /tmp/ca_smpcv2_party_stage_1.crt https://wf-smpcv2-ca-stage-eu-north-1.s3.eu-north-1.amazonaws.com/ca_party_stage_2.stage.smpcv2.worldcoin.dev.pem && \
    curl -o /tmp/ca_smpcv2_party_stage_2.crt https://wf-smpcv2-ca-stage-eu-north-1.s3.eu-north-1.amazonaws.com/ca_party_stage_3.stage.smpcv2.worldcoin.dev.pem && \
    curl -o /tmp/ca_smpcv2_party_prod_0.crt https://wf-smpcv2-ca-prod-eu-north-1.s3.eu-north-1.amazonaws.com/ca_party_prod_1.smpcv2.worldcoin.org.pem && \
    curl -o /tmp/ca_smpcv2_party_prod_1.crt https://wf-smpcv2-ca-prod-eu-north-1.s3.eu-north-1.amazonaws.com/ca_party_prod_2.smpcv2.worldcoin.org.pem && \
    curl -o /tmp/ca_smpcv2_party_prod_2.crt https://wf-smpcv2-ca-prod-eu-north-1.s3.eu-north-1.amazonaws.com/ca_party_prod_3.smpcv2.worldcoin.org.pem

FROM --platform=linux/amd64 build-image as build-app
WORKDIR /src/gpu-iris-mpc
COPY . .
RUN cargo build -p iris-mpc-bins --release --target x86_64-unknown-linux-gnu --bin nccl --bin iris-mpc-gpu --bin client --bin key-manager --bin reshare-server --bin reshare-client

FROM --platform=linux/amd64 public.ecr.aws/deep-learning-containers/base:12.8.0-gpu-py312-cu128-ubuntu22.04-ec2-v1.63
ENV DEBIAN_FRONTEND=noninteractive

# Include client, server and key-manager, upgrade-client and upgrade-server binaries
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/nccl /bin/nccl
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/iris-mpc-gpu /bin/iris-mpc-gpu
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/client /bin/client
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/key-manager /bin/key-manager
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/reshare-server /bin/reshare-server
COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/reshare-client /bin/reshare-client
COPY --from=build-image /tmp/ca_smpcv2_party_*.crt /usr/local/share/ca-certificates/

USER 65534
ENTRYPOINT ["/bin/iris-mpc-gpu"]
