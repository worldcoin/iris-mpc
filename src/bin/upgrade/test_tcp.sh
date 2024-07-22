mkdir -p out0
mkdir -p out1
mkdir -p out2
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --party-id 0 --threads 8 --eye left &
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8100 --party-id 1 --threads 8 --eye left &
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8200 --party-id 2 --threads 8 --eye left 
