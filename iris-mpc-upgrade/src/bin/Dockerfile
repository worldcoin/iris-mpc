FROM --platform=linux/amd64 ghcr.io/worldcoin/iris-mpc:146c2cae43dbeb586144d9d37d152a6b2bfacdd4
USER non-root
ENTRYPOINT "reshare-server --party-id 2 --sender1-party-id 0 --sender2-party-id 1 --bind-addr 0.0.0.0:7000 --environment testing --db-url postgres://postgres:postgres@new-db-4:6203 --batch-size 100 --healthcheck-port 3000"
