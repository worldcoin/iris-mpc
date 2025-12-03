#!/usr/bin/env bash

# this script is meant to be called by Dockerfile.genesis.hawk

set -e

# exec replaces the shell with the binary
# $@ allows arguments to be forwarded from kubernetes
exec /bin/iris-mpc-hawk-genesis $@

