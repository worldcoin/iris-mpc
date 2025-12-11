#!/bin/bash
# Add 10ms delay to outbound traffic
INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
tc qdisc add dev $INTERFACE root handle 1: netem delay 10ms

tc qdisc add dev $INTERFACE parent 1: handle 10: tbf \
    rate 25gbit burst 1mbit latency 50ms

# Start the main application
exec "$@"
