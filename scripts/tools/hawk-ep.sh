#!/bin/bash
# Add 10ms delay to outbound traffic
INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
tc qdisc add dev $INTERFACE root netem delay 10ms

# Start the main application
exec "$@"
