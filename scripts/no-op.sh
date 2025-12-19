#!/bin/bash

# Shutdown handler script that responds to signals and runs a continuous loop

# Flag to control the main loop
RUNNING=true

# Function to handle shutdown signals
shutdown_handler() {
    echo "Received shutdown signal. Cleaning up..."
    RUNNING=false
}

# Trap common shutdown signals
trap shutdown_handler SIGTERM SIGINT SIGHUP SIGQUIT

echo "Starting shutdown handler script (PID: $$)"
echo "Press Ctrl+C or send SIGTERM to gracefully shutdown"

# Main loop
while [ "$RUNNING" = true ]; do
    sleep 1
done

echo "Shutdown complete. Exiting."
exit 0
