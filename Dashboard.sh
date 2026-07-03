#!/usr/bin/env bash
# Dashboard.sh — starts the dashboard server in background (if not already running) and opens browser
# Ported from Dashboard.bat
#
# Usage: ./Dashboard.sh

cd "$(dirname "$0")"

# Check if port 8899 is already listening
if ! lsof -i :8899 >/dev/null 2>&1; then
    echo "Starting Dashboard server on port 8899..."
    python3 dashboard.py >/dev/null 2>&1 &
    sleep 2
fi

open http://localhost:8899
