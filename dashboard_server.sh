#!/usr/bin/env bash
# dashboard_server.sh — launches the dashboard server in the foreground on macOS
# Ported from dashboard_server.bat
#
# Usage: ./dashboard_server.sh

cd "$(dirname "$0")"
python3 dashboard.py
