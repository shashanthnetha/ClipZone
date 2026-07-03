#!/usr/bin/env bash
# digest.sh — emails the daily summary log report on macOS
# Ported from digest.bat
#
# Usage: ./digest.sh

cd "$(dirname "$0")"

echo "" >> digest_run.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DIGEST run" >> digest_run.log

# Ensure Postgres & Postiz containers are running
docker start postiz postiz-postgres >/dev/null 2>&1

# Run digest script
python3 daily_digest.py >> digest_run.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DIGEST done" >> digest_run.log
