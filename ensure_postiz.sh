#!/usr/bin/env bash
# ensure_postiz.sh - make sure Docker + the Postiz containers are UP so the scripts can read the
# YouTube credentials on macOS.
#
# Ported from ensure_postiz.ps1.

# 1) Docker daemon up (launch Docker Desktop on macOS if needed)
if ! docker info >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] Docker daemon down -> launching Docker Desktop"
    open -a Docker
    deadline=$((SECONDS + 240))
    while ! docker info >/dev/null 2>&1 && [ $SECONDS -lt $deadline ]; do
        sleep 10
    done
fi
if ! docker info >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] ERROR: Docker still down after 4 min"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Docker is up"

# 2) resolve the Postiz directory
POSTIZ_DIR="${POSTIZ_DIR:-$HOME/auto posting tool/postiz-app-main}"
if [ ! -d "$POSTIZ_DIR" ]; then
    # try fallback
    if [ -d "$HOME/postiz-app-main" ]; then
        POSTIZ_DIR="$HOME/postiz-app-main"
    else
        echo "[$(date +%H:%M:%S)] WARNING: Postiz directory not found at $POSTIZ_DIR"
        echo "Please set the POSTIZ_DIR environment variable to the path of postiz-app-main"
    fi
fi

# 3) ensure the Postiz stack is running
if [ -d "$POSTIZ_DIR" ]; then
    echo "[$(date +%H:%M:%S)] Starting Postiz stack at $POSTIZ_DIR"
    pushd "$POSTIZ_DIR" >/dev/null
    docker compose up -d
    popd >/dev/null
else
    # fallback to just docker start if directory not found
    echo "[$(date +%H:%M:%S)] Directory not found, attempting direct container start"
    docker start postiz postiz-postgres >/dev/null 2>&1
fi

# 4) wait for Postgres
deadline=$((SECONDS + 120))
ready=false
while [ $SECONDS -lt $deadline ]; do
    sleep 4
    if docker exec postiz-postgres pg_isready -U postiz-user -d postiz-db-local >/dev/null 2>&1; then
        ready=true
        break
    fi
done
echo "[$(date +%H:%M:%S)] Postiz DB ready: $ready"

# 5) self-heal PM2 backend if needed
on3000=$(docker exec postiz sh -c "netstat -tln 2>/dev/null | grep -c ':3000' || ss -tln 2>/dev/null | grep -c ':3000'" 2>/dev/null || echo "0")
if [ "${on3000//[[:space:]]/}" = "0" ]; then
    echo "[$(date +%H:%M:%S)] backend not listening on :3000 -> restarting pm2 backend process"
    docker exec postiz sh -c "npx pm2 restart backend" >/dev/null 2>&1
fi
