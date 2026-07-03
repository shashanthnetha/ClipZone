#!/usr/bin/env bash
# learn_shorts.sh — weekly self-improvement learning run on macOS
# Ported from learn_shorts.bat
#
# Usage: ./learn_shorts.sh

cd "$(dirname "$0")"

echo "" >> learn_run.log
echo "============================================================" >> learn_run.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] LEARNING run STARTING" >> learn_run.log

# 1) Make sure Docker + Postiz are up
./ensure_postiz.sh >> learn_run.log 2>&1

# 2) Run the learning analyst headless
claude -p "Execute the weekly learning run: follow every instruction in learn_and_improve_prompt.md. Work fully autonomously and do not ask any questions." --dangerously-skip-permissions >> learn_run.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] LEARNING run FINISHED" >> learn_run.log
