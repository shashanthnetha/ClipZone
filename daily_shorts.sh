#!/usr/bin/env bash
# daily_shorts.sh — DailyShorts autonomous runner on macOS
# Ported from daily_shorts.bat
#
# Usage: ./daily_shorts.sh

# Navigate to the workspace directory
cd "$(dirname "$0")"

echo "" >> daily_run.log
echo "============================================================" >> daily_run.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DailyShorts run STARTING" >> daily_run.log

# 1) Make sure Docker + Postiz are up
./ensure_postiz.sh >> daily_run.log 2>&1

# 2) Run the autonomous producer headless
# On macOS, claude command is typically globally installed via npm or globally on PATH.
claude -p "Execute today's run: follow every instruction in daily_shorts_prompt.md. Work fully autonomously and do not ask any questions." --dangerously-skip-permissions >> daily_run.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DailyShorts run FINISHED" >> daily_run.log
