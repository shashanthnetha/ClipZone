#!/usr/bin/env bash
# study_creators.sh — CreatorStudy study loop runner on macOS
# Ported from study_creators.bat
#
# Usage: ./study_creators.sh

cd "$(dirname "$0")"

echo "" >> study_run.log
echo "============================================================" >> study_run.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] CREATOR STUDY run STARTING" >> study_run.log

# 1) Make sure Docker + Postiz are up
./ensure_postiz.sh >> study_run.log 2>&1

# 2) Run the competitor-intelligence analyst headless
claude -p "Execute the creator study run: follow every instruction in creator_study_prompt.md. Work fully autonomously and do not ask any questions." --dangerously-skip-permissions >> study_run.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] CREATOR STUDY run FINISHED" >> study_run.log
