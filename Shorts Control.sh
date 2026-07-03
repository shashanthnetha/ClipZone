#!/usr/bin/env bash
# Shorts Control.sh — interactive control menu on macOS
# Ported from Shorts Control.bat
#
# Usage: ./Shorts\ Control.sh

cd "$(dirname "$0")"

show_menu() {
    clear
    echo ""
    echo "  =================================================="
    echo "      SHORTS AUTOMATION  -  one-click control (macOS)"
    echo "  =================================================="
    echo ""
    echo "     [1]  Make + POST a video NOW"
    echo "     [2]  Study top creators NOW  (grow playbook)"
    echo "     [3]  Learn from analytics NOW (improve SKILL.md)"
    echo "     [4]  Run ALL THREE now"
    echo ""
    echo "     [5]  Email me a DIGEST now (test the email)"
    echo "     [6]  Watch the live log (Ctrl+C to stop)"
    echo "     [7]  Status of every task"
    echo "     [0]  Exit"
    echo ""
    read -p "   Type a number and press Enter: " choice
}

while true; do
    show_menu
    case "$choice" in
        1)
            echo ""
            echo "   Starting: making + posting a video in background..."
            ./daily_shorts.sh >/dev/null 2>&1 &
            sleep 4
            ;;
        2)
            echo ""
            echo "   Starting: studying top creators in background..."
            ./study_creators.sh >/dev/null 2>&1 &
            sleep 4
            ;;
        3)
            echo ""
            echo "   Starting: learning from analytics in background..."
            ./learn_shorts.sh >/dev/null 2>&1 &
            sleep 4
            ;;
        4)
            echo ""
            echo "   Starting ALL THREE in background..."
            ./daily_shorts.sh >/dev/null 2>&1 &
            ./study_creators.sh >/dev/null 2>&1 &
            ./learn_shorts.sh >/dev/null 2>&1 &
            sleep 4
            ;;
        5)
            echo ""
            echo "   Sending digest email..."
            docker start postiz postiz-postgres >/dev/null 2>&1
            python3 daily_digest.py
            echo ""
            read -p "   Press Enter to continue..." temp
            ;;
        6)
            echo ""
            echo "   Tailing daily_run.log ... press Ctrl+C to stop."
            echo ""
            tail -n 30 -f daily_run.log
            ;;
        7)
            echo ""
            echo "Task Status (macOS processes):"
            echo "-----------------------------------"
            pgrep -lf "python3.*dashboard.py|claude.*daily_shorts|claude.*study_creators|claude.*learn_shorts" || echo "No background automation tasks running."
            echo ""
            read -p "   Press Enter to continue..." temp
            ;;
        0)
            exit 0
            ;;
        *)
            echo "Invalid choice."
            sleep 2
            ;;
    esac
done
