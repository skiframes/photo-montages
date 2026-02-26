#!/bin/bash
# Restart Flask edge server with ARM Homebrew Python
# Logs to /tmp/flask.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/homebrew/bin/python3"
LOG_FILE="/tmp/flask.log"

echo "Stopping existing Flask server..."
lsof -ti:5000 | xargs kill -9 2>/dev/null
sleep 1

echo "Starting Flask server..."
cd "$SCRIPT_DIR"
nohup "$PYTHON" app.py >> "$LOG_FILE" 2>&1 &

sleep 2

if curl -s http://localhost:5000/api/sessions > /dev/null 2>&1; then
    echo "Flask server started successfully on port 5000"
    echo "Logs: $LOG_FILE"
    tail -3 "$LOG_FILE"
else
    echo "Failed to start Flask server. Check $LOG_FILE for errors."
    tail -10 "$LOG_FILE"
    exit 1
fi
