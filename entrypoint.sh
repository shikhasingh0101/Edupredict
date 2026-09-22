#!/bin/sh
set -eu

uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true &
UI_PID=$!

trap 'kill "$API_PID" "$UI_PID" 2>/dev/null || true' INT TERM EXIT

wait -n "$API_PID" "$UI_PID"
