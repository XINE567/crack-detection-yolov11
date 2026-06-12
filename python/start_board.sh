#!/bin/sh
# RK3588 启动脚本

export BACKEND_URL="${BACKEND_URL:-http://192.168.73.181:5000/upload_result}"
export FRONTEND_WS_URL="${FRONTEND_WS_URL:-ws://192.168.73.132:8765}"
export FRONTEND_FPS="${FRONTEND_FPS:-5}"

cd /userdata && python3 crack_inspection_full.py
