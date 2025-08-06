#!/bin/bash

# tmux起動 & 縦3画面に割る

# tmuxセッションを作成（既存の場合は接続）
tmux new-session -d -s texture_generation

# 最初のペインでComfyUIを起動
tmux send-keys -t texture_generation:0.0 "cd ComfyUI && uv run python main.py" C-m

# 2番目のペインを作成してMake-It-Animatableを起動
tmux split-window -h -t texture_generation:0.0
tmux send-keys -t texture_generation:0.1 "cd Make-It-Animatable && uv run python api_server/main.py" C-m

# 3番目のペインを作成してworkflow_apiを起動
tmux split-window -h -t texture_generation:0.1
tmux send-keys -t texture_generation:0.2 "cd workflow_api && uv run python api_server.py" C-m

# ペインのサイズを均等に調整
tmux select-layout -t texture_generation:0 even-horizontal

# tmuxセッションに接続
tmux attach-session -t texture_generation