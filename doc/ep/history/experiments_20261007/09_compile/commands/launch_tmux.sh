#!/usr/bin/env bash
set -euo pipefail
RUN=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009
SESSION=ep6_top2_compile_20261009
command -v tmux >/dev/null
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Existing session: $SESSION" >&2
  exit 1
fi
test ! -e "$RUN/training_exit_code.txt"
tmux new-session -d -s "$SESSION" -c "$RUN" "bash '$RUN/train_in_tmux.sh'"
tmux set-option -t "$SESSION" remain-on-exit on
tmux display-message -p -t "$SESSION" '#{session_name} #{pane_pid} #{pane_dead}'
