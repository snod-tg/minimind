#!/usr/bin/env bash
set -u
RUN=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage09_compile_20261009
export PYTHONUNBUFFERED=1
bash "$RUN/run_train.sh" > "$RUN/pretrain_ep6_top2_compile.log" 2>&1
training_status=$?
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN/training_finished_utc.txt"
printf '%s\n' "$training_status" > "$RUN/training_exit_code.txt"
exit "$training_status"
