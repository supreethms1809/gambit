#!/usr/bin/env bash
# Watch the seed-major queue until seed 0 fully completes (training + val
# games), then stop the queue driver so seeds 1-4 wait for the G0/G1 review.
#
# The driver prints "seed 0: done (...)" only after both eval launchers exit,
# so that line is the completion signal, success or failure. Stopping the
# driver is the load-bearing action; the SEED0_* line is for the notification
# watching this script's stdout.
#
# Process matching is anchored to real interpreter command lines, so the
# shell-tool sandbox wrappers (whose command lines merely contain these
# strings) are never matched. A PID is only signalled after its /proc
# command line is re-verified, so PID reuse cannot misdirect a kill.
#
# Runs as a background shell job: `bash scripts/monitor_seed0.sh`.
set -u
cd "$(dirname "$0")/.."
LOG=results/paper/logs/seed_queue.log
DRIVER_RE='^bash scripts/run_seed_queue\.sh$'
LAUNCH_RE='^/[^ ]*python -u /[^ ]*scripts/launch_paper_(training|eval)\.py'

cmdline_matches() { # pid regex: does this pid's command line still match?
  local pid="$1" re="$2" cmd
  # A command line ends in a null byte, which becomes a trailing space that
  # would defeat a `$`-anchored pattern while pgrep (which strips it) matches.
  cmd=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | sed 's/ *$//') || return 1
  [ -n "$cmd" ] || return 1
  printf '%s' "$cmd" | grep -qE "$re"
}

driver_pids() {
  for p in $(pgrep -f "$DRIVER_RE" || true); do
    cmdline_matches "$p" "$DRIVER_RE" && echo "$p"
  done
}

DRIVER_PID=$(driver_pids | head -n 1 || true)
echo "monitor_seed0: watching $LOG (queue driver pid ${DRIVER_PID:-none})"
if [ -z "${DRIVER_PID:-}" ]; then
  echo "monitor_seed0: no queue driver running and seed 0 not done"
  echo "SEED0_DRIVER_GONE"
  exit 1
fi
while true; do
  if grep -q "seed 0: done" "$LOG" 2>/dev/null; then
    if cmdline_matches "$DRIVER_PID" "$DRIVER_RE"; then
      kill "$DRIVER_PID" 2>/dev/null || true
      echo "monitor_seed0: stopped queue driver pid $DRIVER_PID"
    else
      echo "monitor_seed0: driver pid $DRIVER_PID already changed hands, leaving it alone"
    fi
    sleep 10
    # A seed-1 launcher started in the gap would also need stopping. Cell
    # processes carry --one and seed-0 work is already waited on, so only a
    # parent launcher for another seed is killed here.
    for p in $(pgrep -f "$LAUNCH_RE" || true); do
      cmdline_matches "$p" "$LAUNCH_RE" || continue
      cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
      case "$cmd" in
        *"--one"*|*"--seeds 0"*) continue ;;
        *) kill "$p" 2>/dev/null || true; echo "monitor_seed0: stopped slipped launcher pid $p: $cmd" ;;
      esac
    done
    grep "seed 0: done" "$LOG" | tail -n 1
    echo "monitor_seed0: train done markers for seed 0: $(ls results/paper/logs/train/*seed0.done 2>/dev/null | wc -l)/22"
    echo "monitor_seed0: val done markers for seed 0: $(ls results/paper/runs/_markers/val/*seed0.done 2>/dev/null | wc -l)/14"
    echo "SEED0_COMPLETE"
    exit 0
  fi
  if ! cmdline_matches "$DRIVER_PID" "$DRIVER_RE"; then
    echo "monitor_seed0: queue driver pid $DRIVER_PID is gone before seed 0 finished"
    tail -n 5 "$LOG" 2>/dev/null || true
    echo "SEED0_DRIVER_GONE"
    exit 1
  fi
  sleep 30
done
