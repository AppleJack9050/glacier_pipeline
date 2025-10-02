#!/usr/bin/env bash
#
# monitor_runner.sh: Run a workload (custom command) while monitoring system & process metrics.
# Logs CSV-like lines to OUTFILE.
#
# Columns:
#   timestamp, sys_cpu_percent, sys_mem_used_mb, gpu_util_percent, vram_used_mb, proc_cpu_percent, proc_mem_mb, proc_vram_mb
#
# Examples:
#   ./monitor_runner.sh -i 5 -o run.log -c "python train.py --epochs 10"
#   ./monitor_runner.sh -i 2 -t 600 -- python train.py --epochs 10
#

set -euo pipefail

# ---- Defaults ----
INTERVAL=30
OUTFILE="monitor.log"
VERBOSE=0
QUIET=0
TIMEOUT_SECS=0
CUSTOM_CMD=""

# ---- Logging helper ----
log() {
  local level="$1"; shift
  local msg="$*"
  if [[ "$QUIET" -eq 1 && "$level" == "INFO" ]]; then
    return
  fi
  if [[ "$VERBOSE" -eq 1 || "$level" != "INFO" ]]; then
    echo "[$level] $msg"
  fi
}

# ---- Help ----
show_help() {
  cat << EOF
Usage: $0 [-i interval_in_seconds] [-o logfile] [-c "command"] [-v|-q] [-t timeout_seconds] [-h] [-- command args...]

Options:
  -i    Sampling interval (default: 30s)
  -o    Output log file (default: monitor.log)
  -c    Custom command to run (quoted). Alternative: use '--' then command args
  -v    Verbose logging
  -q    Quiet mode (suppress info logs)
  -t    Timeout in seconds (kill workload if exceeded; uses 'timeout' if present)
  -h    Show this help

Examples:
  $0 -i 5 -o run.log -c "python train.py --epochs 10"
  $0 -i 2 -t 600 -- python train.py --epochs 10
EOF
  exit 1
}

# ---- Parse options ----
while getopts ":i:o:c:vqt:h" opt; do
  case $opt in
    i) INTERVAL="$OPTARG" ;;
    o) OUTFILE="$OPTARG" ;;
    c) CUSTOM_CMD="$OPTARG" ;;
    v) VERBOSE=1 ;;
    q) QUIET=1 ;;
    t) TIMEOUT_SECS="$OPTARG" ;;
    h) show_help ;;
    \?) echo "Invalid option: -$OPTARG" >&2; show_help ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; show_help ;;
  esac
done
shift $((OPTIND-1))

# Allow passing command after '--'
if [[ $# -gt 0 && -z "$CUSTOM_CMD" ]]; then
  # Build a safely-quoted command string from the remaining args
  CUSTOM_CMD=$(printf "%q " "$@")
fi

# ---- Workload launcher (runs in its own process group) ----
run_workload() {
  if [[ -z "$CUSTOM_CMD" ]]; then
    if ! command -v stress-ng >/dev/null 2>&1; then
      echo "[ERR] No custom command and 'stress-ng' not found. Provide -c or install stress-ng." >&2
      return 127
    fi
    log INFO "Running default workload: stress-ng --cpu 8 --timeout 60s"
    exec stress-ng --cpu 8 --timeout 60s
  else
    log INFO "Running custom command: $CUSTOM_CMD"
    exec bash -lc "$CUSTOM_CMD"
  fi
}

# ---- Start time ----
START_TS=$(date +%s)

# ---- Init log file with header ----
if [[ ! -f "$OUTFILE" ]]; then
  echo "timestamp, sys_cpu_percent, sys_mem_used_mb, gpu_util_percent, vram_used_mb, proc_cpu_percent, proc_mem_mb, proc_vram_mb" > "$OUTFILE"
  log INFO "Created new log file: $OUTFILE"
else
  log INFO "Appending to existing log file: $OUTFILE"
fi

# ---- Detect GPU availability ----
GPU_AVAIL=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_AVAIL=1
fi

# ---- System CPU usage helper (mpstat preferred) ----
cpu_percent() {
  if command -v mpstat >/dev/null 2>&1; then
    local idle
    idle=$(mpstat 1 1 | awk '/Average:/ && $2 ~ /all/ {print $(NF)}')
    if [[ -z "$idle" ]]; then
      idle=$(mpstat 1 1 | awk '/all/ {v=$NF} END{print v}')
    fi
    awk -v idle="$idle" 'BEGIN { if (idle ~ /^[0-9.]+$/) { printf "%.1f", (100 - idle) } else { print "N/A" } }'
  else
    echo "N/A"
  fi
}

# ---- System RAM used (MB) ----
mem_used_mb() {
  if command -v free >/dev/null 2>&1; then
    free -m | awk '/Mem:/ {print $3}'
  else
    echo "N/A"
  fi
}

# ---- Process metrics (CPU %, RSS MB) ----
proc_cpu_percent() {
  local pid="$1"
  if ps -p "$pid" -o %cpu= >/dev/null 2>&1; then
    ps -p "$pid" -o %cpu= | awk '{printf "%.1f", $1}'
  else
    echo "N/A"
  fi
}

proc_mem_mb() {
  local pid="$1"
  if ps -p "$pid" -o rss= >/dev/null 2>&1; then
    # rss is in KB
    ps -p "$pid" -o rss= | awk '{printf "%.0f", $1/1024}'
  else
    echo "N/A"
  fi
}

# ---- Per-process VRAM used (MB) via nvidia-smi compute-apps ----
proc_vram_mb() {
  local pid="$1"
  if [[ "$GPU_AVAIL" -eq 1 ]]; then
    # Sum VRAM across GPUs if multiple entries exist for the PID
    local sum
    sum=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
          | awk -F',' -v p="$pid" 'BEGIN{s=0} $1+0==p {gsub(/ /,"",$2); s+=($2+0)} END{print s}')
    if [[ -z "$sum" ]]; then
      echo "0"
    else
      echo "$sum"
    fi
  else
    echo "0"
  fi
}

# ---- Overall GPU metrics ----
gpu_util_percent() {
  if [[ "$GPU_AVAIL" -eq 1 ]]; then
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "N/A"
  else
    echo "N/A"
  fi
}

vram_used_mb() {
  if [[ "$GPU_AVAIL" -eq 1 ]]; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0"
  else
    echo "0"
  fi
}

# ---- Cleanup on exit (kill process group) ----
CLEANED=0
cleanup() {
  if [[ $CLEANED -eq 1 ]]; then return; fi
  CLEANED=1
  if [[ -n "${PGID:-}" && "$PGID" =~ ^-?[0-9]+$ ]]; then
    log INFO "Stopping workload process group: $PGID"
    # First try SIGTERM, then after grace send SIGKILL
    kill -TERM "-$PGID" 2>/dev/null || true
    sleep 2
    kill -KILL "-$PGID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

# ---- Launch workload in background, new process group ----
# Using setsid to create a new session; $! will be the leader PID
if (( TIMEOUT_SECS > 0 )) && command -v timeout >/dev/null 2>&1; then
  log INFO "Applying timeout: ${TIMEOUT_SECS}s"
  setsid bash -c 'run_workload' &
  PID=$!
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  # Wrap with timeout in a watcher that kills the group if timeout triggers
  (
    if ! timeout "${TIMEOUT_SECS}s" bash -c "wait $PID"; then
      log WARN "Timeout reached; terminating workload group $PGID"
      kill -TERM "-$PGID" 2>/dev/null || true
    fi
  ) &
else
  if (( TIMEOUT_SECS > 0 )); then
    log WARN "timeout command not found; proceeding without enforcing -t"
  fi
  setsid bash -c 'run_workload' &
  PID=$!
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
fi

log INFO "Workload PID: $PID (PGID: $PGID)"

# ---- Monitoring loop ----
while kill -0 "$PID" 2>/dev/null; do
  TS=$(date +%s)

  SYS_CPU=$(cpu_percent)
  SYS_MEM_MB=$(mem_used_mb)
  GPU_UTIL=$(gpu_util_percent)
  GPU_MEM=$(vram_used_mb)

  PROC_CPU=$(proc_cpu_percent "$PID")
  PROC_MEM=$(proc_mem_mb "$PID")
  PROC_VRAM=$(proc_vram_mb "$PID")

  echo "${TS}, ${SYS_CPU}, ${SYS_MEM_MB}, ${GPU_UTIL}, ${GPU_MEM}, ${PROC_CPU}, ${PROC_MEM}, ${PROC_VRAM}" >> "$OUTFILE"

  sleep "$INTERVAL"
done

# ---- Wrap up ----
set +e
wait "$PID"
EXIT_STATUS=$?
set -e
END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
HMS=$(printf '%02d:%02d:%02d' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

log INFO "Workload exited with status: $EXIT_STATUS"
echo "# exit_status=${EXIT_STATUS}, duration_s=${DURATION}, duration_hms=${HMS}" >> "$OUTFILE"

# Ensure cleanup (trap will run too)
cleanup
