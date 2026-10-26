#!/bin/bash
# Peak-VRAM measurement poller.
#
# [Origin] verbatim copy of poll_vram.sh of an earlier calibration run.
# Copied and wired for the "measurement on the first chunk" of the baseline generation (2026-09-16).
# The logic is not changed at all (an existing verified implementation is reused).
#
# Usage: poll_vram.sh <PID> <output_csv> [interval_sec=5] [hard_limit_min=120]
#
# Self-termination conditions:
#   (a) stops when the monitored PID has ended (decided with kill -0; pgrep -f is not used)
#   (b) always ends at the hard limit (the caller passes hard_limit_min. Following the instruction not to
#       extrapolate the one-chunk result of another arm, in the caller run_chunk_tilt_baseline.sh the generation itself is waited for by a
#       foreground `wait`; the expiry of the hard limit of this poller only "stops the sampling"
#       and does not kill the generation process itself)
#
# Output: appends a CSV "timestamp_utc,pid,used_memory_mib" to <output_csv>.
#   Of the output of nvidia-smi --query-compute-apps, only the rows that match the monitored PID are recorded
#   (the per-process used_memory is taken, not memory.used of the whole GPU).

set -uo pipefail

PID="${1:?usage: poll_vram.sh <PID> <output_csv> [interval_sec] [hard_limit_min]}"
OUT_CSV="${2:?usage: poll_vram.sh <PID> <output_csv> [interval_sec] [hard_limit_min]}"
INTERVAL="${3:-5}"
HARD_LIMIT_MIN="${4:-120}"
HARD_LIMIT_SEC=$((HARD_LIMIT_MIN * 60))

START_TS=$(date +%s)
echo "timestamp_utc,pid,used_memory_mib" > "${OUT_CSV}"

while kill -0 "${PID}" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TS))
  if [ "${ELAPSED}" -ge "${HARD_LIMIT_SEC}" ]; then
    echo "poll_vram.sh: hard limit ${HARD_LIMIT_MIN} min reached for PID ${PID}; exiting" >&2
    break
  fi

  # nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
  # -> 1 row per process. Format "pid, used_memory_mib" (at most one blank as the separator per row).
  QUERY_OUT=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -n "${QUERY_OUT}" ]; then
    while IFS=',' read -r qpid qmem; do
      qpid_trim=$(echo "${qpid}" | tr -d '[:space:]')
      qmem_trim=$(echo "${qmem}" | tr -d '[:space:]')
      if [ "${qpid_trim}" = "${PID}" ] && [ -n "${qmem_trim}" ]; then
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        echo "${ts},${qpid_trim},${qmem_trim}" >> "${OUT_CSV}"
      fi
    done <<< "${QUERY_OUT}"
  fi

  sleep "${INTERVAL}"
done

echo "poll_vram.sh: exiting for PID ${PID} (process ended or hard limit reached)" >&2
