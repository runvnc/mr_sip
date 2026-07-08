#!/usr/bin/env bash
# collect_sip_logs.sh - bundle the mr_sip dead-air / latency diagnostic logs
# into one timestamped tar.gz for easy download off the H200.
#
# Usage:
#   ./collect_sip_logs.sh [--reset] [output_dir] [extra_file_or_glob ...]
#
#   --reset       after archiving, truncate the 4 /tmp/sip diagnostic logs so
#                 the next test run starts clean (does NOT touch mindroot.log).
#   output_dir    where to write the tarball (default: /tmp).
#   extra args    any additional files/globs to include, e.g. a session chat
#                 log path:  ./collect_sip_logs.sh /tmp /path/to/chatlog.json
#
# Examples:
#   ./collect_sip_logs.sh                       # bundle to /tmp/sip_logs_<ts>.tar.gz
#   ./collect_sip_logs.sh --reset               # bundle, then blank the logs
#   ./collect_sip_logs.sh /tmp "/workspace/data/chat/*/chatlog.json"
set -u

RESET=0
if [ "${1:-}" = "--reset" ]; then RESET=1; shift; fi

OUT_DIR="${1:-/tmp}"
if [ $# -gt 0 ]; then shift; fi   # remaining args = extra paths/globs

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="${OUT_DIR%/}/sip_logs_${STAMP}.tar.gz"

# The core diagnostic logs (see the dead-air log map).
CORE_TMP_LOGS=(
  /tmp/smart_turn_v3_stt.log
  /tmp/sip_e2e_latency.log
  /tmp/sip_deadair.log
  /tmp/silero_cohere_stt.log
  /tmp/sip_hangup.log
)
OTHER_LOGS=(
  /workspace/logs/mindroot.log
)

CANDIDATES=( "${CORE_TMP_LOGS[@]}" "${OTHER_LOGS[@]}" )
for x in "$@"; do CANDIDATES+=( $x ); done   # unquoted so globs expand

echo "Collecting logs -> ${DEST}"
INCLUDE=()
for f in "${CANDIDATES[@]}"; do
  if [ -f "$f" ]; then
    INCLUDE+=( "$f" )
    printf '  + %-40s %s\n' "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
  else
    printf '  - missing: %s\n' "$f"
  fi
done

if [ "${#INCLUDE[@]}" -eq 0 ]; then
  echo 'No log files found - nothing to archive.' >&2
  exit 1
fi

# -h dereferences symlinks; tar strips the leading '/' (harmless warning).
tar -czhf "$DEST" "${INCLUDE[@]}" 2>/dev/null
echo
echo "Wrote ${DEST}  ($(du -h "$DEST" 2>/dev/null | cut -f1))"

if [ "$RESET" -eq 1 ]; then
  echo 'Resetting /tmp/sip diagnostic logs (mindroot.log left untouched)...'
  for f in "${CORE_TMP_LOGS[@]}"; do
    [ -f "$f" ] && : > "$f" && echo "  cleared $f"
  done
fi
