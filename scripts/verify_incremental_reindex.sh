#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/verify_incremental_reindex.sh [options]

Verifies automatic incremental cache update behavior.

Options:
  --tmp-dir PATH         Use an existing temporary workspace.
  --cache-dir PATH       Override cache directory (default under tmp dir).
  --keep-workdir         Do not delete temporary workspace.
  --help                 Show this message.
USAGE
}

TMP_WORKDIR=""
CACHE_DIR_OVERRIDE=""
KEEP_WORKDIR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmp-dir)
      TMP_WORKDIR=$2
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR_OVERRIDE=$2
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
ALZABO_BIN="${ALZABO_BIN:-alzabo}"
TIME_BIN="${TIME_BIN:-$(command -v /usr/bin/time || command -v time)}"

cd "${ROOT_DIR}"

TMP_WORKDIR="${TMP_WORKDIR:-$(mktemp -d -t alzabo-incremental-XXXXXX)}"
TRANSCRIPTS_DIR="${TMP_WORKDIR}/transcripts"
CODEX_DIR="${TMP_WORKDIR}/codex"
CACHE_DIR="${CACHE_DIR_OVERRIDE:-${TMP_WORKDIR}/cache}"

log() {
  echo
  echo "==> $*"
}

cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${TMP_WORKDIR}"
  fi
}
trap cleanup EXIT

require_file_contains() {
  local file_path=$1
  local expected=$2
  if ! grep -q -- "${expected}" "${file_path}"; then
    echo "FAIL: expected '${expected}' in ${file_path}"
    echo "--- ${file_path} ---"
    cat "${file_path}"
    exit 1
  fi
}

require_file_not_contains() {
  local file_path=$1
  local unexpected=$2
  if grep -q -- "${unexpected}" "${file_path}"; then
    echo "FAIL: expected not to see '${unexpected}' in ${file_path}"
    echo "--- ${file_path} ---"
    cat "${file_path}"
    exit 1
  fi
}

assert_dir() {
  local dir_path=$1
  if [[ ! -d "${dir_path}" ]]; then
    echo "FAIL: missing directory ${dir_path}"
    exit 1
  fi
}

mkdir -p "${TRANSCRIPTS_DIR}" "${CODEX_DIR}" "${CACHE_DIR}"

LOG_DIR="${TMP_WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

log "workspace: ${TMP_WORKDIR}"
log "transcripts: ${TRANSCRIPTS_DIR}"
log "codex: ${CODEX_DIR}"
log "cache-dir: ${CACHE_DIR}"

cat > "${TRANSCRIPTS_DIR}/initial.jsonl" <<'JSONL'
{"type":"user","sessionId":"session-old","message":{"content":"alpha query for incremental baseline"}}
{"type":"assistant","sessionId":"session-old","message":{"content":"baseline response"}}
JSONL

log "step 1: cold full reindex"
"${TIME_BIN}" -p "${UV_BIN}" run "${ALZABO_BIN}" \
  reindex \
  --transcripts-dir "${TRANSCRIPTS_DIR}" \
  --codex-dir "${CODEX_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  >"${LOG_DIR}/reindex.out" 2>"${LOG_DIR}/reindex.err"

require_file_contains "${LOG_DIR}/reindex.err" "reindexing transcripts"
require_file_contains "${LOG_DIR}/reindex.err" "reindex complete"

log "step 2: baseline status (cache exists)"
"${UV_BIN}" run "${ALZABO_BIN}" \
  status \
  --format json \
  --transcripts-dir "${TRANSCRIPTS_DIR}" \
  --codex-dir "${CODEX_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  >"${LOG_DIR}/status_before.out" \
  2>"${LOG_DIR}/status_before.err"

require_file_contains "${LOG_DIR}/status_before.err" "loading from cache"
require_file_contains "${LOG_DIR}/status_before.out" "total_turns"

log "step 2.5: wait out debounce window before mutation check"
sleep 31

log "step 3: add changed source file and run search"
cat > "${TRANSCRIPTS_DIR}/changed.jsonl" <<'JSONL'
{"type":"user","sessionId":"session-new","message":{"content":"bravo query should trigger incremental update"}}
{"type":"assistant","sessionId":"session-new","message":{"content":"incremental response"}}
JSONL

"${UV_BIN}" run "${ALZABO_BIN}" \
  search "bravo" \
  --format json \
  --transcripts-dir "${TRANSCRIPTS_DIR}" \
  --codex-dir "${CODEX_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  >"${LOG_DIR}/search_after_change.out" \
  2>"${LOG_DIR}/search_after_change.err"

require_file_contains "${LOG_DIR}/search_after_change.err" "files changed, updating index"
require_file_not_contains "${LOG_DIR}/search_after_change.err" "indexing transcripts"
require_file_contains "${LOG_DIR}/search_after_change.out" "bravo"

log "step 4: ensure manifest updated with changed file"
MANIFEST="${CACHE_DIR}/manifest.json"
assert_dir "${CACHE_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "FAIL: manifest not found at ${MANIFEST}"
  exit 1
fi
require_file_contains "${MANIFEST}" "changed.jsonl"

log "step 5: debounce check (within 30s should skip source scan)"
"${UV_BIN}" run "${ALZABO_BIN}" \
  search "alpha" \
  --format text \
  --transcripts-dir "${TRANSCRIPTS_DIR}" \
  --codex-dir "${CODEX_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  >"${LOG_DIR}/search_debounce.out" \
  2>"${LOG_DIR}/search_debounce.err"

require_file_contains "${LOG_DIR}/search_debounce.err" "cache checked recently; skipping source scan"
require_file_not_contains "${LOG_DIR}/search_debounce.err" "files changed, updating index"

log "manual incremental verification complete"
echo "Artifacts:" 
printf '  logs: %s\n' "${LOG_DIR}"
echo "Done"
