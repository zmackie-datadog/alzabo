#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/verify_cli_only.sh [options]

Options:
  --skip-tests       Skip `uv run pytest tests/`
  --skip-cli         Skip CLI end-to-end checks (help, search, status, list)
  --tmp-dir PATH     Use existing temp dir for workspace
  --cache-dir PATH   Use existing cache directory (under tmp-dir by default)
  --keep-workdir     Do not delete temporary workspace
  --help             Show this message
USAGE
}

RUN_TESTS=1
RUN_CLI=1
KEEP_WORKDIR=0
TMP_WORKDIR=""
CACHE_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    --skip-cli)
      RUN_CLI=0
      shift
      ;;
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
UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache-verify}"
TMP_WORKDIR="${TMP_WORKDIR:-$(mktemp -d -t alzabo-cli-verify-XXXXXX)}"
TRANSCRIPTS_DIR="${TMP_WORKDIR}/transcripts"
CODEX_DIR="${TMP_WORKDIR}/codex"
CACHE_DIR="${CACHE_DIR_OVERRIDE:-${TMP_WORKDIR}/cache}"
QUERY="hello world"
PASS=1

export UV_CACHE_DIR

cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${TMP_WORKDIR}"
  fi
}
trap cleanup EXIT

log() {
  echo
  echo "==> $*"
}

log "workspace: ${TMP_WORKDIR}"
log "transcripts: ${TRANSCRIPTS_DIR}"
log "codex: ${CODEX_DIR}"
log "cache-dir: ${CACHE_DIR}"

mkdir -p "${TRANSCRIPTS_DIR}" "${CODEX_DIR}" "${CACHE_DIR}"

if command -v "${UV_BIN}" >/dev/null 2>&1; then
  log "uv found: ${UV_BIN} (cache: ${UV_CACHE_DIR})"
else
  echo "ERROR: uv not found"
  exit 1
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  log "running pytest"
  (cd "${ROOT_DIR}" && "${UV_BIN}" run pytest tests/)
fi

if [[ "${RUN_CLI}" -eq 1 ]]; then
  cd "${ROOT_DIR}"
  log "cli --help"
  "${UV_BIN}" run "${ALZABO_BIN}" --help

  rm -rf "${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"

  log "search warmup #1"
  "${TIME_BIN}" -p "${UV_BIN}" run "${ALZABO_BIN}" \
    search "${QUERY}" \
    --transcripts-dir "${TRANSCRIPTS_DIR}" \
    --codex-dir "${CODEX_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${TMP_WORKDIR}/search1.out" \
    2>"${TMP_WORKDIR}/search1.stderr"
  cat "${TMP_WORKDIR}/search1.stderr"

  log "search warmup #2 (cache reuse validation)"
  "${TIME_BIN}" -p "${UV_BIN}" run "${ALZABO_BIN}" \
    search "${QUERY}" \
    --transcripts-dir "${TRANSCRIPTS_DIR}" \
    --codex-dir "${CODEX_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${TMP_WORKDIR}/search2.out" \
    2>"${TMP_WORKDIR}/search2.stderr"
  cat "${TMP_WORKDIR}/search2.stderr"

  if ! grep -Eq "loading from cache|cache loaded" "${TMP_WORKDIR}/search2.stderr"; then
    echo "FAIL: second search run did not show cache reuse"
    PASS=0
  fi

  if grep -qi "loading embedding model" "${TMP_WORKDIR}/search2.stderr"; then
    echo "FAIL: second search run shows embedding model load path"
    PASS=0
  fi

  log "status check"
  "${TIME_BIN}" -p "${UV_BIN}" run "${ALZABO_BIN}" \
    status --no-cache \
    --transcripts-dir "${TRANSCRIPTS_DIR}" \
    --codex-dir "${CODEX_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${TMP_WORKDIR}/status.out" \
    2>"${TMP_WORKDIR}/status.stderr"
  cat "${TMP_WORKDIR}/status.out"

  log "list check"
  "${TIME_BIN}" -p "${UV_BIN}" run "${ALZABO_BIN}" \
    list --limit 3 --quiet \
    --transcripts-dir "${TRANSCRIPTS_DIR}" \
    --codex-dir "${CODEX_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    >"${TMP_WORKDIR}/list.out" \
    2>"${TMP_WORKDIR}/list.stderr"
  cat "${TMP_WORKDIR}/list.out"
fi

if [[ "${PASS}" -ne 1 ]]; then
  echo
  echo "Verification failed."
  exit 1
fi

echo
echo "Verification passed. Artifacts:"
echo "  ${TMP_WORKDIR}"
