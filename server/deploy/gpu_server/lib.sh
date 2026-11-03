#!/usr/bin/env bash
# 공통 함수 · 경로. 각 단계 스크립트가 맨 위에서 `source "$(dirname "$0")/lib.sh"` 한다.
#
# 경로 규칙
#   APP_ROOT   : 저장소 전체가 올라간 위치 (orchestrator/, rag_service/, indexing/, docs/ ...)
#   DEPLOY_DIR : 이 스크립트들이 있는 곳 = ${APP_ROOT}/deploy/gpu_server
#   ENV_FILE   : 운영 환경변수 파일 = ${DEPLOY_DIR}/.env  (03_deploy_code.sh가 만든다. git에 넣지 않는다)
#   APP_VENV   : 앱 서비스 공용 가상환경 = ${APP_ROOT}/app-env

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="${APP_ROOT:-$(cd "${DEPLOY_DIR}/../.." && pwd)}"
ENV_FILE="${DEPLOY_DIR}/.env"
APP_VENV="${APP_ROOT}/app-env"
VLLM_VENV="${APP_ROOT}/vllm-env"
# 3.11로 실측 검증했지만 3.11 전용 문법은 없다 — NCP 이미지에 3.12가 기본으로 있으면
# deadsnakes PPA를 새로 추가하지 않고 그걸 그대로 쓴다(01단계가 "이미 설치됨"으로 건너뜀).
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RUN_USER="${RUN_USER:-$(id -un)}"

log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)]\033[0m $*"; }
warn() { echo -e "\033[1;33m[경고]\033[0m $*"; }
die()  { echo -e "\033[1;31m[오류]\033[0m $*" >&2; exit 1; }

# .env를 읽어 현재 셸에 export 한다 (있을 때만)
load_env() {
    if [ -f "${ENV_FILE}" ]; then
        set -a
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        set +a
    fi
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "'$1' 명령이 없습니다. 이전 단계 스크립트를 먼저 실행하세요."; }

# 지정한 URL이 200을 줄 때까지 기다린다: wait_http <url> <최대초> [헤더]
wait_http() {
    local url="$1" max="$2" hdr="${3:-}" i=0
    while [ "$i" -lt "$max" ]; do
        if [ -n "$hdr" ]; then
            curl -fsS -m 3 -H "$hdr" "$url" >/dev/null 2>&1 && return 0
        else
            curl -fsS -m 3 "$url" >/dev/null 2>&1 && return 0
        fi
        sleep 2; i=$((i + 2))
    done
    return 1
}

# vLLM 가상환경 (답변 LLM vllm_llm · 인덱싱 VLM vllm_vlm 공용). 최초 1회 설치, 10분 이상 걸린다.
ensure_vllm_venv() {
    if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
        log "vLLM 가상환경 설치: ${VLLM_VENV} (최초 1회, 10분 이상)"
        "python${PYTHON_VERSION}" -m venv "${VLLM_VENV}"
        "${VLLM_VENV}/bin/pip" install --upgrade pip >/dev/null
        "${VLLM_VENV}/bin/pip" install vllm
    fi
    log "vLLM: $("${VLLM_VENV}/bin/vllm" --version 2>/dev/null | tail -1)"
}
