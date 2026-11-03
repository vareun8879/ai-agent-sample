#!/usr/bin/env bash
# 1단계 — OS 기본 패키지 · Python(PYTHON_VERSION, 기본 3.12) · NVIDIA 드라이버 확인
#
#   Ubuntu 22.04 / 24.04 (NCP GPU 서버 이미지) 기준. sudo 필요.
#   사용법: bash 01_setup_system.sh
source "$(dirname "$0")/lib.sh"

log "apt 패키지"
sudo apt-get update -y
sudo apt-get install -y \
    build-essential git curl wget unzip jq \
    software-properties-common ca-certificates \
    libgl1 libglib2.0-0

# ── Python ${PYTHON_VERSION} (기본 3.12 — NCP 이미지에 이미 있음. 코드는 3.11에서 검증했지만 3.11 전용 문법은 없다) ──
if ! command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    log "Python ${PYTHON_VERSION} 설치 (deadsnakes PPA)"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
else
    log "Python ${PYTHON_VERSION} 이미 설치됨"
fi
# 인터프리터가 이미 있어도 venv·dev 패키지는 별도다 — NCP 이미지의 3.12 에는 python3.12-venv 가 없어
# 03단계의 `python3.12 -m venv` 가 "ensurepip is not available" 로 실패했다(실측). 없으면 설치한다.
if ! "python${PYTHON_VERSION}" -m venv --help >/dev/null 2>&1 || ! "python${PYTHON_VERSION}" -c "import ensurepip" >/dev/null 2>&1; then
    log "python${PYTHON_VERSION}-venv / -dev 설치"
    sudo apt-get install -y "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
else
    log "python${PYTHON_VERSION}-venv 이미 설치됨"
fi

# ── NVIDIA 드라이버 ─────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA 드라이버 정상: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
    warn "드라이버가 없습니다. ubuntu-drivers로 설치를 시도합니다 (NCP '드라이버 포함' 이미지면 이 분기로 오지 않습니다)."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    warn "설치 후 반드시 재부팅: sudo reboot  → 재부팅 뒤 00_check_gpu.sh로 확인하고 02단계로 진행"
fi

# ── 시간대 (로그 시각을 한국 기준으로) ──────────────────────────
if command -v timedatectl >/dev/null 2>&1; then
    sudo timedatectl set-timezone Asia/Seoul || true
fi

# ── 앱 루트 ──────────────────────────────────────────────────
mkdir -p "${APP_ROOT}"
log "앱 루트: ${APP_ROOT}"

cat <<EOF

1단계 완료.
  다음: bash ${DEPLOY_DIR}/03_deploy_code.sh

방화벽 메모: NCP는 ufw가 아니라 콘솔의 ACG로 포트를 관리합니다.
  외부에 열 포트는 22(SSH, 관리자 IP만)와 8000(orchestrator, 키오스크 IP만) 두 개뿐입니다.
  8001(rag_service) / 9000(vLLM 답변 LLM) / 10000·10001(VLM 인덱싱)은 127.0.0.1에만 바인딩하므로 열지 않습니다.
EOF
