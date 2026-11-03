#!/usr/bin/env bash
# 0단계 — 서버 상태 점검 (아무것도 설치하지 않는다)
#
#   NCP gp1l8-g3 (NVIDIA L4 24GB, 8 vCPU, 48GB RAM) 기준으로 드라이버·디스크·메모리·
#   인터넷 연결을 확인하고, 다음 단계에서 막힐 것을 미리 알려준다.
#
#   사용법: bash 00_check_gpu.sh
source "$(dirname "$0")/lib.sh"

log "OS / 커널"
grep -E "^(PRETTY_NAME|VERSION_ID)=" /etc/os-release || true
uname -r

log "NVIDIA 드라이버 · GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    case "$GPU_NAME" in
        *L4*)  log "L4 확인 — 예상대로 24GB 구성입니다." ;;
        *)     warn "예상한 L4가 아닙니다 (${GPU_NAME}). VRAM 예산(문서/gpu_server/README.md '메모리 예산')을 다시 계산하세요." ;;
    esac
else
    warn "nvidia-smi가 없거나 응답하지 않습니다."
    warn "NCP GPU 서버 이미지 중 'GPU 드라이버 포함' 이미지를 골랐는지 확인하세요."
    warn "드라이버가 없다면 01_setup_system.sh가 설치를 시도하며, 그 뒤 재부팅이 필요합니다."
fi

log "디스크 (모델·인덱스·로그를 담을 루트 볼륨, 40GB 이상 여유 권장)"
df -h / | tail -1
FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
[ "${FREE_GB:-0}" -ge 40 ] || warn "여유 디스크가 ${FREE_GB}GB입니다. A.X 원본(15GB) + VLM(9GB) + bge 모델(3GB) + 파이썬·vLLM 환경(15GB)을 담기엔 부족할 수 있습니다. NCP 콘솔에서 블록 스토리지를 추가하세요."

log "메모리 / CPU"
free -h | head -2
nproc

log "Python"
for v in 3.11 3.12 3; do
    if command -v "python${v}" >/dev/null 2>&1; then echo "python${v}: $(python${v} --version 2>&1)"; fi
done

log "인터넷 (모델·패키지 다운로드에 필요: HuggingFace, PyPI)"
for host in https://huggingface.co https://pypi.org; do
    if curl -fsS -m 5 -o /dev/null "$host"; then echo "OK   $host"; else warn "접속 실패 $host — ACG 아웃바운드 규칙을 확인하세요"; fi
done

log "이미 설치된 구성 요소"
[ -x "${VLLM_VENV}/bin/vllm" ] && echo "vllm-env: 있음 ($("${VLLM_VENV}/bin/vllm" --version 2>/dev/null | tail -1))" || echo "vllm-env: 없음 (04단계에서 생성)"
[ -d "${APP_VENV}" ] && echo "app-env: 있음 (${APP_VENV})" || echo "app-env: 없음 (03단계에서 생성)"
[ -f "${ENV_FILE}" ] && echo ".env: 있음" || echo ".env: 없음 (03단계에서 생성)"

log "점검 끝. 경고가 없으면 01_setup_system.sh로 진행하세요."
