#!/usr/bin/env bash
# 3단계 — 코드 배포 · 운영 키(.env) · 파이썬 환경 · systemd 등록
#
#   전제: 저장소의 server/ 폴더 내용이 APP_ROOT(기본: 이 스크립트의 두 단계 위 = ~/ai-agent)에 올라와 있다.
#     SFTP 드래그 또는  rsync -avz "/c/dev/GPU 노트북 코드/server/" root@<서버IP>:~/ai-agent/
#   필요한 폴더: orchestrator/ rag_service/ indexing/ docs/ vlm_service/ deploy/gpu_server/  (+ deploy/gpu_server/.env)
#   키오스크 쪽 코드(stt_ondevice, ai agent wpf)는 server/ 밖에 있어 서버에 올라오지 않는다.
#
#   하는 일
#     1) .env 확인 — 저장소의 .env 를 그대로 쓴다 (없거나 키가 비어 있으면 중단)
#     2) app-env 가상환경 + 각 서비스 requirements 설치 (Linux torch는 기본이 CUDA 빌드)
#     3) bge-m3 · bge-reranker-v2-m3 를 HF 캐시(.env 의 HF_HOME)에 미리 받아 첫 기동 지연 제거
#     4) systemd 유닛 등록 (rag_service, orchestrator). 기동은 06단계에서
#
#   사용법: bash 03_deploy_code.sh
source "$(dirname "$0")/lib.sh"
need_cmd "python${PYTHON_VERSION}"

for d in orchestrator rag_service indexing docs; do
    [ -d "${APP_ROOT}/${d}" ] || die "${APP_ROOT}/${d} 가 없습니다. 저장소를 먼저 올리세요 (스크립트 머리말 참고)."
done

# ── 1. .env — 설정의 단일 원천. 저장소의 server/deploy/gpu_server/.env 를 그대로 올린다 ────
#   (키 2개·경로·모델 설정이 모두 들어 있다. 별도 템플릿 파일은 없다)
if [ -f "${ENV_FILE}" ]; then
    chmod 600 "${ENV_FILE}"
    if grep -q "__INTERNAL_API_KEY__\|__KIOSK_API_KEY__\|__APP_ROOT__" "${ENV_FILE}"; then
        die ".env 에 채워지지 않은 자리표시자(__…__)가 있습니다. 키는 'openssl rand -hex 32' 로 만들어 넣으세요."
    fi
    log ".env 확인: ${ENV_FILE}"
else
    die ".env 가 없습니다: ${ENV_FILE}
  저장소의 server/deploy/gpu_server/.env 를 같이 올리세요 (PC 탐색기에서 '숨긴 항목 표시' 를 켜야 보입니다).
  잃어버렸다면: 새 키 2개를 'openssl rand -hex 32' 로 만들어 .env 의 INTERNAL_API_KEY / KIOSK_API_KEY 에 넣고,
  키오스크 App.config 의 KioskApiKey 도 같은 값으로 바꿉니다."
fi
load_env
[ -n "${INTERNAL_API_KEY:-}" ] && [ -n "${KIOSK_API_KEY:-}" ] || die ".env 에 INTERNAL_API_KEY / KIOSK_API_KEY 가 비어 있습니다."
mkdir -p "${HF_HOME:-$APP_ROOT/models/hf-cache}" "${APP_ROOT}/models" \
         "${APP_ROOT}/orchestrator/logs" "${APP_ROOT}/rag_service/logs"

# ── 2. 가상환경 · 의존성 ───────────────────────────────────────
if [ ! -d "${APP_VENV}" ]; then
    log "가상환경 생성: ${APP_VENV}"
    "python${PYTHON_VERSION}" -m venv "${APP_VENV}"
fi
# shellcheck disable=SC1091
source "${APP_VENV}/bin/activate"
pip install --upgrade pip wheel >/dev/null
for svc in orchestrator rag_service indexing vlm_service; do
    req="${APP_ROOT}/${svc}/requirements.txt"
    if [ -f "$req" ]; then
        log "pip install: ${svc}"
        pip install -r "$req"
    fi
done
pip install "huggingface_hub[cli]" >/dev/null

log "torch CUDA 확인"
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")
EOF

# ── 3. 검색·재순위화 모델 미리 받기 (HF_HOME 은 .env 에서 왔다) ────
log "HF 모델 캐시 (bge-m3, bge-reranker-v2-m3) → ${HF_HOME}  — 약 3GB, 최초 1회"
python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-m3", device="cpu")
CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
print("HF 모델 캐시 완료")
EOF
deactivate

# ── 4. systemd 유닛 ─────────────────────────────────────────
log "systemd 유닛 등록 (사용자: ${RUN_USER})"
for unit in rag_service orchestrator vllm_llm vlm_service vllm_vlm; do
    tpl="${DEPLOY_DIR}/systemd/${unit}.service.tpl"
    [ -f "$tpl" ] || continue
    sed -e "s|__APP_ROOT__|${APP_ROOT}|g" \
        -e "s|__ENV_FILE__|${ENV_FILE}|g" \
        -e "s|__USER__|${RUN_USER}|g" \
        "$tpl" | sudo tee "/etc/systemd/system/${unit}.service" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable rag_service orchestrator >/dev/null

cat <<EOF

3단계 완료.
  - 설정·키:  ${ENV_FILE}   (키오스크 App.config 의 KioskApiKey = $(grep '^KIOSK_API_KEY=' "${ENV_FILE}" | cut -d= -f2))
  - 가상환경: ${APP_VENV}
  - 유닛:     rag_service, orchestrator, vllm_llm(답변 LLM), vllm_vlm·vlm_service(인덱싱용) — 등록만, 미기동
  다음: bash ${DEPLOY_DIR}/04_prepare_llm.sh
EOF
