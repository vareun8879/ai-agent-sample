#!/usr/bin/env bash
# 5단계 — 문서 인덱싱 (docs/ 의 PDF → Chroma 벡터DB)
#
#   두 가지 방법 중 하나를 고른다.
#
#   A) 개발 PC에서 만든 인덱스를 그대로 가져온다 (가장 빠름, vLLM 불필요)
#        개발 PC의 indexing/chroma_db 와 indexing/output* 을 서버의 같은 위치로 복사한 뒤
#        bash 05_index_documents.sh --reload-only
#
#   B) 서버에서 새로 인덱싱한다 (docs/ 가 바뀌었을 때)
#        bash 05_index_documents.sh              # 텍스트 레이어만 (스캔본 페이지가 없을 때)
#        bash 05_index_documents.sh --with-vlm   # 스캔본·깨진 레이어 페이지를 VLM(vLLM)으로 전사
#        bash 05_index_documents.sh --only 7010170 [--with-vlm]   # docs/7010170/ 만 다시 인덱싱 (기관 문서 갱신 때)
#        bash 05_index_documents.sh --only common,7010170         # 여러 폴더는 쉼표로. 다른 기관 색인은 그대로 둔다
#
#   --with-vlm 은 vLLM 가상환경(첫 실행 시 자동 설치, 10분 이상)과 비전 모델(기본 Qwen3-VL-8B-FP8, 약 9GB) 다운로드가 필요하고,
#   인덱싱 동안 VRAM을 나눠 쓰기 위해 rag_service 를 잠시 내리고 LLM을 VRAM에서 내렸다가 끝나면 복구한다.
#
#   docs/ 구조:  docs/common/*.pdf  → 모든 기관 공통
#                docs/<기관코드>/*.pdf → 해당 기관(키오스크 App.config 의 InstitutionId)에서만 검색
source "$(dirname "$0")/lib.sh"
load_env

MODE="index"
WITH_VLM=0
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --reload-only) MODE="reload"; shift ;;
        --with-vlm)    WITH_VLM=1; shift ;;
        --only)        ONLY="$2"; shift 2 ;;      # docs/ 밑 폴더 이름(common 또는 기관ID, 쉼표 구분) — 그 폴더만 다시 인덱싱
        --only=*)      ONLY="${1#--only=}"; shift ;;
        *) die "알 수 없는 옵션: $1" ;;
    esac
done

INDEX_DIR="${APP_ROOT}/indexing"
VLM_MODEL_ID="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"

reload_rag() {
    if systemctl is-active --quiet rag_service; then
        log "rag_service 에 인덱스 재로드 요청"
        curl -fsS -X POST "http://127.0.0.1:8001/reload" -H "X-Internal-Key: ${INTERNAL_API_KEY}" && echo
    else
        log "rag_service 가 떠 있지 않음 — 06단계에서 기동하면 새 인덱스를 읽습니다"
    fi
}

if [ "$MODE" = "reload" ]; then
    [ -d "${INDEX_DIR}/chroma_db" ] || die "${INDEX_DIR}/chroma_db 가 없습니다. 개발 PC에서 복사했는지 확인하세요."
    reload_rag
    exit 0
fi

# ── B) 서버에서 인덱싱 ─────────────────────────────────────────
ls "${APP_ROOT}/docs"/*/*.pdf >/dev/null 2>&1 || die "${APP_ROOT}/docs/<폴더>/*.pdf 가 없습니다."
# shellcheck disable=SC1091
source "${APP_VENV}/bin/activate"

cleanup() {
    if [ "$WITH_VLM" = "1" ]; then
        log "VLM 정리: vllm_vlm · vlm_service 중지"
        sudo systemctl stop vlm_service vllm_vlm 2>/dev/null || true
        if [ "${LLM_WAS_ACTIVE:-0}" = "1" ]; then
            log "vllm_llm 복구 (모델 로드에 수 분 — 그동안 rag_service /health 의 generator 는 unreachable)"
            sudo systemctl start vllm_llm
        fi
    fi
    if [ "${RAG_WAS_ACTIVE:-0}" = "1" ]; then
        # 재기동이라 새 인덱스와 새 코드가 함께 올라간다 (reload 불필요). 모델 로드에 30~60초
        log "rag_service 복구 (기동에 30~60초)"
        sudo systemctl start rag_service
    fi
}
trap cleanup EXIT

# VRAM 확보 1: rag_service 는 임베딩·reranker 모델을 GPU 에 올려두므로(약 4.7GB) 떠 있으면 내린다.
# 실측(09-17): vllm_llm 16.7GB + rag_service 4.7GB 상태에서 embed_chunks 가 CUDA OOM(남은 9MiB) 으로 실패했다.
# 임베딩 동안(2~3분) 키오스크 질문은 실패한다 — 재인덱싱은 한가한 시간에.
RAG_WAS_ACTIVE=0
if systemctl is-active --quiet rag_service; then
    log "rag_service 중지 (임베딩 VRAM 확보 — 끝나면 자동 복구)"
    RAG_WAS_ACTIVE=1; sudo systemctl stop rag_service
fi

if [ "$WITH_VLM" = "1" ]; then
    VLM_BACKEND="${VLM_BACKEND:-vllm}"
    if [ "${VLM_BACKEND}" = "vllm" ]; then ensure_vllm_venv; fi

    # VRAM 확보 2: 답변 LLM(vllm_llm)도 내린다 — VLM 이 24GB 를 다 쓴다.
    # 그동안 키오스크 질문은 실패하므로 --with-vlm 재인덱싱은 점검 시간에 돌린다.
    LLM_WAS_ACTIVE=0
    if systemctl is-active --quiet vllm_llm; then LLM_WAS_ACTIVE=1; sudo systemctl stop vllm_llm; fi

    if [ "${VLM_BACKEND}" = "transformers" ]; then
        # A.X-4.0-VL-Light: vLLM 이 못 받는 커스텀 아키텍처 → vlm_service 가 transformers 로 직접 로드한다
        VLM_DIR="${VLM_HF_DIR:-${APP_ROOT}/models/hf/$(basename "${VLM_MODEL_ID}")}"
        if [ ! -f "${VLM_DIR}/config.json" ]; then
            log "비전 모델 다운로드: ${VLM_MODEL_ID} → ${VLM_DIR} (bf16 약 16GB, 최초 1회)"
            # shellcheck disable=SC1091
            source "${APP_VENV}/bin/activate"
            mkdir -p "${VLM_DIR}"
            huggingface-cli download "${VLM_MODEL_ID}" --local-dir "${VLM_DIR}" --exclude "*.gguf" ${HF_TOKEN:+--token "$HF_TOKEN"}
            deactivate
        fi
        log "vlm_service 기동 (transformers · ${VLM_MODEL_ID} — 모델 로드에 1~3분)"
        sudo systemctl start vlm_service
        wait_http "http://127.0.0.1:10000/health" 1800 || die "vlm_service 가 30분 안에 뜨지 않았습니다: sudo journalctl -u vlm_service -n 100"
    else
        log "vLLM(비전 모델 ${VLM_MODEL_ID}) + vlm_service 기동"
        sudo systemctl start vllm_vlm
        wait_http "http://127.0.0.1:10001/v1/models" 1800 || die "vLLM 이 30분 안에 뜨지 않았습니다 (첫 실행은 모델 다운로드): sudo journalctl -u vllm_vlm -n 100"
        sudo systemctl start vlm_service
        wait_http "http://127.0.0.1:10000/health" 60 || die "vlm_service 가 응답하지 않습니다: sudo journalctl -u vlm_service -n 50"
    fi
    export VLM_REQUIRED=1
else
    warn "VLM 없이 인덱싱합니다 (VLM_REQUIRED=0). 스캔본 페이지는 빈 텍스트로 들어가고, 글자 사이가 벌어진 PDF는 검색 품질이 떨어집니다."
    warn "그런 페이지가 있으면 --with-vlm 으로 다시 돌리세요. (load_documents 로그의 'vlm_pages'/'broken layer' 경고로 알 수 있습니다)"
    warn "단, 전에 --with-vlm 으로 한 번 전사한 페이지는 indexing/vlm_cache/ 에서 재사용되므로(로그 'VLM 캐시 재사용') 조각 규칙만 바꿨을 땐 이대로 충분합니다."
    export VLM_REQUIRED=0
fi

# 임베딩은 L4에서 GPU로 (인덱싱 3단계). 서비스와 달리 배치라 VRAM 공유 문제 없음
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
export INTERNAL_API_KEY

if [ -n "${ONLY}" ]; then
    log "부분 인덱싱: docs/{${ONLY}} 만 (load → chunk → embed → store, 다른 폴더 색인은 유지)"
else
    log "인덱싱 파이프라인 실행 (load → chunk → embed → store)"
fi
cd "${INDEX_DIR}"
python run_all.py ${ONLY:+--only "${ONLY}"}
deactivate

# rag_service 를 내렸다면 cleanup(EXIT) 이 재기동한다 — 새 코드·새 인덱스 모두 반영. 내리지 않았다면(원래 꺼져 있던 경우) 06 이 띄운다

cat <<EOF

5단계 완료. 인덱스: ${INDEX_DIR}/chroma_db
  다음: bash ${DEPLOY_DIR}/06_start_and_check.sh
EOF
