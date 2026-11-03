#!/usr/bin/env bash
# 4단계 — 답변 생성 LLM 준비 (vLLM)
#
#   HF 원본(skt/A.X-4.0-Light, safetensors)을 받아 vllm_llm 유닛(:9000)으로 띄운다. 로드 시 FP8 로
#   동적 양자화(L4 = Ada, FP8 지원) — GGUF 변환이 없다. 납품 키오스크 40대+ 의 동시 요청을
#   continuous batching 으로 처리한다. rag_service 는 LLM_MODEL(--served-model-name)로 호출한다.
#
#   NCP 서버에는 Ollama 를 설치하지 않는다. 개발 PC(8GB)에서만 rag_service 의 LLM_BACKEND=ollama 를 쓴다.
#
#   양자화 (VLLM_LLM_QUANT, .env):
#     fp8  ≈ 7.5GB — 기본. 정확도 손실 거의 없음
#     none ≈ 15GB  — bf16 원본. FP8 이 문제일 때의 폴백 (VLLM_LLM_GPU_UTIL 을 0.85 로)
#
#   사용법: bash 04_prepare_llm.sh
source "$(dirname "$0")/lib.sh"
load_env

LLM_TAG="${LLM_MODEL:-ax4-light}"
LLM_HF_REPO="${LLM_HF_REPO:-skt/A.X-4.0-Light}"
LLM_HF_DIR="${LLM_HF_DIR:-${APP_ROOT}/models/hf/$(basename "${LLM_HF_REPO}")}"
VLLM_PORT=9000

log "HuggingFace 원본 다운로드: ${LLM_HF_REPO} → ${LLM_HF_DIR} (이미 있으면 건너뜀)"
# shellcheck disable=SC1091
source "${APP_VENV}/bin/activate"
mkdir -p "${LLM_HF_DIR}"
huggingface-cli download "${LLM_HF_REPO}" --local-dir "${LLM_HF_DIR}" \
    --exclude "*.gguf" "original/*" "*.bin" ${HF_TOKEN:+--token "$HF_TOKEN"}
deactivate
[ -f "${LLM_HF_DIR}/config.json" ] || die "${LLM_HF_DIR}/config.json 이 없습니다 — 다운로드 실패"

ensure_vllm_venv

# 인덱싱용 VLM 이 떠 있으면 내린다 — VRAM 은 답변 LLM 이 쓴다
sudo systemctl stop vlm_service vllm_vlm 2>/dev/null || true

log "vllm_llm 기동 — ${LLM_HF_REPO} · ${VLLM_LLM_QUANT:-fp8} · :${VLLM_PORT}  (첫 기동은 FP8 변환·CUDA 그래프 캡처로 5~10분)"
sudo systemctl enable vllm_llm >/dev/null
sudo systemctl restart vllm_llm
wait_http "http://127.0.0.1:${VLLM_PORT}/v1/models" 1800 || die "vllm_llm 이 30분 안에 뜨지 않았습니다: sudo journalctl -u vllm_llm -n 200"

log "모델 정보"
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" | python3 -c 'import sys,json; [print(" -", m["id"], "| max_model_len:", m.get("max_model_len")) for m in json.load(sys.stdin)["data"]]'

log "워밍업 — 200토큰 생성으로 실제 속도 측정"
START=$(date +%s.%N)
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"${LLM_TAG}\",\"messages\":[{\"role\":\"user\",\"content\":\"학교 안내 키오스크가 할 수 있는 일을 다섯 문장으로 설명해 주세요.\"}],\"max_tokens\":200,\"temperature\":0}" \
    > /tmp/vllm_warmup.json
END=$(date +%s.%N)
python3 - "$START" "$END" <<'PY'
import json, sys
d = json.load(open("/tmp/vllm_warmup.json"))
t = float(sys.argv[2]) - float(sys.argv[1])
tok = d["usage"]["completion_tokens"]
print("응답:", d["choices"][0]["message"]["content"].strip()[:200])
print(f"생성 {tok}토큰 / {t:.1f}초 → 약 {tok / t:.1f} tok/s (프롬프트 처리 포함 벽시계 기준)")
PY

log "VRAM 사용량 (vLLM 상주 후)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

cat <<EOF

4단계 완료. rag_service 는 .env 의 VLLM_LLM_URL=http://127.0.0.1:${VLLM_PORT}/v1 로 호출합니다.
  상태: sudo systemctl status vllm_llm      로그: sudo journalctl -u vllm_llm -f
  다음: bash ${DEPLOY_DIR}/05_index_documents.sh   (문서 인덱싱)
EOF
