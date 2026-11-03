#!/usr/bin/env bash
# 6단계 — 서비스 기동 · 종단 확인
#
#   답변 LLM(vllm_llm :9000) → rag_service(127.0.0.1:8001) → orchestrator(0.0.0.0:8000) 순으로 띄우고,
#   health 3종 + 실제 질문 1건을 키오스크와 똑같은 경로(X-API-Key)로 보내 본다.
#
#   사용법: bash 06_start_and_check.sh ["테스트 질문"]
source "$(dirname "$0")/lib.sh"
load_env
[ -n "${KIOSK_API_KEY:-}" ] || die ".env 에 KIOSK_API_KEY 가 없습니다 (03단계)."

QUESTION="${1:-교무 업무 조직을 설명해줘}"

log "서비스 기동"
sudo systemctl start vllm_llm
wait_http "http://127.0.0.1:9000/v1/models" 1800 || die "vllm_llm 이 뜨지 않았습니다: sudo journalctl -u vllm_llm -n 200"
sudo systemctl restart rag_service   # 재배포(코드·.env 변경) 반영을 위해 start 가 아니라 restart
wait_http "http://127.0.0.1:8001/health" 180 || die "rag_service 가 3분 안에 뜨지 않았습니다: sudo journalctl -u rag_service -n 100"
sudo systemctl restart orchestrator
wait_http "http://127.0.0.1:8000/health" 60 || die "orchestrator 가 뜨지 않았습니다: sudo journalctl -u orchestrator -n 100"

log "health"
echo "vllm_llm     : $(curl -s http://127.0.0.1:9000/v1/models | python3 -c 'import sys,json; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
echo "rag_service  : $(curl -s http://127.0.0.1:8001/health)"
echo "orchestrator : $(curl -s http://127.0.0.1:8000/health)"

log "VRAM"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

log "질문 테스트 (키오스크 경로): ${QUESTION}"
START=$(date +%s.%N)
curl -s -X POST "http://127.0.0.1:8000/ask/text" \
    -H "X-API-Key: ${KIOSK_API_KEY}" -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'question': sys.argv[1]}, ensure_ascii=False))" "${QUESTION}")" \
    | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("grounded  :", d.get("grounded"), "| needs_retry:", d.get("needs_retry"))
print("answer    :", (d.get("answer") or d.get("message") or "")[:400])
print("sources   :", [(s.get("source_file"), s.get("page_number")) for s in d.get("sources", [])][:5])
print("request_id:", d.get("request_id"))'
END=$(date +%s.%N)
echo "소요 시간: $(python3 -c "print(round(${END}-${START},1))")초"

log "골든 질문 세트 채점 (rag_service/golden/*.jsonl — 파일마다 따로: 검색 근거 페이지 + 답변 키워드)"
# 세트마다 따로 채점한다 — 기본 30문항(mangmi_2026)과 확장 100문항(mangmi_2026_ext)의 통과율을 각각 본다.
# 보고서는 golden/last_report_<파일명>.json. 반복 실행·특정 세트만 돌리기는 golden_check.sh.
if ls "${APP_ROOT}"/rag_service/golden/*.jsonl >/dev/null 2>&1; then
    GOLDEN_FAIL=0
    for gf in "${APP_ROOT}"/rag_service/golden/*.jsonl; do
        name="$(basename "${gf}" .jsonl)"
        echo -e "\n── 골든 세트: ${name} ──"
        (cd "${APP_ROOT}/rag_service" && "${APP_VENV}/bin/python" golden_eval.py "golden/${name}.jsonl" --answer \
            --url http://127.0.0.1:8001 --key "${INTERNAL_API_KEY}" --report "golden/last_report_${name}.json") \
            || { GOLDEN_FAIL=1; warn "${name}: 통과율이 기준(90%) 미만 — rag_service/golden/last_report_${name}.json 의 why 항목을 확인하세요."; }
    done
    [ "${GOLDEN_FAIL}" = "0" ] && echo -e "\n모든 골든 세트가 기준(90%)을 넘었습니다."
else
    echo "골든 세트 없음 — 건너뜀"
fi

cat <<EOF

6단계 완료. 상시 기동 상태:
  sudo systemctl status rag_service orchestrator vllm_llm
로그:
  sudo journalctl -u rag_service -f      # 검색·생성 로그
  sudo journalctl -u orchestrator -f     # 키오스크 요청
  ${APP_ROOT}/orchestrator/logs/qa_log.jsonl   # 문답 기록 (품질 분석)

키오스크 App.config:
  VoiceQa.OrchestratorUrl = http://<이 서버 공인 IP>:8000   (HTTPS는 07_https_caddy.sh)
  VoiceQa.KioskApiKey     = ${KIOSK_API_KEY}
EOF
