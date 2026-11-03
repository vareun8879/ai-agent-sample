#!/usr/bin/env bash
# 골든 질문 세트만 채점한다 — 세트마다 따로, 원하면 여러 번 반복해서 편차까지.
#
#   bash golden_check.sh                         # golden/*.jsonl 전부, 세트마다 1회
#   bash golden_check.sh -n 3                    # 세트마다 3회 반복 (7B 편차 확인 — '3회 다 실패' 만 코드로 잡는다)
#   bash golden_check.sh mangmi_2026_ext         # 확장 100문항만
#   bash golden_check.sh -n 3 mangmi_2026 mangmi_2026_ext
#   bash golden_check.sh --lint                  # 채점 대신 골든셋 자체 검사 (키워드가 기대 페이지에 있는지, LLM 불필요)
#   bash golden_check.sh --full                  # FAIL 줄만이 아니라 문항별 전체 출력
#
# 06_start_and_check.sh 는 서비스 기동 + health + 테스트 질문 + 골든(세트별 1회)을 다 하고,
# 이 스크립트는 서비스가 떠 있는 상태에서 골든만 빠르게 돌린다 (세트당 30문항 ≈ 1분, 100문항 ≈ 3분).
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/lib.sh"
load_env

RUNS=1
FULL=0
LINT=0
SETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -n) RUNS="$2"; shift 2 ;;
        --full) FULL=1; shift ;;
        --lint) LINT=1; shift ;;
        -h|--help) sed -n 2,12p "$0"; exit 0 ;;
        *) SETS+=("$1"); shift ;;
    esac
done

GOLDEN_DIR="${APP_ROOT}/rag_service/golden"
if [ ${#SETS[@]} -eq 0 ]; then
    for f in "${GOLDEN_DIR}"/*.jsonl; do SETS+=("$(basename "$f" .jsonl)"); done
fi
[ ${#SETS[@]} -gt 0 ] || die "${GOLDEN_DIR} 에 골든 세트(*.jsonl)가 없습니다"

cd "${APP_ROOT}/rag_service"

if [ "${LINT}" = "1" ]; then
    # 골든셋 자체 검사 — 인덱싱 1단계 출력(indexing/output/**/*.json)의 페이지 텍스트로 대조
    for name in "${SETS[@]}"; do
        echo -e "\n── 린트: ${name} ──"
        "${APP_VENV}/bin/python" golden_lint.py "golden/${name}.jsonl" || true
    done
    exit 0
fi

# rag_service 는 재시작 후 임베딩·reranker 적재에 30~60초 걸린다 — 05 나 restart 직후 바로 돌려도 되게 기다린다
wait_http "http://127.0.0.1:8001/health" 120 || die "rag_service(:8001) 가 2분째 응답하지 않습니다 — 먼저 bash 06_start_and_check.sh"

# /health 는 rag_service 만 보므로, 부팅 직후엔 vLLM 이 아직 모델을 올리는 중이라 /answer 가 502 를 낸다
# (실측 09-18: 복원 직후 첫 회차 41문항이 전부 502). 06 과 같이 vLLM 의 모델 목록이 열릴 때까지 기다린다 (최대 10분).
# (/answer 에 인사말을 보내는 방식은 검색 0건이면 LLM 을 안 거치고 200 을 돌려줘 미준비를 못 잡았다)
wait_http "http://127.0.0.1:9000/v1/models" 600 || die "vllm_llm(:9000) 이 10분째 준비되지 않음 — journalctl -u vllm_llm -n 50"

# 세트 × 회차. 회차마다 보고서를 golden/last_report_<세트>_<회차>.json 에 따로 남긴다 (다음 실행이 덮어쓰기 전까지).
declare -A SUMMARY
for name in "${SETS[@]}"; do
    [ -f "golden/${name}.jsonl" ] || die "golden/${name}.jsonl 이 없습니다"
    for i in $(seq 1 "${RUNS}"); do
        echo -e "\n=== ${name}  RUN ${i}/${RUNS} ==="
        out="$("${APP_VENV}/bin/python" golden_eval.py "golden/${name}.jsonl" --answer \
              --url http://127.0.0.1:8001 --key "${INTERNAL_API_KEY}" --report "golden/last_report_${name}_${i}.json" 2>&1 || true)"
        if [ "${FULL}" = "1" ]; then
            echo "${out}"
        else
            # FAIL 문항(+답변 줄)과 총점 줄만
            echo "${out}" | grep -E "FAIL|검색 통과" -A1 | grep -vE "^--$" || true
        fi
        SUMMARY["${name}#${i}"]="$(echo "${out}" | grep -E "검색 통과" | tail -1)"
    done
done

echo -e "\n── 요약 ──"
for name in "${SETS[@]}"; do
    for i in $(seq 1 "${RUNS}"); do
        printf "%-22s RUN %d  %s\n" "${name}" "${i}" "${SUMMARY["${name}#${i}"]:-(결과 없음)}"
    done
done
echo "보고서: ${GOLDEN_DIR}/last_report_<세트>_<회차>.json (why 항목에 실패 원인)"
