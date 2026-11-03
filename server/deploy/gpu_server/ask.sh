#!/usr/bin/env bash
# 질문 하나를 rag_service 에 직접 던져 답변·근거 페이지·검증 로그를 본다 (골든 없이 손으로 확인할 때).
#   bash ask.sh "행정실은 몇 층이야?"              # 공통 문서(망미중)만
#   bash ask.sh "실증장비는 몇 종이야?" Z039353     # 기관 문서 포함 (7010170 배재고, Z039353 AICA)
#   bash ask.sh "…" "" full                        # 세 번째 인자 full = 긴 답변 모드 (기본 voice)
set -euo pipefail
Q="${1:?질문을 따옴표로 감싸서 주세요}"
INST="${2:-}"
MODE="${3:-voice}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY="$(grep '^INTERNAL_API_KEY=' "$ROOT/deploy/gpu_server/.env" | cut -d= -f2-)"
[ -n "$KEY" ] || { echo "INTERNAL_API_KEY 를 .env 에서 못 읽음"; exit 1; }

BODY="$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "institution_id": sys.argv[2] or None, "answer_mode": sys.argv[3]}, ensure_ascii=False))' "$Q" "$INST" "$MODE")"
START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "질문: $Q   (기관: ${INST:-없음}, 모드: $MODE)"
curl -s -X POST http://127.0.0.1:8001/answer -H "Content-Type: application/json" -H "X-Internal-Key: $KEY" -d "$BODY" \
  | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("답변:", d.get("answer"))
print("근거:", [(s.get("source_file", "")[:18], "p%s" % s.get("page_number")) for s in d.get("sources", [])])
print("grounded:", d.get("grounded"), "| 검증에 걸린 값:", d.get("unverified") or "없음")
'
echo "── 아래는 내부 과정 로그 (검색 → 근거 → 인용 → 검증). 키오스크로 나간 답은 위의 '답변:' 한 줄뿐 ──"
sudo journalctl -u rag_service --since "$START" --no-pager \
  | grep -E "낱말 일치 후보|rerank:|컨텍스트:|핵심 근거 보강|인용:|재생성|재질문|거절했지만|검증|문서 구절로 답변|답변 생성 완료|근거 없음" \
  | sed -E 's/^.*INFO |^.*WARNING /  /' | cut -c1-400
