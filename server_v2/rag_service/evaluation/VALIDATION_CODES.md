# 검증 오류 코드

이 회차는 기존 검사 조건·오류 문자열·재시도·최종 응답 정책을 유지한다.
CheckedItem.issues와 trace의 attempts[].issues / items[].issues에 오류 코드를 추가한다.
외부 API의 answer / sources / grounded / unverified는 기존 형식이다.

| 코드 | 의미 |
|---|---|
| evidence_not_allowed | 항목에 제공하지 않은 근거 ID |
| quote_not_in_source | 인용이 원문에 없음 |
| value_not_in_quote | 선택 값이 인용에 없음 |
| value_quantity_missing_in_answer | 선택 값·단위가 답변에서 확인되지 않음 |
| value_text_missing_in_answer | 선택한 이름·설명 구절이 답변에서 확인되지 않음 |
| answer_quantity_not_in_quote | 답변의 값·단위가 유효한 인용에서 확인되지 않음 |
| parse_failure | 모델 출력 파싱 실패 |
| generation_output_error | 출력 길이 초과 등 생성 결과 처리 오류 |

각 claim의 오류에는 evidence_id와 0부터 시작하는 claim_index를 기록한다.
같은 오류가 여러 claim에 있으면 구조화 기록에는 각 위치가 남으며, 기존 오류
문자열 목록은 이전처럼 중복을 제거한다. 코드 자체가 의미적 모순을 확정하지는 않는다.
단위 생략과 잘못된 값의 구분 등 검사 의미 변경은 다음 회차에서 다룬다.

클라우드에서 먼저 모델 없는 검사와 과거 trace 대조를 실행한다.

```bash
cd /root/ai-agent-v2
bash deploy/gpu_server/check.sh && (
cd rag_service
/root/ai-agent/app-env/bin/python -m evaluation.audit_validation_codes \
  ../reports/20261103_153046/trace.jsonl \
  --output "../reports/validation_codes_$(date '+%Y%m%d_%H%M%S').json"
)
```

checked가 0보다 크고 mismatch_count=0이어야 한다. 오류 문자열의 순서는
집합 순회로 프로세스마다 달라질 수 있어 내용으로 비교한다. 입력 누락 행은
skipped로 따로 집계한다. 이 도구는 모델 호출과 원본 파일 변경을 하지 않는다.

대조가 통과하면 rag_service를 재시작한 뒤 기존 골든 3세트×3회로 확인한다.
새 환경변수나 재인덱싱은 필요 없다.
