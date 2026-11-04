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

## 3-1 단위 경계 수정

`quantities()`가 숫자 뒤 공백을 넘어 다음 낱말의 첫 글자를 단위로 읽던 문제를 수정한다.
`2024 실증`, `2025\n개교`, `1 회의실`은 각각 단위 없는 숫자로 남긴다.
붙여 쓴 단위는 기존처럼 읽고, 띄어 쓴 단위는 끝 경계 또는 일반 조사·서술어 꼬리를
확인한다. `83명이`, `83 명이`, `5 층에`, `50 백만원입니다`와 독립된 다음 줄 단위는 보존한다.
이 규칙은 형태소 분석이나 의미 판정이 아니며, 날짜·단위 생략·설명형 비교는 아직 변경하지 않는다.

이제 검증 의미가 바뀌므로 위 동등성 검사 대신 `--compare`를 사용한다.
이 모드는 실제 파서로 첫 출력을 읽고 추가·제거 오류, 재시도 생략 후보, 새 재시도 필요,
재시도 피드백 변경을 구별한다. 새 모델 출력·최종 지연은 예측하지 않는다.
보고서 전체의 추가 판정과 값 규칙을 함께 재채점해 변화도 기록한다.

클라우드에서 가장 최근의 시각 이름 골든 폴더를 선택한다. 출력된 기준 경로가
261104_1 측정 폴더인지 확인한다. 재채점 폴더는 선택하지 않는다.

```bash
cd /root/ai-agent-v2
bash deploy/gpu_server/check.sh && (
cd rag_service
BASELINE=$(find /root/ai-agent-v2/reports -mindepth 1 -maxdepth 1 -type d -name '20??????_??????' | sort | tail -n 1)
printf 'Baseline: %s\n' "$BASELINE"
/root/ai-agent/app-env/bin/python -m evaluation.audit_validation_codes \
  "$BASELINE/trace.jsonl" --compare \
  --reports "$BASELINE"/last_report_*_[123].json \
  --output "../reports/unit_boundary_audit_$(date '+%Y%m%d_%H%M%S').json"
)
```

먼저 report_changes=0인지 확인하고, 추가 오류를 모두 설명할 수 있어야 하며,
실제 숫자 오류도 계속 검출돼야 한다. 전체 차이는 출력 JSON 파일에 저장한다.
대조 결과를 검토한 뒤 rag_service를 재시작하고 기존 골든 3세트×3회를 측정한다.
