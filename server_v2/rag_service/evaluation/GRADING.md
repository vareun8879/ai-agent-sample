# 평가 기준 버전 3

버전 3은 일반 거절 표현 '확인할 수 없습니다'를 인식한다. 숫자·URL·영문 정보가
함께 있는 응답은 기존대로 review로 분리한다.

서비스 생성·검색·검증 동작은 변경하지 않는다. 실시간 평가와 재채점은
answer_grading.evaluate를 함께 사용한다.

- answer_ok는 이전 키워드 판정을 보존한다.
- adjusted_answer_ok는 공통 추가 판정을 반영한다. 거절 문항은 답변 내용으로 판단한다.
  grounded=false 또는 출처 없음만으로 통과시키지 않는다.
- review는 수동 확인이 필요한 결과다. 확정 실패와 별도 집계하며 통과에는 포함하지 않는다.
  --fail-only는 review의 문항별 출력을 생략하고 요약에 개수를 남긴다.
- 서비스 호출 실패와 빈 거절 응답은 통과시키지 않는다.
- 신규 보고서는 전체 answer, sources, grounded, unverified를 저장한다.
- HTTP 평가의 retrieval_ok는 호환성을 위해 보존하는 최종 답변 출처 기준이다.
  citation_ok가 같은 판정이며 실제 검색 성공은 actual_retrieval_ok=null,
  retrieval_observed=false로 표시한다. 실제 검색 성능은 trace의 검색 단계와 대조해야 한다.
- 예전 보고서에서 누락된 메타데이터는 null로 보존한다. 300자 이상인 옛 답변은
  잘렸을 가능성이 있어 review로 둔다. 신규 보고서의 긴 답변은 이 길이 제한을 적용하지 않는다.
- golden_eval 종료 코드는 추가 기준 통과율을 사용한다. 이전 판정과 신규 판정의
  통과율은 각각 출력하며, 서로 다른 채점 버전의 점수를 바로 비교하지 않는다.

클라우드에서 필수 검사 후 기존 보고서만 재채점한다. 모델은 호출하지 않는다.

```bash
cd /root/ai-agent-v2
bash deploy/gpu_server/check.sh

cd rag_service
/root/ai-agent/app-env/bin/python -m evaluation.regrade \
  --output-dir "../reports/regrade_$(date '+%Y%m%d_%H%M%S')" \
  ../reports/20261103_153046/last_report_*_[123].json
```

원본은 수정하지 않는다. 같은 출력 파일이 이미 있으면 중단한다.
재시작·재인덱싱·새 골든 생성은 이 회차에 필요 없다.
