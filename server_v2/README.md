# server_v2 — 기존 인덱스를 사용하는 첫 실행본

기술 스택과 외부 API를 유지하면서 RAG 처리 경로를 다시 작성했다.
현재 단계는 **새 RAG + 기존 인덱스 스냅샷** 비교이다. 운영 대체 승인이나 정확도 개선이 확인된 상태가 아니다.
기존 server 및 TextQaClient는 수정하지 않는다. 모델 파일을 복제하거나 다운로드하지 않는다.

## 이번 구현 범위

- Python/FastAPI, Chroma, bge-m3, bge-reranker-v2-m3, A.X/vLLM 유지.
- 질문 항목 → 허용 근거 ID → 모델 JSON → 전체 파싱 → 항목별 검사 → 최종 문장.
- 출력 항목 누락/중복, 알 수 없는 근거 ID, 원문에 없는 인용, 값·단위 불일치 검사.
- 모든 생성/형식 복구/값 보정을 합쳐 LLM 최대 2회. 별도 한 줄 재답 호출 없음.
- 의미적 관련성 전체를 코드로 증명하지는 못한다. ID·인용·수치 통과는 완전한 사실성 보증이 아니다.
- 기관 필터는 검색에서 적용하고, 세션은 기관별로 분리한다.
- 스트리밍은 전체 처리 후 최종 문장을 delta, done 순으로 전달한다. 내부 JSON은 공개하지 않는다.
- 같은 systemd unit과 포트를 사용해 구현을 전환한다. 두 RAG 모델을 동시에 적재하지 않는다.

## 호출 계약

TextQaClient의 OrchestratorUrl을 변경하지 않는다.

| 항목 | 계약 |
|---|---|
| 외부 경로 | POST /ask/text, GET /health, POST /ask/text/stream |
| 외부 포트 | 기존 8000 |
| 인증 | X-API-Key |
| 요청 | question, session_id, institution_id, model(기존 a.x), answer_mode |
| 응답 | request_id, question, answer, sources, grounded, needs_retry 및 기존 재입력 필드 |
| 출처 | source_file, page_number, distance |
| 내부 경로 | /health, /reload, /answer, /answer/stream |
| 내부 포트/인증 | 8001, X-Internal-Key |

`grounded`는 항목 전부가 근거 검사를 통과했을 때만 true다. 일부 항목만 확인되면 false이며,
내부 trace의 items에 상태를 보존한다. insufficient와 conflict는 코드가 안내문을 조립한다.
`model` 필드는 기존 클라이언트 호환용이며 이 실행본에서 추가 모델을 선택하지 않는다.

## 이관한 부분과 새로 쓴 부분

새 구현: rag/contracts, query, retrieval, evidence, generation, parsing, validation,
rendering, pipeline, config, rag_service/main, 전환/점검 스크립트, 평가 보강과 테스트.

호환성 때문에 이관: orchestrator HTTP/인증/인사/세션 처리, VLM HTTP 어댑터,
기존 한국어 textutil, requirements, log_config, 기존 골든셋·채점기·린트.
orchestrator의 준비 상태와 기관별 세션 분리는 보완했다.
기존 verify.py, engine.py의 재생성/폴백 체인은 복사하지 않았다.

## 첫 클라우드 실행

아래 경로를 기준으로 한다. 로컬의 server_v2 내용만 `/root/ai-agent-v2/`로 업로드한다.
기존 `/root/ai-agent/`를 덮어쓰지 않는다. .env는 첫 전환 때 기존 인증·모델 설정에서 생성한다.

1. 현재 골든 보고서와 기존 코드/설정을 별도 보관한다.
2. 서버에서 모델 없는 구문/단위/HTTP 계약 검사를 실행한다.

```bash
cd /root/ai-agent-v2
bash deploy/gpu_server/check.sh /root/ai-agent/app-env/bin/python
```

3. 기존 보고서에 새 채점 기준을 적용해 기존 키워드 통과와 차이를 확인한다.
   일부 문항만 새 규칙이 정의되어 있다. legacy_only를 새 채점 통과로 합산하지 않는다.

```bash
cd /root/ai-agent-v2/rag_service
/root/ai-agent/app-env/bin/python -m evaluation.regrade /root/ai-agent/rag_service/golden/last_report_*_[123].json
```

4. 점검 통과 후 전환한다. `source`로 실행하지 않는다.

```bash
sudo bash /root/ai-agent-v2/deploy/gpu_server/switch.sh v2
```

전환은 기존 앱 중지 → 종료/포트/GPU PID 확인 → 최초 인덱스 스냅샷 복사 →
v2 환경 생성 → 같은 unit의 실행 경로 변경 → 기동/준비 상태 확인 순이다.
오류가 나면 대상 서비스를 중지한 뒤 이전 구현으로 복구한다.
복구도 실패하면 두 앱 서비스를 중지한다. 기준 서버의 코드·인덱스는 변경하지 않는다.
활성 구현은 `/var/lib/ai-agent/active-implementation`에 기록된다.
부팅과 자동 재시작도 현재 선택된 unit 경로를 따른다.

공용 `vllm_llm`은 한 개만 유지한다. 전환 중에도 이 단일 모델 서버는 공유하므로 그 GPU 메모리는 유지될 수 있다.
기존 rag_service의 프로세스/GPU 사용이 사라졌는지 확인한다. VLM/인덱싱 서비스가 실행 중이면 전환을 거부한다.
가상환경은 기존 app-env를 사용하며 의존성 버전은 기존 requirements를 유지한다.

5. 첫 측정은 `.env`의 `V2_VALIDATION_MODE=log`로 한다.

```bash
cd /root/ai-agent-v2/deploy/gpu_server
bash golden_check.sh -n 3 mangmi_2026 aica_2026 paichai_2026
```

보고서는 `/root/ai-agent-v2/reports/<시각>/`에 저장하므로 이전 측정을 덮어쓰지 않는다.
JSON 원문/항목 상태/근거 ID/호출 횟수는 rag_service/logs/trace.jsonl와 journal에 남는다.

```bash
cd /root/ai-agent-v2/rag_service
/root/ai-agent/app-env/bin/python -m evaluation.audit_trace logs/trace.jsonl /root/ai-agent/rag_service/golden/last_report_*_[123].json
```

log 모드는 새 값 검사로 답 문장을 제거하지 않는다. 오류는 unverified에 남고 grounded=false다.
파싱 불가나 근거 부족은 표시할 유효 답이 없으므로 안내한다.
기존 통과 문항에 대한 새 검출을 원문과 대조한 뒤에만 enforce를 측정한다.
enforce는 오류가 남은 항목을 안내문으로 대체한다. 이 변경의 통과율·거절 증가도 측정해야 한다.
배포 승인 전에는 log 결과를 운영 품질 보증으로 해석하지 않는다.

기존 구현으로 돌아가기:

```bash
sudo bash /root/ai-agent-v2/deploy/gpu_server/switch.sh v1
```

## 아직 검증하지 않은 것

로컬 모델·테스트를 실행하지 않았다. 클라우드에서 구문·테스트·systemd 전환/복구·GPU 종료·실제 JSON 생성과
3세트×3회·동시 요청 지연을 확인해야 한다. 자동 복구도 실제 서비스 환경 검증 전이다.

새 문서(다른 학교 1 + 비학교 1)의 정답·근거 검토와 기존 서버 기준선 측정은 아직 필요하다.
현재 세 문서로 만든 추가 표현을 미사용 문서 검증으로 취급하지 않는다.

기존 인덱스의 잘못된 추출과 손실된 표 관계는 이 단계에서 복원되지 않는다.
다음 별도 회차에서 원문/셀 관계/추출 버전을 보존하는 인덱싱을 구현하고 비교한다.
그 전에 이번 실행본의 오류를 분류하고 기존 통과 문항 회귀를 확인한다.
골든 기준이 달라졌으므로 예전 113~114/120을 새 채점의 목표치로 직접 사용하지 않는다.
동일 채점 기준에서 기존 통과 문항 회귀 0, 미사용 문서 개선, 동일 동시 요청 조건의 p95 지연 비교가 전환 판단 기준이다.

## ID 및 저장 구조

스냅샷 revision은 기존 chunks.jsonl의 SHA256을 포함한다. 근거 ID는 revision·문서 경로·페이지·기존 조각 ID·원문의
해시이며 버전 안에서 추적 가능하다. 모델의 E1/E2는 한 요청 안에서만 유효하다.
ID를 버전 사이에서 영구 불변이라고 가정하지 않는다. 새 인덱싱 시 문서 버전/위치/표 관계 JSON을 추가한다.
