# Orchestrator

키오스크가 직접 호출하는 **얇은 질의응답 창구**. 무거운 모델은 하나도 들지 않습니다.

상용 구조에서 STT는 키오스크 안([stt_ondevice](../stt_ondevice/))에서 끝나므로,
이 창구는 전사된 질문 **텍스트**만 받습니다 — 질문 게이트를 거쳐
[rag_service](../rag_service/)에 위임하고, 세션·로그를 기록해 돌려줍니다.

## 처리 흐름

```
키오스크 ──X-API-Key──→ orchestrator ──X-Internal-Key──→ rag_service (검색+생성)
  (온디바이스 STT 완료)      │ 질문 게이트 (빈/짧은 질문 → 재발화 요청)
                             │ 멀티턴 세션 (직전 문답을 rag_service로 전달)
                             └ QA 로그(JSONL) 기록
```

파이프라인에 들어간 품질 장치:

- **질문 게이트** — 전사가 비었거나 너무 짧으면 검색/생성을 태우지 않고
  `needs_retry: true` + 재발화 안내 문구를 반환합니다.
- **멀티턴 세션** — `session_id`를 주면 최근 문답(기본 3턴, TTL 10분)을 기억해서
  "그건 어디서 해요?" 같은 후속 질문의 맥락을 잇습니다. (인메모리 — 프로세스 1개 기준)
- **기관별 문서 필터링** — `institution_id`를 주면 rag_service가 공통 문서 + 해당 기관 문서만
  검색 대상으로 삼습니다. 안 주면 공통 문서만 검색됩니다. 키오스크가 어느 기관에 설치됐는지는
  이 값으로 구분합니다.
- **답변 모드** — `answer_mode`를 주면 rag_service로 그대로 전달합니다. `voice`(기본): 음성 안내용
  — 첫 문장에 핵심, 3문장 이내, 목록은 3항목까지. `full`: 표·목록을 빠짐없이 나열(테스트 화면·개발용).
  키오스크는 답을 TTS로 읽어 주므로 기본을 `voice`로 둡니다 (실측: 같은 질문이 full 45.6초 → voice 수 초).
- **QA 로그** — 모든 문답을 `logs/qa_log.jsonl`에 기록합니다 (request_id, 질문, 답변, 근거,
  소요 시간). 오답 분석과 골든셋 확장의 원천 데이터입니다.

## 설치 및 실행

먼저 [rag_service](../rag_service/)(8001)가 떠 있어야 합니다 (인덱싱 `chroma_db` + Ollama 준비 필요).

```bash
pip install -r requirements.txt
set INTERNAL_API_KEY=<내부 서비스 공유 키>
set KIOSK_API_KEY=<키오스크용 키>
uvicorn main:app --host 0.0.0.0 --port 8000
```

**키 정책**: 두 키 중 하나라도 미설정이면 기동을 거부합니다. 로컬 개발에서만
`ALLOW_DEV_KEYS=1`로 개발용 기본 키를 쓸 수 있습니다 (운영 배포 금지).

파워셸 : $env:ALLOW_DEV_KEYS="1"
CMD : set ALLOW_DEV_KEYS=1

전체 스택 일괄 기동은 [../scripts/start_all.ps1](../scripts/start_all.ps1) 참고.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KIOSK_API_KEY` | (필수) | 키오스크 → 오케스트레이터 인증 키 (`X-API-Key` 헤더) |
| `INTERNAL_API_KEY` | (필수) | 내부 서비스 호출 키. rag_service와 반드시 동일한 값 |
| `ALLOW_DEV_KEYS` | - | `1`이면 키 미설정 시 개발용 기본 키 사용 (로컬 개발 전용) |
| `RAG_SERVICE_URL` | `http://localhost:8001` | 질의응답(RAG) 서비스 주소 |
| `RAG_TIMEOUT` | `330` | RAG 호출 타임아웃(초). LLM 생성까지 포함하므로 넉넉히 |
| `MIN_QUESTION_CHARS` | `2` | 이보다 짧은 질문은 재발화 요청 |
| `STT_RETRY_MESSAGE` | (안내 문구) | 재발화 요청 시 키오스크에 보낼 문구 |
| `SESSION_TTL` | `600` | 세션 유지 시간(초) |
| `SESSION_MAX_TURNS` | `3` | 세션당 기억할 최근 문답 수 |
| `QA_LOG_PATH` | `logs/qa_log.jsonl` | 질문·답변 기록 파일 경로 |

## API

`/ask/*` 엔드포인트는 `X-API-Key` 헤더(= `KIOSK_API_KEY`)가 필요합니다.

- `GET /health` — 오케스트레이터 + RAG 서비스 상태 (인증 불필요)
- `POST /ask/text` — 질문 텍스트 → 근거 기반 답변
- `POST /ask/text/stream` — SSE 스트리밍 버전 (토큰 단위로 답변 전송)

### 예시

```bash
curl -X POST "http://localhost:8000/ask/text" \
  -H "X-API-Key: <키오스크 키>" -H "Content-Type: application/json" \
  -d '{"question": "스쿨패스는 어떻게 신청하나요?", "session_id": "kiosk-01", "institution_id": "7010170"}'
```

`institution_id`는 선택값입니다 — 생략하면 공통 문서만 검색됩니다.
`answer_mode`도 선택값입니다 — 생략하면 `voice`(짧은 음성 안내). 전체 목록이 필요하면 `"answer_mode": "full"`.

정상 응답:

```json
{
  "request_id": "a1b2c3d4e5f6",
  "question": "스쿨패스는 어떻게 신청하나요?",
  "answer": "스쿨패스는 학교 홈페이지의 사전신청 서비스에 접속해 신청합니다. ...",
  "sources": [
    { "source_file": "스쿨패스 사전신청 안내문.pdf", "page_number": 1, "chunk_index": 0, "distance": 0.104 }
  ],
  "grounded": true,
  "needs_retry": false
}
```

게이트에 걸린 경우 (재발화 요청):

```json
{
  "request_id": "a1b2c3d4e5f6",
  "question": "아",
  "answer": "",
  "sources": [],
  "needs_retry": true,
  "retry_reason": "too_short",
  "message": "질문을 잘 알아듣지 못했습니다. 다시 한번 말씀해 주세요."
}
```

- `grounded: false`면 근거를 못 찾아 고정 안내문을 반환한 것 (무관 질문 차단)
- 키오스크는 `needs_retry`가 true면 `message`를 표시하고 재녹음을 유도하면 됩니다

`/ask/text/stream`은 SSE로 `data: {"type":"delta","text":"..."}` 이벤트가 이어지다가
`data: {"type":"done","answer":...,"sources":[...]}` 로 끝납니다.
