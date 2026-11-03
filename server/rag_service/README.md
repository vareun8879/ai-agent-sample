# RAG Service

**2단계 질의응답 파이프라인**(검색 → 생성)을 하나로 묶은 내부 서비스.
[orchestrator](../orchestrator/)가 질문 텍스트를 넘기면 근거 기반 답변을 돌려준다.

무거운 모델(임베딩 모델)을 기동 시 적재하고, `X-Internal-Key`로
직접 접근을 막는다 — 오케스트레이터를 통해서만 호출된다.

| 단계 | 모듈 | 하는 일 |
|---|---|---|
| API 계층 | [main.py](main.py) | HTTP 변환만 — 인증, 스키마, 엔진 결과↔응답 변환 (로직 없음) |
| **RAG 엔진** | [rag/engine.py](rag/engine.py) | 파이프라인 파사드 — 검색→필터→병합→생성 오케스트레이션을 소유 |
| **1. 검색** | [rag/retriever.py](rag/retriever.py) | 질문 임베딩(bge-m3) → Chroma 벡터 검색 → 근거 조각 top-k + 거리 임계값 필터 |
| **1.2 재순위화** | [rag/reranker.py](rag/reranker.py) | cross-encoder(bge-reranker-v2-m3, GPU) 재채점 — 임베딩이 놓친 정답 조각 복원 + 인접 페이지 구제. 입력에서 문서 제목 접두 `[…]` 는 뗀다(짧은 지문 확률이 4~10배 떨어짐, 09-17 실측) |
| **1.5 확장** | [rag/expansion.py](rag/expansion.py) | small-to-big 페이지 병합 — 조각이 속한 페이지 전체(연속 페이지 포함)를 컨텍스트로. 표 행 사실 조각(`kind=fact`)은 전문에서 빼는 대신 **[핵심 근거]** 로 페이지마다 붙여 프롬프트 맨 앞에 보여준다(`HIGHLIGHT_MAX`) — 7B 가 긴 표에서 옆 행사 날짜를 섞던 결함 대응. 검색된 행만으로 상한이 안 차면 **고른 페이지의 행 조각 중 질문 낱말이 겹치는 줄**(`textutil.question_overlap`)로 채운다 — 09-18 실측: 정답 행(예산 총액·동아리 배정인원·수업공개 첫 날)이 색인은 됐는데 reranker 상위에 못 들어 프롬프트에도 폴백에도 없던 4문항 대응 |
| **2. 생성 입력** | [rag/prompt.py](rag/prompt.py) | 시스템 프롬프트 — `voice`(음성 안내용 3문장) / `full`(완전 나열) 두 벌 + `[문서]` 블록 + 멀티턴 메시지 조립 |
| **2. 생성** | [rag/generator.py](rag/generator.py) | LLM(A.X-4.0-Light — vLLM 또는 Ollama, `LLM_BACKEND`) 호출 + 스트리밍. `full` 모드에서만 자동 이어쓰기 |
| **2. 생성 보정** | [rag/continuation.py](rag/continuation.py) | (`full` 모드) 이어쓰기 발동 판단, 회차 결과 정리(중복·"끝" 마커·메타 발언 필터) |
| **2.5 검증** | [rag/verify.py](rag/verify.py) | voice 모드는 **인용 우선 생성**(`QUOTE_FIRST`): 모델이 `근거:` 줄에 [문서] 구절을 그대로 복사한 뒤 `답변:`. 검증 = 답변의 3자리 이상 숫자는 근거 텍스트에, **날짜(M월 D일)는 인용 구절에** 있어야 함. 실패하면 지적하며 1회 재생성(`VERIFY_RETRY`), 그래도 실패하면 인용 구절이 [문서]에 실제로 있는 줄일 때 **인용을 그대로 읽어 줌**(`QUOTE_ANSWER_PREFIX`), 아니면 확인 안내문. 실측(7B): 인용은 정확한데 옮겨 쓰며 틀림 — "근거: 1982.03.01. 개교 / 답변: 1980년", "근거: 계 255(5) / 답변: 2,555", 입학식↔졸업식 날짜 혼동. 09-18 추가: **동문서답 검사**(답에 질문 낱말이 하나도 없으면 관련 내용 없음 — "여기에 질문" 이 설문 수치로 답하던 실측) · **근거 있는 거절 재질문**("찾을 수 없다" 고 했는데 핵심 근거에 질문 낱말이 충분히 겹치는 줄이 있으면 그 줄을 짚어 주며 1회 더 — 질문 표현이 문서와 살짝 다를 때 7B 가 포기하던 실측) · **행 단위 수량 대조**("N층/N명…" 은 질문 대상이 적힌 근거 줄의 값이어야 함 — "행정실은 2층" 이 다른 줄의 2 로 통과하던 실측; 다른 값이면 인용 대신 질문 대상 줄을 읽어 줌) |
| 골든 채점 | [golden_eval.py](golden_eval.py) | 골든 세트(`golden/*.jsonl`) 채점 — 검색 근거 페이지 + 답변 키워드. `--only m01,m14` 로 일부만. 06 이 **세트마다 따로** 돌린다 |
| 골든 린트 | [golden_lint.py](golden_lint.py) | 골든셋 자체 검사 — 기대 키워드가 기대 페이지 텍스트에 실제로 있는지 (LLM 불필요, 1초). 문항을 새로 만들면 채점 전에 먼저 |
| 설정 | [rag/config.py](rag/config.py) | **모든 환경변수·캘리브레이션 값** — 값마다 실측 근거 주석 포함, 운영 조정은 여기부터 |

전체 데이터 흐름 도표는 [rag/__init__.py](rag/__init__.py) 모듈 독스트링에 있다.

검색 대상은 인덱싱 파이프라인([../indexing/](../indexing/))이 만든 `chroma_db`다.
검색기의 임베딩 모델·컬렉션 이름·접두사는 인덱싱 때와 **반드시 동일**해야 한다
(다르면 벡터 공간이 달라져 "가까움" 비교가 무의미해진다) — 해당 값들은 `rag/config.py`에 모여 있다.

## 설치 및 실행

먼저 준비돼 있어야 하는 것:
- 인덱싱 산출물: [../indexing/](../indexing/)를 `store_chroma.py`까지 실행해 `chroma_db` 생성
- LLM: `LLM_BACKEND=ollama`(개발 PC 기본)면 Ollama 데몬(:11434)에, `LLM_BACKEND=vllm`(운영 서버)이면 vLLM(:9000, OpenAI 호환)에 답변 생성 모델이 올라와 있어야 함 (아래 참고)

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8001
```

내부 전용 서비스이므로 `127.0.0.1` 바인딩이 기본이다. 오케스트레이터·키오스크가
**다른 호스트**에서 접근해야 할 때만 `--host 0.0.0.0`으로 열고, 방화벽에서 내부망만 허용할 것.

기동 시 검색기(임베딩 모델 + Chroma)를 한 번 로드한다. `chroma_db`가 없으면 서버는 뜨되
로그에 경고를 남기고, `/answer` 호출 시 503으로 알린다.

### 답변 LLM 백엔드 선택 (`LLM_BACKEND`)

| | `ollama` (기본 — 개발 PC) | `vllm` (운영 서버) |
|---|---|---|
| 호출 | `OLLAMA_URL/api/chat` | `VLLM_LLM_URL/chat/completions` (OpenAI 호환, 스트리밍은 SSE + `stream_options.include_usage`) |
| 모델 이름 | Ollama 태그 = `LLM_MODEL` | `vllm serve … --served-model-name $LLM_MODEL` |
| 컨텍스트 | 요청마다 `num_ctx` | 서버 기동 옵션 `--max-model-len` (요청에 안 보냄) |
| 언제 | 8GB 노트북, 동시 1~2건 | 키오스크 40대+가 한 서버 공유 — continuous batching |

호출 형식 차이는 `generator.py` 안에서만 흡수되고, 이어쓰기·스트리밍·프롬프트는 공통이다.
운영 서버 기동은 [deploy/gpu_server](../deploy/gpu_server/) 의 04단계가 한다 (절차·설명서는 저장소의 `문서/gpu_server/`).
로컬에서 vLLM 응답 형식만 검증하려면 OpenAI 호환 모의 서버로 `LLM_BACKEND=vllm VLLM_LLM_URL=http://127.0.0.1:<port>/v1` 을 주면 된다.

### 답변 모드 (`answer_mode` · `LLM_ANSWER_MODE`)

키오스크는 답을 **TTS 로 읽어 준다**. 개발 초기 프롬프트("표·목록을 빠짐없이 그대로")는 화면에 조직표를
완전하게 띄우던 때의 것이라, 음성에서는 담당자 30명·업무를 4~5분 낭독하게 된다(실측: "교무 업무 조직을
설명해줘" 1,500토큰·45.6초). 그래서 모드를 둘로 나눴다.

| 모드 | 프롬프트 | 출력 상한 | 이어쓰기 | 용도 |
|---|---|---|---|---|
| `voice` (기본) | 첫 문장에 핵심(누가·언제·어디서), 3문장 이내, 목록은 3항목 + "더 자세한 내용이 필요하시면…" | `LLM_NUM_PREDICT_VOICE`=512 | 없음 | 운영 키오스크 |
| `full` | 표·목록 완전 나열, 부서 질문은 소속 전원 | `LLM_NUM_PREDICT`=4096 | 최대 3회 | 테스트 화면, 골든셋 완전성 검증 (`golden_eval.py --mode full`) |

요청의 `answer_mode` 가 우선이고, 없으면 `LLM_ANSWER_MODE`(운영 `.env` 는 `voice`). 정확성 규칙(문서만 근거·지어내기 금지·
'근거 없음' 고정 문구)은 두 모드가 같다. 날짜·이름 키워드를 보는 골든 채점은 voice 로도 통과한다 — 짧아도 핵심 값은 들어간다.

### Ollama 준비 (2단계 생성, `LLM_BACKEND=ollama`)

기본 모델 태그는 `ax4-light`다. A.X-4.0-Light를 Ollama에 올리고 태그를 맞추거나,
`LLM_MODEL` 환경변수로 실제 태그를 지정한다.

```bash
ollama serve            # Ollama 데몬 (기본 :11434)
ollama list             # 올라온 모델 태그 확인
# A.X-4.0-Light가 없다면 준비한 뒤 그 태그를 LLM_MODEL로 지정
LLM_MODEL=<태그> uvicorn main:app --host 127.0.0.1 --port 8001
```

## 무관 질문 차단 (거리 임계값)

검색된 조각이 전부 `MAX_RETRIEVAL_DISTANCE`(기본 0.5, 실측 기반)보다 멀면 **LLM을 호출하지 않고**
고정 안내문(`NO_CONTEXT_ANSWER`)을 반환한다 — 매뉴얼과 무관한 질문("오늘 날씨 어때?")에
억지 답변을 만드는 환각을 원천 차단한다. 응답의 `grounded` 필드로 구분할 수 있다
(`true` = 근거 기반 답변, `false` = 근거 없음 안내문).

## API

- `GET /health` — 검색기 로드 상태 + 생성기(Ollama·모델) 상태
- `POST /answer` — 질문 텍스트 → 근거 기반 답변 (`X-Internal-Key` 필요)
- `POST /answer/stream` — 같은 입력, SSE로 토큰 단위 스트리밍 (`delta` → `done` 이벤트)
- `POST /reload` — 인덱싱 재실행 후 재기동 없이 Chroma 컬렉션 다시 열기 (`X-Internal-Key` 필요)

### 예시

```bash
curl -X POST "http://localhost:8001/answer" \
  -H "X-Internal-Key: dev-only-shared-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "스쿨패스는 어떻게 신청하나요?"}'
```

요청 본문:

```json
{
  "question": "그건 어디서 신청해요?",
  "institution_id": null,
  "top_k": 4,
  "history": [ { "question": "스쿨패스가 뭐예요?", "answer": "스쿨패스는 ..." } ]
}
```

- `institution_id` — 주면 공통 문서 + 해당 기관 문서만 검색 (기본 null → 공통 문서만)
- `top_k` — 검색할 근거 조각 수 (기본 `RETRIEVAL_TOP_K`, 다이어그램 기준 3~5)
- `history` — 직전 문답 목록(멀티턴). 오케스트레이터의 세션 저장소가 채워서 보낸다.
  후속 질문("그건 어디서 해요?")의 맥락을 LLM이 이어받는다.

응답:

```json
{
  "answer": "스쿨패스는 학교 홈페이지의 사전신청 서비스에 접속해 신청합니다. ...",
  "sources": [
    { "source_file": "스쿨패스 사전신청 안내문.pdf", "page_number": 1, "chunk_index": 0, "distance": 0.104 }
  ],
  "grounded": true
}
```

## 환경변수

**키 정책**: `INTERNAL_API_KEY` 미설정 시 기동을 거부한다. 로컬 개발에서만
`ALLOW_DEV_KEYS=1`로 개발용 기본 키를 쓸 수 있다 (운영 배포 금지).
파워셸 : $env:ALLOW_DEV_KEYS="1"
CMD : set ALLOW_DEV_KEYS=1

| 변수 | 기본값 | 설명 |
|---|---|---|
| `INTERNAL_API_KEY` | (필수) | 오케스트레이터와 공유하는 내부 키 (STT 서비스와 동일 값) |
| `MAX_RETRIEVAL_DISTANCE` | `0.5` | 이 코사인 거리보다 먼 조각은 근거로 쓰지 않음 (무관 질문 차단). bge-m3 실측: 관련 0.32~0.44 / 무관 0.58~0.66 |
| `NO_CONTEXT_ANSWER` | (고정 안내문) | 근거가 없을 때 반환하는 문구 |
| `CHROMA_DIR` | `../indexing/chroma_db` | 검색할 Chroma DB 경로 |
| `COLLECTION_NAME` | `schoolpass_docs` | Chroma 컬렉션 이름 (인덱싱과 동일) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 질문 임베딩 모델 (인덱싱과 반드시 동일) |
| `RERANK` | `1` | cross-encoder 재순위화 (GPU 실측 1.7~2초/질의). 전제: CUDA torch + 평시 vLLM 내려두기. GPU가 없으면 `RERANK=0` (CPU는 ~30초라 부적합) |
| `LEXICAL_CANDIDATES` | `20` | **하이브리드 검색** — 임베딩 후보에 질문 낱말이 가장 많이 든 짧은 조각(kind=fact) 상위 N 개를 합쳐 reranker 에 넘김. 09-18 rerank_probe 실측: 정답 행이 임베딩 상위 50 에 아예 없던 문항 2건(가정통신 앱, 7교시). reranker 컷은 그대로라 오탐은 늘지 않음. 0 이면 끔 |
| `RERANK_MODEL` / `RERANK_CANDIDATES` | `BAAI/bge-reranker-v2-m3` / `50` | 재채점 모델 / 임베딩에서 넘길 후보 수. 09-18: 산문 조각이 더해져 문서당 조각 2,100→2,500 이 되자 30 으로는 정답 행이 후보에 못 드는 문항이 생겨 50 으로 |
| `QUOTE_FIRST` / `QUOTE_ANSWER_PREFIX` | `1` / `자료에는 다음과 같이 적혀 있습니다. ` | 인용 우선 생성 켬 / 검증 실패 시 인용을 읽어 줄 때 앞에 붙는 말 |
| `VERIFY_RETRY` / `HIGHLIGHT_MAX` | `1` / `5` | 검증 실패 시 재생성 횟수 / [핵심 근거] 줄 수. 09-17 실측: 검색은 맞는데 7B 가 잘못 읽던 3문항(입학식↔졸업식 날짜 혼동, 1980 환각, 255(5)→2555) 대응 |
| `RERANK_RELATIVE_MIN` | `0.4` | 1위 점수 × 비율 미만 후보 제거 (컨텍스트 다이어트). 09-17 실측: 검색은 맞는데 곁다리 페이지(보건실 질문에 교실 수 표, 입학식 질문에 졸업식 페이지)가 딸려 와 7B 가 섞어 읽던 문항 교정. 0 이면 끔 |
| `RERANK_MIN_SCORE` / `RERANK_RESCUE_MARGIN` | `0.03` / `0.03` | 무관 차단 확률 / 인접 페이지 구제 여유. **09-17 재캘리브레이션** — 이전 0.55 는 이중 sigmoid 버그로 뭉개진 척도(0.50~0.73)였다. 실제 확률 실측: 무관 ≤0.012, 관련 정답 페이지 0.096~0.99. 표 페이지는 인덱싱의 **표 행 사실 조각**(indexing/README 2단계)으로 해결 — 행 조각 도입 후 재측정: 무관 최고 0.006, 관련 최저 0.016 → 0.03. `rerank_probe.py` 로 재측정 |
| `RETRIEVAL_TOP_K` | `6` | 검색할 근거 조각 수 (여러 페이지에 걸친 표의 뒷장까지 근거에 들도록 상향) |
| `PAGE_EXPAND` | `1` | 검색된 조각이 속한 페이지 전체를 컨텍스트로 병합 (small-to-big). 같은 문서의 연속 페이지는 한 단위로 묶음 |
| `CONTEXT_CHAR_BUDGET` | `10000` | 병합 컨텍스트 총량 상한(문자) |
| `LLM_NUM_CTX` | `16384` | LLM 컨텍스트 창(토큰) — Ollama 기본 2048은 병합 컨텍스트가 조용히 잘림 |
| `LLM_NUM_PREDICT` | `4096` | 답변 최대 출력 토큰 |
| `LLM_CONTINUE_ROUNDS` | `3` | 자동 이어쓰기 최대 횟수 — 긴 나열을 모델이 스스로 끊는 문제 대응 |
| `LLM_CONTINUE_MIN_TOKENS` | `400` | 이어쓰기 발동 조건: 1차 답변이 나열형이고 출력 토큰 ≥ 400일 때만 (컨텍스트 크기 조건은 오작동이 실측되어 제거됨) |
| `LLM_BACKEND` | `ollama` | `ollama`(개발 PC) / `vllm`(운영 서버) |
| `VLLM_LLM_URL` | `http://127.0.0.1:9000/v1` | vLLM OpenAI 호환 API 베이스 (`--served-model-name` 이 `LLM_MODEL` 과 같아야 함) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 주소 (`LLM_BACKEND=ollama`) |
| `LLM_MODEL` | `ax4-light` | 답변 생성 LLM 이름 — Ollama 태그 또는 vLLM `--served-model-name` |
| `LLM_ANSWER_MODE` | `voice` | 답변 모드 기본값. `voice`: 음성 안내용(첫 문장에 핵심, 3문장 이내, 목록 3항목, 이어쓰기 없음). `full`: 표·목록 완전 나열 + 이어쓰기(개발·화면용). 요청의 `answer_mode` 가 우선 |
| `LLM_NUM_PREDICT_VOICE` | `512` | voice 모드 출력 상한(토큰). full 은 `LLM_NUM_PREDICT` |
| `VERIFY_NUMBERS` | `1` | 답변의 숫자가 근거에 없으면 차단 (`grounded=false`, 응답의 `unverified` 에 해당 숫자). 0 이면 검사 안 함 |
| `LLM_TIMEOUT` | `300` | LLM 호출 타임아웃(초) |
| `LLM_TEMPERATURE` | `0` | 생성 온도 — 문서 그대로 옮기는 용도라 탐욕 디코딩(회차별 편차 제거) |

## 골든 질문 세트 채점 (`golden_eval.py`)

"문서가 정확하게 들어갔는가"는 인덱싱 로그가 아니라 **실제 질문으로 채점**해서 판단합니다.
`golden/*.jsonl`에 질문·기대 근거 페이지·기대 답변 키워드를 두고 돌립니다.

```bash
python golden_eval.py golden/mangmi_2026.jsonl            # 검색만 (엔진 in-process, Ollama 불필요)
python golden_eval.py golden/mangmi_2026.jsonl --answer   # 답변까지 (rag_service :8001 이 떠 있어야 함)
```

- 검색 통과: 기대 페이지 중 하나가 근거(페이지 병합 후)에 있음. `grounded: false` 문항은 근거 0건이어야 통과
- 답변 통과: 검색 통과 + 키워드 그룹(`"a|b"`는 둘 중 하나)이 답변에 모두 있음 + `exclude` 값이 하나도 없음 (옆 행 값과 헷갈리기 쉬운 문항: 시제품 50~150 vs 상용화 100~250 처럼 두 값을 섞어 말하면 오답)
- 필수 문항 통과율이 `--min-pass`(기본 90%) 미만이면 exit 1 → 배포 스크립트 `06_start_and_check.sh`가 경고
- 문서를 추가하면 그 문서용 골든 파일을 같이 추가합니다 (문항 형식은 `golden_eval.py` 머리말 참고)

세트는 문서마다 따로 두고 **따로** 채점합니다 (06 과 `golden_check.sh` 가 파일마다 돌리고 `golden/last_report_<세트>.json` 을 남김):

| 세트 | 문서 (docs/ 위치) | 문항 | 성격 |
|---|---|---|---|
| `golden/mangmi_2026.jsonl` | 망미중 교육계획서 (`common/`) | 30 (관련 28 + 무관 2) | 기본 — 09-17 에 29/29 × 3회 도달한 회귀 세트. 그대로 유지 |
| `golden/mangmi_2026_ext.jsonl` | 망미중 교육계획서 (`common/`) | 104 (관련 94 + 무관 10) | 확장 — 연혁·현황·조직·위원회·학사·동아리·봉사·평가·안전·보건·급식·배치도까지 문서 전반 + 뜻 없는 입력("테스트"·"여보세요") 거절. 09-18 신설 |
| `golden/paichai_2026.jsonl` | 배재고 학교소개 (`7010170/`) | 35 (관련 30 + 무관 5) | 기관 문서 — 문항마다 `"institution": "7010170"` 을 실어 키오스크처럼 공통+그 기관 범위로 검색. 09-18 신설 |
| `golden/aica_2026.jsonl` | 인공지능산업융합사업단 기관소개 (`Z039353/`) | 38 (관련 33 + 무관 5) | 기관 문서 — `"institution": "Z039353"`. 09-18 신설 |

기관 문서 세트의 `institution` 은 `/answer` 의 `institution_id` 로 그대로 전달됩니다 — 없으면 전체 문서에서 찾으므로 다른 기관 문서가 섞여 채점이 흐려집니다. 무관 문항도 같은 범위에서 거절되는지 봅니다.

```bash
# 서버에서 골든만 (서비스가 떠 있을 때) — 세트별·반복 실행·린트
bash ~/ai-agent/deploy/gpu_server/golden_check.sh                     # 모든 세트 각 1회, FAIL 줄과 총점만
bash ~/ai-agent/deploy/gpu_server/golden_check.sh -n 3 paichai_2026 aica_2026  # 기관 문서 세트만 3회
bash ~/ai-agent/deploy/gpu_server/golden_check.sh -n 3 mangmi_2026_ext # 확장 세트만 3회 (편차 확인)
bash ~/ai-agent/deploy/gpu_server/golden_check.sh --lint              # 골든셋 자체 검사 (golden_lint.py)
python golden_lint.py golden/mangmi_2026_ext.jsonl                     # 직접: 키워드가 기대 페이지에 있는지
```

골든셋을 만들 때는 `golden_lint.py` 를 먼저 통과시킵니다 — 기대 페이지에 키워드가 없는 문항은 서비스가 맞아도 FAIL 로 나와 채점을 오염시킵니다. 달력 칸(`12(목2)`)처럼 표기가 다른 값은 키워드에 대안을 넣습니다(`"3월 12일|12(목|3.12"`).
