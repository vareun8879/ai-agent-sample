# VLM Service

이미지를 이해해 텍스트로 옮기는 **내부 전용** 독립 서비스. `rag_service`가 "검색+생성"을
하나로 묶어 제공하듯, 이 서비스는 "이미지 이해"를 `/describe` 엔드포인트 하나로 제공한다.

호출자:

- **인덱싱 배치** ([../indexing/](../indexing/)): 텍스트 레이어가 없는 스캔본/이미지 PDF
  페이지를 렌더링해 이 서비스로 보내고, 표·레이아웃 구조를 살린 전사 텍스트를 받아
  같은 청킹·임베딩 파이프라인에 태운다
- **향후 실시간 질의응답**: 사용자가 화면/사진을 첨부해 질문하는 시나리오가 생기면
  오케스트레이터가 이 서비스를 직접 호출할 수 있다

비전 모델은 `VLM_BACKEND` 로 고른다.

| `VLM_BACKEND` | 모델 로드 | 언제 |
|---|---|---|
| `vllm` (기본) | 별도 **vLLM 서버(OpenAI 호환 API)** 에 위임 — `VLLM_URL` + `VLM_MODEL` | 운영 서버의 **Qwen3-VL-8B-Instruct-FP8**, 개발 PC 의 Qwen2.5-VL-3B-AWQ — vLLM 이 네이티브로 받는 모델 |
| `transformers` | **이 프로세스가 직접 로드** (`vlm/generator_hf.py`, `trust_remote_code`) — `VLM_MODEL` 또는 `VLM_HF_DIR` | vLLM 이 못 받는 커스텀 아키텍처 — 대안인 **SKT A.X-4.0-VL-Light** (답변 LLM 과 같은 계열, Apache-2.0). 기동 시 bf16 약 16GB 를 올리므로 첫 응답까지 1~3분 |

두 백엔드는 같은 인터페이스(`describe` → 텍스트 + 잘림 여부, `health`)라 호출자(인덱싱)는 차이를 모른다.

## 설치 및 실행

먼저 vLLM 서버에 비전 모델이 올라와 있어야 한다. 예: 현재 운영 중인 Qwen2.5-VL(:10001).

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 --port 10001 --max-model-len 32768
```

그다음 vlm_service를 띄운다. `VLM_MODEL`은 **필수**다 — vLLM의 `GET /v1/models`가
반환하는 id(`--served-model-name`을 줬다면 그 별칭)와 정확히 일치해야 하고, 미설정 시
기동을 거부한다.

```bash
pip install -r requirements.txt   # vlm_service 자체 의존성 (fastapi/uvicorn/httpx)
VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8 uvicorn main:app --host 127.0.0.1 --port 10000 --log-config log_config.json
```

모델 이름이 안 맞으면 `/health`의 `model_available`이 `false`로 나오고, 응답의
`served_models` 필드에서 서버가 실제로 서빙 중인 id 목록을 확인할 수 있다.
인덱싱 배치는 이 헬스체크가 통과해야만 시작된다.

`rag_service`·`stt_ondevice`와 동일하게 내부망에서만 접근 가능해야 하므로
`127.0.0.1` 바인딩을 유지한다 (오케스트레이터를 거치지 않은 직접 노출 금지).

## API

- `GET /health` — vLLM 연결 상태 + 대상 비전 모델 로드 여부 (`model_available`, 불일치 시 `served_models` 포함)
- `POST /describe` — 이미지 → 텍스트 (`X-Internal-Key` 필요, base64 기준 기본 20MB 상한)

### 예시

```bash
curl -X POST "http://localhost:10000/describe" \
  -H "X-Internal-Key: dev-only-shared-key" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -w0 page.png)\"}"
```

요청 본문:

```json
{
  "image_base64": "<PNG/JPEG를 base64로 인코딩한 문자열, data: 접두사 없이>",
  "prompt": null
}
```

- `image_base64` — 필수. `data:image/png;base64,` 같은 접두사는 붙이지 않는다
- `prompt` — 생략하면 문서 이미지(표·글자·화면 캡처)용 기본 프롬프트 사용. 호출자가 용도에
  맞게 덮어쓸 수 있다 (예: "이 화면에서 취소 버튼 위치를 설명하라")

응답:

```json
{ "text": "...", "truncated": false, "looped": false }
```

- `truncated` — 출력이 토큰 한도(`VLM_MAX_TOKENS` 또는 vLLM 컨텍스트)에 걸려 중간에
  잘렸으면 `true`. 호출자는 이 경우 불완전한 텍스트임을 알고 처리해야 한다
  (인덱싱 배치는 스캔본 페이지에서 이 값이 `true`면 실패로 처리한다)
- `looped` — 토큰 한도에 걸린 출력의 끝부분이 같은 문구의 반복 루프라 반복 직전까지
  잘라낸 경우 `true` (이때 `truncated`는 `false`). 사진 위주 페이지에서 소형 VLM이 흐릿한
  배경 글자를 되풀이해 읽는 사례(`vlm/repetition.py`) — 남은 본문은 정상 결과로 쓰면 된다

## 환경변수

**키 정책**: `INTERNAL_API_KEY` 미설정 시 기동을 거부한다. 로컬 개발에서만
`ALLOW_DEV_KEYS=1`로 개발용 기본 키를 쓸 수 있다 (운영 배포 금지). `rag_service`·
`orchestrator`와 같은 값을 공유해야 오케스트레이터가 내부 서비스로 호출할 수 있다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `INTERNAL_API_KEY` | (필수) | 다른 내부 서비스와 공유하는 내부 키 |
| `VLM_BACKEND` | `vllm` | `vllm` / `transformers` (위 표) |
| `VLM_HF_DIR` | (없음) | transformers 백엔드에서 로컬 가중치 디렉터리(`config.json` 이 있으면 이걸 로드, 없으면 `VLM_MODEL` 로 HF 에서 받음) |
| `VLM_DEVICE` | `cuda` | transformers 백엔드의 장치 |
| `VLM_MODEL` | (필수) | 대상 모델 id. vllm 백엔드에서는 vLLM의 `GET /v1/models` 응답(또는 `--served-model-name`)과 정확히 일치해야 함 |
| `VLLM_URL` | `http://localhost:10001/v1` | vLLM 서버 주소 (OpenAI 호환 API) |
| `VLM_TIMEOUT` | `120` | VLM 호출 타임아웃(초) |
| `VLM_TEMPERATURE` | `0.2` | 생성 온도 |
| `VLM_MAX_TOKENS` | `4096` | 페이지당 최대 응답 토큰 — 긴 페이지 잘림 방지를 위해 요청에 명시. vLLM `--max-model-len`이 (이미지 토큰 + 이 값)을 감당해야 함 |
| `VLM_MAX_IMAGE_MB` | `20` | `/describe` 요청 base64 크기 상한(MB) |
