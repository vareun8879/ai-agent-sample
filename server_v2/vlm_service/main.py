"""VLM 서비스 — 이미지를 이해해 텍스트로 옮기는 내부 전용 서비스.

rag_service가 "검색+생성"을 하나로 묶어 제공하듯, 이 서비스는 "이미지 이해"를
하나의 엔드포인트(/describe)로 제공한다. 필요할 때 아무 호출자나 갖다 쓸 수 있게
독립 서비스로 분리했다 — 지금 당장 붙는 호출자는 없고, 예를 들면:

- 인덱싱 배치: 스캔본/이미지 PDF 페이지를 렌더링해 이 서비스로 보내고,
  OCR보다 표·순서도 구조를 살린 텍스트를 받아 같은 청킹·임베딩 파이프라인에 태울 수 있다.
- 향후 실시간 질의응답: 사용자가 화면/사진을 첨부해 질문하는 시나리오가 생기면
  오케스트레이터가 이 서비스를 직접 호출할 수 있다.

rag_service와 동일하게 X-Internal-Key로 직접 접근을 막는다. 비전 모델은 VLM_BACKEND 로 고른다:
- vllm:         별도 vLLM 서버(OpenAI 호환)에 위임 — VLLM_URL·VLM_MODEL (vlm/generator.py)
- transformers: 이 프로세스가 직접 로드 — vLLM 이 못 받는 커스텀 아키텍처(SKT A.X-4.0-VL-Light) 대안용
                (vlm/generator_hf.py). 기동 시 모델(bf16 약 16GB)을 올리므로 첫 응답까지 1~3분.
"""

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from vlm import Generator
from vlm.repetition import trim_trailing_repetition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vlm-service")


def resolve_key(env_name: str, dev_default: str) -> str:
    """운영 키는 환경변수로만. 미설정 시 기동을 거부하고, ALLOW_DEV_KEYS=1일 때만 개발 키를 쓴다."""
    value = os.getenv(env_name, "")
    if value:
        return value
    if os.getenv("ALLOW_DEV_KEYS") == "1":
        logger.warning("%s 미설정 — 개발용 기본 키로 대체합니다. 운영 환경에서는 절대 사용 금지!", env_name)
        return dev_default
    raise RuntimeError(
        f"{env_name} 환경변수가 설정되지 않았습니다. "
        f"운영 키를 설정하거나, 로컬 개발에서만 ALLOW_DEV_KEYS=1과 함께 실행하세요."
    )


# 오케스트레이터(또는 인덱싱 스크립트)와 반드시 동일한 값이어야 함
INTERNAL_API_KEY = resolve_key("INTERNAL_API_KEY", "dev-only-shared-key")

# 요청 base64 크기 상한(MB) — 렌더링 DPI 설정 실수 등으로 비정상적으로 큰 이미지가
# 들어오면 vLLM 컨텍스트만 낭비하고 실패하므로 입구에서 거부한다
MAX_IMAGE_B64_BYTES = int(os.getenv("VLM_MAX_IMAGE_MB", "20")) * 1024 * 1024

generator = Generator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VLM 서비스 기동 — 대상 모델: %s (%s)", generator.model, generator.url)
    yield


app = FastAPI(title="VLM Service", version="0.1.0", lifespan=lifespan)


class DescribeRequest(BaseModel):
    # data: 접두사 없는 순수 base64 (PNG/JPEG)
    image_base64: str
    # 비워두면 문서 이미지용 기본 프롬프트 사용 — 호출자가 용도에 맞게 덮어쓸 수 있다
    prompt: Optional[str] = None


class DescribeResponse(BaseModel):
    text: str
    # 출력이 토큰 한도(VLM_MAX_TOKENS 또는 vLLM 컨텍스트)에 걸려 중간에 잘렸는지 여부.
    # 호출자는 이 값이 True면 불완전한 텍스트임을 알고 처리해야 한다.
    truncated: bool = False
    # 잘린 출력의 끝부분이 같은 문구의 반복 루프라 반복 직전까지 잘라낸 경우 True (truncated 는 False 로 돌려준다).
    # 사진 위주 페이지에서 소형 VLM 이 흐릿한 배경 글자를 되풀이해 읽는 실측 사례 — 남은 본문은 정상으로 쓴다.
    looped: bool = False


def verify_internal_key(x_internal_key: str = Header(...)):
    if not secrets.compare_digest(x_internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=403, detail="Direct access to VLM service is not allowed")


@app.get("/health")
async def health():
    return {"status": "ok", "generator": await generator.health()}


@app.post("/describe", response_model=DescribeResponse)
async def describe(req: DescribeRequest, _auth: None = Depends(verify_internal_key)):
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64가 비어 있습니다")
    if len(req.image_base64) > MAX_IMAGE_B64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"이미지가 너무 큽니다 ({len(req.image_base64)} > {MAX_IMAGE_B64_BYTES} bytes) — "
            "렌더링 DPI를 낮추거나 VLM_MAX_IMAGE_MB를 조정하세요",
        )

    start = time.time()
    try:
        text, truncated = await generator.describe(req.image_base64, req.prompt)
    except Exception as exc:
        logger.error("VLM 추론 실패: %s", exc)
        raise HTTPException(status_code=502, detail=f"VLM 추론 실패: {exc}")

    looped = False
    if truncated:
        raw_len = len(text)
        text, looped = trim_trailing_repetition(text)
        if looped:
            truncated = False
            logger.warning("VLM 출력이 반복 루프로 토큰 한도에 걸림 — 반복 직전까지 잘라 사용 (%d자 → %d자)", raw_len, len(text))
        else:
            logger.warning("VLM 출력이 토큰 한도로 잘림 (%d자) — VLM_MAX_TOKENS/--max-model-len 확인 필요", len(text))
    logger.info("VLM 추론 완료 (%.2fs, %d자%s)", time.time() - start, len(text),
                ", 잘림" if truncated else (", 반복 제거" if looped else ""))
    return DescribeResponse(text=text, truncated=truncated, looped=looped)
