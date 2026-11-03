"""RAG 서비스 — HTTP 계층 (앱 조립·인증·엔드포인트).

이 파일에는 파이프라인 로직이 없다. 질의응답 오케스트레이션은 전부
RAG 엔진(rag/engine.py)이 소유하고, 여기서는 세 가지만 한다:

1. 요청을 엔진 호출로 변환 (스키마 검증 포함)
2. 엔진 결과·예외를 HTTP 응답으로 변환 (상태 코드, SSE 이벤트 형식)
3. 접근 제어 (X-Internal-Key — 오케스트레이터만 호출하도록)

모듈 구성과 데이터 흐름 도표는 rag/__init__.py 독스트링 참고.
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import verify  # noqa: F401
from rag.prompt import split_quote_answer
from rag.config import NO_CONTEXT_ANSWER
from rag.engine import RagEngine, RetrieverNotReady

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-service")


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


# 오케스트레이터와 반드시 동일한 값이어야 함
INTERNAL_API_KEY = resolve_key("INTERNAL_API_KEY", "dev-only-shared-key")

# 질의응답 파이프라인 전체를 소유하는 엔진 — 이 파일은 엔진의 HTTP 어댑터다
engine = RagEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 임베딩 모델 + Chroma 컬렉션은 기동 시 한 번만 로드한다.
    # 실패해도 서버는 뜨되(/health로 확인 가능), /answer 호출 시 503으로 이유를 알린다.
    logger.info("엔진 로드 시작")
    start = time.time()
    try:
        engine.load()
        logger.info("엔진 로드 완료 (%.2fs)", time.time() - start)
    except Exception as exc:
        logger.error("엔진 로드 실패 (인덱싱 chroma_db 확인 필요): %s", exc)
    yield


app = FastAPI(title="RAG QA Service", version="1.2.0", lifespan=lifespan)


# ── 요청/응답 스키마 ──────────────────────────────────────────


class Turn(BaseModel):
    question: str
    answer: str


class AnswerRequest(BaseModel):
    question: str
    institution_id: Optional[str] = None
    top_k: Optional[int] = None
    # 직전 문답(멀티턴). 오케스트레이터의 세션 저장소가 채워서 보낸다.
    history: Optional[list[Turn]] = None
    # voice(기본, 음성 안내용 짧은 답) | full(표·목록 완전 나열 — 개발·화면 확인용). 없으면 서버 기본값(LLM_ANSWER_MODE)
    answer_mode: Optional[str] = None


class Source(BaseModel):
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    distance: Optional[float] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    # 검증에서 근거에 없다고 판정된 숫자들 (비어 있으면 통과). 답변은 이미 확인 안내문으로 바뀌어 있다
    unverified: list[str] = []
    # 검색된 근거로 생성한 답변이면 True, 근거가 없어 고정 안내문을 반환했으면 False
    grounded: bool = True


# ── 인증 ─────────────────────────────────────────────────────


def verify_internal_key(x_internal_key: str = Header(...)):
    # bytes로 비교 — str 비교는 비ASCII 헤더 값에서 TypeError(500)가 나는 엣지가 있다
    if not secrets.compare_digest(x_internal_key.encode("utf-8"), INTERNAL_API_KEY.encode("utf-8")):
        raise HTTPException(status_code=403, detail="Direct access to RAG service is not allowed")


# ── 변환 헬퍼 (도메인 객체 ↔ HTTP 표현) ──────────────────────


def _history_dicts(req: AnswerRequest) -> list[dict]:
    return [t.model_dump() for t in req.history or []]


def _sources(chunks: list[dict]) -> list[Source]:
    return [
        Source(
            source_file=c["source_file"],
            page_number=c["page_number"],
            chunk_index=c["chunk_index"],
            distance=c["distance"],
        )
        for c in chunks
    ]


def _single_done_event(answer_text: str, sources: list, grounded: bool) -> StreamingResponse:
    payload = json.dumps(
        {"type": "done", "answer": answer_text, "sources": sources, "grounded": grounded},
        ensure_ascii=False,
    )

    async def single():
        yield f"data: {payload}\n\n"

    return StreamingResponse(single(), media_type="text/event-stream")


# ── 엔드포인트 ───────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", **(await engine.health())}


@app.post("/reload")
def reload_index(_auth: None = Depends(verify_internal_key)):
    """인덱싱 파이프라인 재실행 후 서비스 재기동 없이 Chroma 컬렉션을 다시 연다."""
    try:
        start = time.time()
        engine.load()
        return {"status": "ok", "elapsed": round(time.time() - start, 2)}
    except Exception as exc:
        logger.error("재로드 실패: %s", exc)
        raise HTTPException(status_code=500, detail=f"재로드 실패: {exc}")


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest, _auth: None = Depends(verify_internal_key)):
    start = time.time()
    logger.info("질문 수신: %r", req.question)

    try:
        result = await engine.answer(
            req.question,
            history=_history_dicts(req),
            top_k=req.top_k,
            institution_id=req.institution_id,
            mode=req.answer_mode,
        )
    except RetrieverNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("질의응답 파이프라인 실패 (%.2fs): %s", time.time() - start, exc)
        # 내부 예외 문자열(백엔드 URL 등)을 응답에 싣지 않는다 — 상세는 위 로그로만
        raise HTTPException(status_code=502, detail="질의응답 파이프라인 실패 (서버 로그 확인)")

    logger.info("답변 생성 완료 (%.2fs, grounded=%s): %r", time.time() - start, result["grounded"], result["answer"])
    return AnswerResponse(
        answer=result["answer"],
        sources=_sources(result["chunks"]),
        grounded=result["grounded"],
        unverified=result.get("unverified") or [],
    )


@app.post("/answer/stream")
async def answer_stream(req: AnswerRequest, _auth: None = Depends(verify_internal_key)):
    """/answer의 SSE 스트리밍 버전.

    이벤트 형식:
    - data: {"type": "delta", "text": "..."}          — 답변 조각
    - data: {"type": "done", "answer": ..., "sources": [...], "grounded": ...}  — 완료
    - data: {"type": "error", "detail": "..."}         — 생성 중 오류

    스트리밍은 프로토콜 특성상 근거를 먼저 확정해야 해서(done 이벤트의 sources),
    엔진의 retrieve_context → answer_stream을 두 단계로 호출한다.
    """
    question = req.question.strip()
    start = time.time()
    logger.info("질문 수신(stream): %r", question)

    if not question:
        return _single_done_event("", [], grounded=False)

    try:
        chunks = await engine.retrieve_context_async(question, req.top_k, req.institution_id)
    except RetrieverNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not chunks:
        logger.info("근거 없음 (임계값 초과) — 고정 안내문 스트림 반환 (%.2fs): %r", time.time() - start, question)
        return _single_done_event(NO_CONTEXT_ANSWER, [], grounded=False)

    history = _history_dicts(req)
    sources = [s.model_dump() for s in _sources(chunks)]

    async def event_stream():
        raw_parts: list[str] = []
        try:
            async for piece in engine.answer_stream(question, chunks, history, req.answer_mode, raw_parts):
                yield f"data: {json.dumps({'type': 'delta', 'text': piece}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("스트리밍 생성 실패 (%.2fs): %s", time.time() - start, exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': '답변 생성 실패 (서버 로그 확인)'}, ensure_ascii=False)}\n\n"
            return
        # delta 는 '답변:' 이후만 흘렀지만 검증에는 인용 구절이 필요하므로 원문 전체를 다시 나눈다
        answer_text, quote = split_quote_answer("".join(raw_parts).strip())
        # 2.5단계 검증 — 스트림은 이미 조각이 나갔지만, 최종 answer 를 바꿔 보내면 클라이언트(키오스크)가
        # 그것을 TTS 로 읽는다 (delta 는 화면 미리보기용). 검증 실패 시 비스트리밍으로 1회 재생성.
        answer_text, ok, bad = await engine.verify_with_retry(question, chunks, history, req.answer_mode, answer_text, quote)
        logger.info("답변 생성 완료(stream) (%.2fs, verified=%s): %r", time.time() - start, ok, answer_text)
        done = {"type": "done", "answer": answer_text, "sources": sources, "grounded": ok, "unverified": bad}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
