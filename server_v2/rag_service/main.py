"""기존 /answer API 호환 계층. 구조화 출력은 검사 후 최종 문장으로 전달한다."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import httpx
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag.config import Settings
from rag.generation import Generator
from rag.pipeline import Pipeline
from rag.retrieval import Search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


def auth(x_internal_key: str = Header(...)):
    key = os.environ.get("INTERNAL_API_KEY", "")
    if not key or not secrets.compare_digest(key, x_internal_key):
        raise HTTPException(403, "Invalid API key")


@asynccontextmanager
async def lifespan(app):
    if not os.environ.get("INTERNAL_API_KEY"):
        raise RuntimeError("INTERNAL_API_KEY is required")
    settings = Settings.from_env()
    app.state.ready = False
    app.state.settings = settings
    app.state.slots = asyncio.Semaphore(settings.concurrency)
    app.state.search = Search(settings)
    await asyncio.to_thread(app.state.search.load)
    async with httpx.AsyncClient() as client:
        app.state.client = client
        app.state.pipeline = Pipeline(settings, app.state.search, Generator(settings, client))
        app.state.ready = True
        yield
        app.state.ready = False


app = FastAPI(title="RAG QA Service v2", lifespan=lifespan)


class Turn(BaseModel):
    question: str = Field(max_length=4000)
    answer: str = Field(max_length=8000)


class Request(BaseModel):
    question: str = Field(max_length=4000)
    institution_id: str | None = None
    history: list[Turn] = Field(default_factory=list, max_length=10)
    top_k: int | None = Field(default=None, ge=1, le=12)
    answer_mode: str | None = None


@app.get("/health")
async def health():
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "not ready")
    try:
        response = await app.state.client.get(app.state.settings.llm_url + "/models", timeout=3)
        response.raise_for_status()
        names = [x.get("id") for x in response.json().get("data", [])]
        if app.state.settings.llm_model not in names:
            raise ValueError("configured model unavailable")
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, "LLM not ready") from exc
    return {"status": "ok", "implementation": "v2", "index_revision": app.state.settings.revision,
            "validation_mode": app.state.settings.validation_mode}


@app.post("/reload", dependencies=[Depends(auth)])
async def reload_index():
    # 모델은 그대로 유지하며 검색 자료만 다시 읽는다. 인덱스 버전 교체는 서비스 재시작으로.
    await asyncio.to_thread(app.state.search.reload)
    return {"status": "ok", "index_revision": app.state.settings.revision}


async def run(req):
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "not ready")
    if req.answer_mode not in (None, "voice", "full"):
        raise HTTPException(422, "invalid answer_mode")
    if not req.question.strip():
        return {"answer": "", "sources": [], "grounded": False, "unverified": []}
    try:
        async with asyncio.timeout(app.state.settings.request_timeout):
            async with app.state.slots:
                out = await app.state.pipeline.answer(req.question.strip(), req.institution_id,
                    [t.model_dump() for t in req.history], req.answer_mode or "voice", req.top_k)
                return out.response()
    except TimeoutError as exc:
        raise HTTPException(504, "answer time budget exceeded") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "model service request failed") from exc


@app.post("/answer", dependencies=[Depends(auth)])
async def answer(req: Request):
    return await run(req)


@app.post("/answer/stream", dependencies=[Depends(auth)])
async def stream(req: Request):
    async def events():
        try:
            out = await run(req)
            # 내부 JSON, 인용, 검사 전 문장은 공개 스트림에 넣지 않는다.
            yield "data: " + json.dumps({"type": "delta", "text": out["answer"]}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "done", **out}, ensure_ascii=False) + "\n\n"
        except HTTPException as exc:
            yield "data: " + json.dumps({"type": "error", "detail": exc.detail}, ensure_ascii=False) + "\n\n"
        except Exception:
            logging.exception("SSE answer failed")
            yield 'data: {"type":"error","detail":"답변 처리 실패"}\n\n'
    return StreamingResponse(events(), media_type="text/event-stream")
