"""Orchestrator — 키오스크가 직접 호출하는 얇은 질의응답 창구.

상용 구조에서 STT는 키오스크 안(stt_ondevice, Qwen3-ASR 온디바이스)에서 끝난다.
키오스크는 전사된 질문 텍스트만 이 창구로 보내고, 오케스트레이터는
질문 게이트 → rag_service 위임 → 세션·로그 기록만 담당한다.
무거운 모델은 하나도 들지 않는다.
"""

import json
import logging
import os
import re
import secrets
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")


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


# rag_service와 반드시 동일한 값이어야 함
INTERNAL_API_KEY = resolve_key("INTERNAL_API_KEY", "dev-only-shared-key")
# 키오스크 → 오케스트레이터 호출용 키 (내부 키와 별도로 관리)
KIOSK_API_KEY = resolve_key("KIOSK_API_KEY", "dev-only-kiosk-key")
INTERNAL_HEADERS = {"X-Internal-Key": INTERNAL_API_KEY}

# LLM 답변 생성까지 포함하므로 넉넉히 잡는다
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "330"))

# --- 질문 게이트 ---
# 온디바이스 STT가 넘긴 전사가 이 기준에 못 미치면 검색/생성을 태우지 않고 재발화를 요청한다.
MIN_QUESTION_CHARS = int(os.getenv("MIN_QUESTION_CHARS", "2"))
RETRY_MESSAGE = os.getenv("STT_RETRY_MESSAGE", "질문을 잘 알아듣지 못했습니다. 다시 한번 말씀해 주세요.")

# --- 인사말 게이트 ---
# "안녕하세요" 같은 인사·시험 발화를 검색에 태우면 매뉴얼의 인사 예절 조항("안녕히 계십시오" 등)이
# 걸려 grounded 답변이 나온다(2026-09-19 실측: 망미중 p76 교무실 출입 인사말을 읽어 줌).
# 인사만 있는 입력은 검색·LLM 없이 고정 인사말로 응답한다. 인사 뒤에 질문이 붙어 있으면 통과.
GREETING_MESSAGE = os.getenv(
    "GREETING_MESSAGE", "안녕하세요. 무엇을 도와드릴까요? 궁금한 내용을 말씀해 주세요."
)
_GREETING_WORDS = (
    r"안녕|안녕하세요|안녕하십니까|안녕하세여|안녕하셔요|안뇽|하이|헬로|헬로우|hello|hi|hey|"
    r"반갑습니다|반가워요|반가워|반갑네요|방가워요|방가워|방가|잘 지내세요|잘 지내|잘 지냈어|오랜만이야|오랜만이에요|"
    r"수고하세요|수고하십니까|수고했어|고맙습니다|감사합니다|감사해요|감사|고마워요|고마워|고맙다|땡큐|"
    r"잘 부탁드립니다|잘 부탁해요|좋은 아침|좋은 하루|좋은 하루 되세요|안녕히 계세요|안녕히 가세요|잘 가|바이|bye|"
    r"테스트|마이크 테스트|시험|여보세요|저기요|계세요|계십니까"
)


def _spaces_optional(pattern: str) -> str:
    """패턴 속 띄어쓰기를 '있어도 없어도 됨'(\s*)으로 바꾼다 — "잘지내세요", "마이크테스트"처럼 붙여 쓴 입력도 잡는다."""
    return pattern.replace(" ", r"\s*")


_GREETING_RE = re.compile(
    r"^(?:[아어음네예]\s*)*(?:(?:" + _spaces_optional(_GREETING_WORDS) + r")[\s!.,~?요]*){1,3}$", re.IGNORECASE
)

# "너 뭐야", "역할이 뭐야", "뭘 할 수 있어" 같은 자기소개 요구도 매뉴얼에 없어 RAG로 가면 "자료에서 찾을 수 없다"로
# 끝난다. 고정 소개 문구로 답한다. 기관마다 문구가 다르면 INTRO_MESSAGE 로 바꾼다.
INTRO_MESSAGE = os.getenv(
    "INTRO_MESSAGE",
    "저는 이 기관의 안내 도우미입니다. 안내 자료에 있는 내용을 질문해 주시면 찾아서 답해 드립니다. 무엇이 궁금하세요?",
)
_INTRO_RE = re.compile(_spaces_optional(
    r"^(?:[아어음네예]\s*)*(?:"
    r"(?:너|넌|너는|니|네|당신|당신은|자네)\s*(?:는|은|의|가)?\s*"
    r"(?:뭐야|뭐니|뭐냐|누구야|누구니|누구세요|누구예요|누구입니까|뭐하는|뭘 하는|무엇을 하는|어떤|뭐 하는)"
    r"(?:\s*(?:애|거|봇|시스템|프로그램|기계|로봇|사람)(?:야|니|냐|이야|예요|에요|입니까)?)?"
    r"|(?:(?:너|니|네|당신|자네)\s*(?:의)?\s*)?(?:역할|이름|정체|기능|하는 일)\s*(?:은|는|이|가)?\s*(?:뭐야|뭐니|뭐냐|뭐지|무엇|뭐예요|뭐에요|알려줘|알려 줘)"
    r"|(?:뭘|뭐를|무엇을|무슨 일을|어떤 걸|어떤 것을|뭐)\s*(?:할 수 있|해 줄 수 있|해줄 수 있|도와줄 수 있|도와 줄 수 있|해줘|해 줘|도와줘|도와 줘)"
    r"|(?:누구세요|누구십니까|누구야|누구니|정체가 뭐야|자기\s*소개|소개)(?:\s*(?:해줘|해 줘|해주세요|해 주세요|좀|부탁해|부탁))?"
    r")[\s!.,~?요어지니냐]*$"),
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """인사·감사·시험 발화만으로 이루어진 입력이면 True (질문이 함께 있으면 False)."""
    return bool(_GREETING_RE.match(text.strip()))


def is_intro_request(text: str) -> bool:
    """챗봇 자기소개·역할을 묻는 입력이면 True ("너 뭐야", "역할이 뭐야", "뭘 할 수 있어")."""
    return bool(_INTRO_RE.match(text.strip()))


def canned_answer(text: str) -> Optional[str]:
    """검색 없이 고정 문구로 답할 입력이면 그 문구, 아니면 None."""
    if is_greeting(text):
        return GREETING_MESSAGE
    if is_intro_request(text):
        return INTRO_MESSAGE
    return None

# --- 멀티턴 세션 ---
SESSION_TTL = float(os.getenv("SESSION_TTL", "600"))
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "3"))

# --- 질문·답변 기록 (품질 분석용 JSONL) ---
QA_LOG_PATH = Path(os.getenv("QA_LOG_PATH", str(Path(__file__).resolve().parent / "logs" / "qa_log.jsonl")))


def evaluate_question_gate(text: str) -> Optional[str]:
    """질문 텍스트를 질의응답에 태울 수 있는지 판정한다. 통과면 None, 아니면 사유를 반환."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) < MIN_QUESTION_CHARS:
        return "too_short"
    return None


class SessionStore:
    """키오스크 대화 세션의 최근 문답을 기억하는 인메모리 저장소.

    오케스트레이터 프로세스가 하나뿐인 현재 구조에서는 이걸로 충분하다.
    수평 확장 시에는 Redis 등 외부 저장소로 교체해야 한다.
    """

    def __init__(self, ttl: float = SESSION_TTL, max_turns: int = SESSION_MAX_TURNS) -> None:
        self._ttl = ttl
        self._max_turns = max_turns
        self._sessions: dict[str, dict] = {}

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, s in self._sessions.items() if now - s["last_used"] > self._ttl]
        for sid in expired:
            del self._sessions[sid]

    def history(self, session_id: str) -> list[dict]:
        self._prune()
        session = self._sessions.get(session_id)
        if not session:
            return []
        session["last_used"] = time.monotonic()
        return [{"question": q, "answer": a} for q, a in session["turns"]]

    def append(self, session_id: str, question: str, answer: str) -> None:
        self._prune()
        session = self._sessions.setdefault(
            session_id, {"turns": deque(maxlen=self._max_turns), "last_used": time.monotonic()}
        )
        session["turns"].append((question, answer))
        session["last_used"] = time.monotonic()


sessions = SessionStore()


def log_qa(record: dict) -> None:
    """질문·답변 1건을 JSONL로 남긴다. 로깅 실패가 응답을 막으면 안 되므로 예외는 삼킨다."""
    try:
        QA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(QA_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("QA 로그 기록 실패: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 커넥션 풀을 재사용하도록 앱 수명주기 동안 클라이언트 하나만 유지한다
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="Kiosk Orchestrator", version="1.0.0", lifespan=lifespan)


def verify_kiosk_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, KIOSK_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
async def health():
    try:
        resp = await app.state.client.get(f"{RAG_SERVICE_URL}/health", timeout=5.0)
        rag_status = resp.json()
    except Exception as exc:
        rag_status = {"status": "unreachable", "error": str(exc)}
    return {"status": "ok", "rag_service": rag_status}


async def _answer(
    question: str, history: list[dict], institution_id: Optional[str] = None, answer_mode: Optional[str] = None
) -> dict:
    # 2단계 질의응답 파이프라인(검색 + 생성)은 rag_service가 담당한다 — 내부 HTTP 호출
    try:
        rag_resp = await app.state.client.post(
            f"{RAG_SERVICE_URL}/answer",
            json={"question": question, "history": history, "institution_id": institution_id, "answer_mode": answer_mode},
            headers=INTERNAL_HEADERS,
            timeout=RAG_TIMEOUT,
        )
        rag_resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("RAG service returned error: %s", exc)
        # rag_service가 준비 안 됨(503) 등은 그 상태코드를 그대로 전달
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.HTTPError as exc:
        logger.error("RAG service call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"질의응답 서비스 호출 실패: {exc}")

    return rag_resp.json()


def _greeting_response(request_id: str, question: str, answer: str) -> dict:
    """인사말·자기소개 요구에는 검색 없이 고정 문구로 답한다. 키오스크는 grounded=False·출처 없음으로 받는다."""
    return {
        "request_id": request_id,
        "question": question,
        "answer": answer,
        "sources": [],
        "grounded": False,
        "needs_retry": False,
        "greeting": True,
    }


def _retry_response(request_id: str, question: str, reason: str) -> dict:
    return {
        "request_id": request_id,
        "question": question,
        "answer": "",
        "sources": [],
        "needs_retry": True,
        "retry_reason": reason,
        "message": RETRY_MESSAGE,
    }


class TextQuestion(BaseModel):
    question: str
    session_id: Optional[str] = None
    # 키오스크가 설치된 기관 코드. 주면 공통 문서 + 해당 기관 문서를 검색 대상으로 삼는다 (rag_service로 그대로 전달)
    institution_id: Optional[str] = None
    # voice(기본): 음성 안내용 3문장 답변. full: 표·목록 완전 나열 (테스트 화면·개발용). rag_service 로 그대로 전달
    answer_mode: Optional[str] = None


@app.post("/ask/text")
async def ask_text(req: TextQuestion, _auth: None = Depends(verify_kiosk_key)):
    """키오스크가 온디바이스 STT(stt_ondevice)로 전사한 질문 텍스트를 보내는 경로.

    rag_service를 키오스크가 직접 부르지 않고 이 창구를 거치게 해서,
    내부 서비스는 X-Internal-Key 뒤에 그대로 숨긴다.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.time()
    question = req.question.strip()

    gate_reason = evaluate_question_gate(question)
    if gate_reason:
        log_qa({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "session_id": req.session_id,
            "institution_id": req.institution_id,
            "question": question,
            "needs_retry": True,
            "retry_reason": gate_reason,
            "elapsed": round(time.time() - started, 2),
        })
        return _retry_response(request_id, question, gate_reason)

    canned = canned_answer(question)
    if canned:
        log_qa({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "session_id": req.session_id,
            "institution_id": req.institution_id,
            "question": question,
            "needs_retry": False,
            "greeting": True,
            "answer": canned,
            "elapsed": round(time.time() - started, 2),
        })
        return _greeting_response(request_id, question, canned)

    history = sessions.history(req.session_id) if req.session_id else []
    rag_result = await _answer(question, history, req.institution_id, req.answer_mode)
    answer_text = rag_result.get("answer", "")

    if req.session_id:
        sessions.append(req.session_id, question, answer_text)

    log_qa({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": request_id,
        "session_id": req.session_id,
        "institution_id": req.institution_id,
        "question": question,
        "needs_retry": False,
        "answer": answer_text,
        "grounded": rag_result.get("grounded"),
        "sources": [s.get("source_file") for s in rag_result.get("sources", [])],
        "elapsed": round(time.time() - started, 2),
    })

    return {
        "request_id": request_id,
        "question": question,
        "answer": answer_text,
        "sources": rag_result.get("sources", []),
        "grounded": rag_result.get("grounded"),
        "needs_retry": False,
    }


@app.post("/ask/text/stream")
async def ask_text_stream(req: TextQuestion, _auth: None = Depends(verify_kiosk_key)):
    """/ask/text의 스트리밍 버전 — rag_service의 SSE를 그대로 중계한다.

    답변 전체가 생성될 때까지 기다리지 않고 토큰 단위로 흘려보내므로,
    키오스크 화면에 첫 글자가 뜨기까지의 체감 대기가 크게 줄어든다.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.time()
    question = req.question.strip()

    gate_reason = evaluate_question_gate(question)
    if gate_reason:
        payload = json.dumps({"type": "done", **_retry_response(request_id, question, gate_reason)}, ensure_ascii=False)

        async def single_event():
            yield f"data: {payload}\n\n"

        return StreamingResponse(single_event(), media_type="text/event-stream")

    canned = canned_answer(question)
    if canned:
        log_qa({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "session_id": req.session_id,
            "institution_id": req.institution_id,
            "question": question,
            "needs_retry": False,
            "greeting": True,
            "answer": canned,
            "elapsed": round(time.time() - started, 2),
            "stream": True,
        })
        payload = json.dumps({"type": "done", **_greeting_response(request_id, question, canned)}, ensure_ascii=False)

        async def greeting_event():
            yield f"data: {payload}\n\n"

        return StreamingResponse(greeting_event(), media_type="text/event-stream")

    history = sessions.history(req.session_id) if req.session_id else []

    async def event_stream():
        answer_parts: list[str] = []
        final_info: dict = {}
        try:
            async with app.state.client.stream(
                "POST",
                f"{RAG_SERVICE_URL}/answer/stream",
                json={"question": question, "history": history, "institution_id": req.institution_id, "answer_mode": req.answer_mode},
                headers=INTERNAL_HEADERS,
                timeout=RAG_TIMEOUT,
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    yield f"data: {json.dumps({'type': 'error', 'detail': body}, ensure_ascii=False)}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    yield line + "\n\n"
                    # 세션·로그용으로 델타를 누적한다
                    try:
                        event = json.loads(line[len("data: "):])
                    except ValueError:
                        continue
                    if event.get("type") == "delta":
                        answer_parts.append(event.get("text", ""))
                    elif event.get("type") == "done":
                        final_info = event
        except httpx.HTTPError as exc:
            logger.error("RAG stream call failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            return

        answer_text = final_info.get("answer") or "".join(answer_parts)
        if req.session_id and answer_text:
            sessions.append(req.session_id, question, answer_text)
        log_qa({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "session_id": req.session_id,
            "institution_id": req.institution_id,
            "question": question,
            "needs_retry": False,
            "answer": answer_text,
            "grounded": final_info.get("grounded"),
            "sources": [s.get("source_file") for s in final_info.get("sources", [])],
            "elapsed": round(time.time() - started, 2),
            "stream": True,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
