"""질의응답 2단계 — 생성(Generation).

1단계에서 검색·병합한 근거를 프롬프트(prompt.py)로 조립해 LLM에 전달하고,
근거 기반 답변을 생성한다. 긴 나열형 답변이 중간에 끊기면 자동 이어쓰기
(continuation.py)로 나머지를 이어 받는다.

- 모델: A.X-4.0-Light (config.LLM_MODEL). 이 서비스가 모델을 직접 안고 있지 않고,
  별도 프로세스로 떠 있는 서빙 서버의 REST API를 호출한다. 서빙 백엔드는 둘 중 하나:
    · ollama (LLM_BACKEND=ollama, 개발 PC 기본): Ollama `/api/chat`
    · vllm   (LLM_BACKEND=vllm, 운영 서버):     vLLM OpenAI 호환 `/v1/chat/completions`
  키오스크 40대+가 서버 하나를 같이 쓰므로 운영은 vLLM(continuous batching)이고,
  8GB 개발 노트북에서는 vLLM이 못 돌아 Ollama를 남겨 둔다. 두 백엔드는 같은 모델·
  같은 프롬프트를 쓰고, 요청/응답 형식만 다르다 — 그 차이는 이 파일 안에서만 흡수한다.
- 멀티턴: 직전 문답(history)을 대화 메시지로 앞에 붙인다.
- 스트리밍: 키오스크 체감 대기를 줄이기 위한 토큰 단위 스트리밍(generate_stream) 지원.
  1회차는 그대로 중계하고, 이어쓰기 회차는 "끝" 응답을 사용자에게 노출하지 않기 위해
  비스트리밍으로 받아 정리 후 통째로 흘려보낸다.
"""

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from .config import (
    LOG_PROMPT,
    QUOTE_FIRST,
    LLM_ANSWER_MODE,
    LLM_BACKEND,
    LLM_CONTINUE_MIN_TOKENS,
    LLM_CONTINUE_ROUNDS,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_NUM_PREDICT_VOICE,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_URL,
    VLLM_LLM_URL,
)
from .continuation import CONTINUE_PROMPT, clean_continuation, looks_enumeration
from .prompt import (  # noqa: F401 (하위 호환 재노출)
    ANSWER_LABEL,
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
    retry_prompt,
    split_quote_answer,
    split_retry_answer,
)

logger = logging.getLogger("rag.generator")

_BACKENDS = ("ollama", "vllm")
_MODES = ("voice", "full")


def resolve_mode(mode: Optional[str]) -> str:
    """요청의 answer_mode → 유효한 모드. 없거나 이상하면 config 기본값(LLM_ANSWER_MODE)."""
    m = (mode or LLM_ANSWER_MODE or "voice").lower()
    return m if m in _MODES else ("full" if LLM_ANSWER_MODE == "full" else "voice")


class Generator:
    """근거 조각 + 질문 (+ 직전 문답) → LLM 답변."""

    def __init__(self) -> None:
        if LLM_BACKEND not in _BACKENDS:
            raise ValueError(f"LLM_BACKEND 값이 잘못됐습니다: {LLM_BACKEND!r} (ollama | vllm)")
        self.backend = LLM_BACKEND
        # vLLM 은 OpenAI 호환 베이스(…/v1)까지가 URL 이다
        self.url = (OLLAMA_URL if self.backend == "ollama" else VLLM_LLM_URL).rstrip("/")
        self.model = LLM_MODEL
        logger.info("LLM 백엔드: %s (%s, model=%s)", self.backend, self.url, self.model)

    # ── 백엔드별 요청/응답 형식 ──────────────────────────────
    #
    # 이 구획 밖(생성 루프·이어쓰기)은 백엔드를 모른다. 새 백엔드를 붙일 때도 여기만 늘린다.

    def _chat_url(self) -> str:
        return f"{self.url}/api/chat" if self.backend == "ollama" else f"{self.url}/chat/completions"

    def _payload(self, messages: list[dict], stream: bool, max_tokens: int = LLM_NUM_PREDICT) -> dict:
        if self.backend == "ollama":
            return {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_ctx": LLM_NUM_CTX,
                    "num_predict": max_tokens,
                },
            }
        # vLLM: 컨텍스트 창은 서버 기동 옵션(--max-model-len)으로 고정되어 요청마다 주지 않는다.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": max_tokens,
        }
        if stream:
            # 마지막 청크에 usage(출력 토큰 수)를 실어 달라 — 이어쓰기 발동 판단에 쓴다
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _parse_once(self, data: dict) -> tuple[str, int]:
        """비스트리밍 응답 → (답변, 출력 토큰 수)."""
        if self.backend == "ollama":
            return data["message"]["content"].strip(), int(data.get("eval_count") or 0)
        content = data["choices"][0]["message"]["content"] or ""
        return content.strip(), int((data.get("usage") or {}).get("completion_tokens") or 0)

    async def _chat_once(
        self, client: httpx.AsyncClient, messages: list[dict], max_tokens: int = LLM_NUM_PREDICT
    ) -> tuple[str, int]:
        """1회 생성 → (답변, 출력 토큰 수). 토큰 수는 자동 이어쓰기 여부 판단에 쓴다."""
        resp = await client.post(self._chat_url(), json=self._payload(messages, False, max_tokens))
        resp.raise_for_status()
        return self._parse_once(resp.json())

    async def _stream_once(
        self, client: httpx.AsyncClient, messages: list[dict], stats: dict, max_tokens: int = LLM_NUM_PREDICT
    ) -> AsyncIterator[str]:
        """1회 생성을 조각 단위로 흘려보낸다. 끝나면 stats["tokens"] 에 출력 토큰 수를 남긴다.

        Ollama 는 줄 단위 JSON(마지막 줄 done=true 에 eval_count), vLLM 은 SSE(`data: {...}`,
        마지막에 `data: [DONE]`, 그 직전 청크의 usage 에 completion_tokens)다.
        """
        stats["tokens"] = 0
        pieces = 0
        async with client.stream("POST", self._chat_url(), json=self._payload(messages, True, max_tokens)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if self.backend == "vllm":
                    if not line.startswith("data:"):
                        continue
                    line = line[len("data:"):].strip()
                    if line == "[DONE]":
                        break
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if self.backend == "ollama":
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        pieces += 1
                        yield piece
                    if data.get("done"):
                        stats["tokens"] = int(data.get("eval_count") or 0)
                        break
                else:
                    usage = data.get("usage")
                    if usage:  # include_usage 로 요청한 마지막 청크 (choices 는 비어 있다)
                        stats["tokens"] = int(usage.get("completion_tokens") or 0)
                    choices = data.get("choices") or []
                    if choices:
                        piece = (choices[0].get("delta") or {}).get("content") or ""
                        if piece:
                            pieces += 1
                            yield piece
        if not stats["tokens"]:
            # usage 를 못 받은 경우(구버전 vLLM 등) — 청크 수로 근사 (청크 ≈ 토큰)
            stats["tokens"] = pieces

    # ── 생성 (비스트리밍) ─────────────────────────────────────

    async def generate(
        self, question: str, chunks: list[dict], history: Optional[list[dict]] = None, mode: Optional[str] = None
    ) -> str:
        """답변 문장만 (인용 우선 형식이면 '답변:' 이후)."""
        return (await self.generate_with_quote(question, chunks, history, mode))[0]

    async def generate_with_quote(
        self, question: str, chunks: list[dict], history: Optional[list[dict]] = None, mode: Optional[str] = None
    ) -> tuple[str, str]:
        """(답변, 인용 구절). 인용 우선 형식이 아니거나 모델이 형식을 안 지켰으면 인용은 ''."""
        mode = resolve_mode(mode)
        max_tokens = LLM_NUM_PREDICT if mode == "full" else LLM_NUM_PREDICT_VOICE
        messages = build_messages(question, chunks, history, mode)
        self._log_context(chunks, messages)
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            piece, eval_count = await self._chat_once(client, messages, max_tokens)
            pieces = [piece]
            # 이어쓰기는 full 모드에서만 — 1차 답변이 나열형이고(judgment은 continuation.py 참고)
            # 실제 출력이 임계 토큰 이상일 때만. voice 모드는 짧은 답이 목적이라 이어 쓰지 않는다.
            continuable = mode == "full" and looks_enumeration(piece)
            for _ in range(LLM_CONTINUE_ROUNDS):
                if not continuable or eval_count < LLM_CONTINUE_MIN_TOKENS:
                    break
                messages.append({"role": "assistant", "content": piece})
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                piece, eval_count = await self._chat_once(client, messages)
                cleaned, stop = clean_continuation(piece, "\n\n".join(pieces))
                if cleaned:
                    pieces.append(cleaned)
                if stop or cleaned is None:
                    break
        answer, quote = split_quote_answer("\n\n".join(pieces).strip())
        if quote:
            logger.info("인용: %r → 답변: %r", quote[:160], answer[:160])
        return answer, quote

    @staticmethod
    def _log_context(chunks: list[dict], messages: list[dict]) -> None:
        """근거 페이지·핵심 근거를 로그에 남긴다 (LOG_PROMPT=1 이면 프롬프트 전문도)."""
        pages = [f'p{c.get("page_number")}' for c in chunks]
        hls = [h for c in chunks for h in (c.get("highlights") or [])]
        logger.info("컨텍스트: 페이지 %s (%d자), 핵심 근거 %d줄: %s", pages, sum(len(c.get("text", "")) for c in chunks),
                    len(hls), [h[:60] for h in hls])
        if LOG_PROMPT:
            logger.info("프롬프트 전문:\n%s", messages[-1]["content"])

    async def regenerate(
        self,
        question: str,
        chunks: list[dict],
        history: Optional[list[dict]],
        mode: Optional[str],
        prev_answer: str,
        bad_items: list[str],
        evidence_lines: Optional[list[str]] = None,
    ) -> tuple[str, str]:
        """검증 실패 후 1회 재생성 → (답변, 인용). 같은 근거에 직전 답변과 '이 값은 근거에 없다' 지적을 붙여
        다시 묻는다. 이어쓰기는 하지 않는다(짧은 사실 교정이 목적)."""
        mode = resolve_mode(mode)
        max_tokens = LLM_NUM_PREDICT if mode == "full" else LLM_NUM_PREDICT_VOICE
        messages = build_messages(question, chunks, history, mode)
        messages.append({"role": "assistant", "content": prev_answer})
        messages.append({"role": "user", "content": retry_prompt(bad_items, evidence_lines)})
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            piece, _ = await self._chat_once(client, messages, max_tokens)
        logger.info("재생성 원문: %r", piece.strip()[:300])
        return split_quote_answer(piece)

    async def answer_from_line(self, question: str, line: str) -> str:
        """문서 한 줄만 근거로 한 문장 답 (prompt.LINE_ANSWER_SYSTEM). 실패·'없음' 이면 ''."""
        from .prompt import line_answer_messages

        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            piece, _ = await self._chat_once(client, line_answer_messages(question, line), 160)
        text = piece.strip()
        logger.info("한 줄 재답 원문: %r", text[:200])
        if not text or text.replace(".", "").strip() == "없음":
            return ""
        return text

    # ── 생성 (스트리밍) ──────────────────────────────────────

    async def generate_stream(
        self,
        question: str,
        chunks: list[dict],
        history: Optional[list[dict]] = None,
        mode: Optional[str] = None,
        raw_parts: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """답변을 토큰(조각) 단위로 흘려보낸다.

        1회차는 백엔드의 stream 모드를 그대로 중계하고, 이어쓰기 회차는 비스트리밍으로
        받아 정리(clean_continuation) 후 통째로 흘려보낸다 — "끝" 마커나 중복 회차를
        사용자에게 노출하지 않으려면 회차 전체를 보고 판단해야 하기 때문이다.
        """
        mode = resolve_mode(mode)
        max_tokens = LLM_NUM_PREDICT if mode == "full" else LLM_NUM_PREDICT_VOICE
        messages = build_messages(question, chunks, history, mode)
        self._log_context(chunks, messages)
        quote_first = mode == "voice" and QUOTE_FIRST
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            collected: list[str] = []
            stats: dict = {}
            held = ""  # 인용 우선 형식: '답변:' 이 나올 때까지 붙들어 두는 앞부분
            answer_started = not quote_first
            async for piece in self._stream_once(client, messages, stats, max_tokens):
                collected.append(piece)
                if raw_parts is not None:
                    raw_parts.append(piece)
                if answer_started:
                    yield piece
                    continue
                held += piece
                idx = held.find(ANSWER_LABEL)
                if idx >= 0:
                    answer_started = True
                    rest = held[idx + len(ANSWER_LABEL):].lstrip()
                    if rest:
                        yield rest
            if not answer_started and held:
                # 모델이 형식을 안 지킴 — 붙들었던 것을 통째로 흘린다
                yield held
            eval_count = int(stats.get("tokens") or 0)

            piece = "".join(collected).strip()
            accumulated = piece
            continuable = mode == "full" and looks_enumeration(piece)
            for _ in range(LLM_CONTINUE_ROUNDS):
                if not continuable or eval_count < LLM_CONTINUE_MIN_TOKENS:
                    break
                messages.append({"role": "assistant", "content": piece})
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                piece, eval_count = await self._chat_once(client, messages)
                cleaned, stop = clean_continuation(piece, accumulated)
                if cleaned:
                    accumulated += "\n\n" + cleaned
                    if raw_parts is not None:
                        raw_parts.append("\n\n" + cleaned)
                    yield "\n\n" + cleaned
                if stop or cleaned is None:
                    break

    # ── 상태 확인 ────────────────────────────────────────────

    async def health(self) -> dict:
        """서빙 서버가 떠 있고 대상 모델이 올라와 있는지 확인한다."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if self.backend == "ollama":
                    resp = await client.get(f"{self.url}/api/tags")
                    resp.raise_for_status()
                    names = [m.get("name", "") for m in resp.json().get("models", [])]
                    model_ready = any(n.split(":")[0] == self.model.split(":")[0] for n in names)
                else:
                    resp = await client.get(f"{self.url}/models")
                    resp.raise_for_status()
                    names = [m.get("id", "") for m in resp.json().get("data", [])]
                    # --served-model-name 을 LLM_MODEL 로 띄우는 것이 원칙. 원본 경로/리포 id 로
                    # 띄운 경우도 마지막 경로 요소가 같으면 인정한다.
                    model_ready = any(n == self.model or n.rstrip("/").split("/")[-1] == self.model for n in names)
            return {"status": "ok", "backend": self.backend, "model": self.model, "model_available": model_ready}
        except Exception as exc:  # 서빙 서버 미기동 등
            return {"status": "unreachable", "backend": self.backend, "model": self.model, "error": str(exc)}
