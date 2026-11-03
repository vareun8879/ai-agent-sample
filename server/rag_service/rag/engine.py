"""RAG 엔진 — 파이프라인 전체를 소유하는 파사드(facade).

질문 하나가 답변이 되기까지의 오케스트레이션(검색 → 임계값 필터 → 페이지 병합 →
생성/이어쓰기 → 근거 없음 처리)을 이 클래스가 책임진다. main.py는 HTTP 변환만
담당한다: 요청을 엔진 호출로 바꾸고, 엔진 결과·예외를 응답 형식으로 바꾼다.

엔진은 HTTP를 모른다 — FastAPI/HTTPException을 import하지 않고, 상태 문제는
도메인 예외(RetrieverNotReady)로 알린다. 덕분에 이 엔진은 CLI 배치, 평가 스크립트,
다른 프로토콜 서버 어디서든 그대로 재사용할 수 있다.

    engine = RagEngine()
    engine.load()
    result = await engine.answer("교무 업무 조직 설명해줘")
    # result = {"answer": str, "chunks": [...], "grounded": bool}
"""

import asyncio
import logging
import re
import threading
from typing import AsyncIterator, Optional

from . import verify
from .config import NO_CONTEXT_ANSWER, RERANK, RERANK_CANDIDATES, ROW_VALUE_AUDIT_MODE
from .generator import Generator
from .reranker import Reranker
from .retriever import Retriever, filter_relevant

logger = logging.getLogger("rag.engine")


class RetrieverNotReady(RuntimeError):
    """검색기가 아직 로드되지 않았다 (chroma_db 없음 또는 load() 미호출/실패)."""


class RagEngine:
    """검색기 + 생성기를 묶은 질의응답 파이프라인의 단일 진입점."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = Generator()
        self.reranker = Reranker() if RERANK else None
        # 검색·재순위는 GPU/CPU 블로킹 작업(0.3~2초)이라 스레드로 뺀다. 동시 요청이 와도
        # 모델 호출은 이 락으로 한 번에 하나만 — 결과는 이전과 같고, 그동안 이벤트 루프는
        # 다른 요청의 LLM 스트리밍·/health 응답을 계속 처리할 수 있다.
        self._retrieve_lock = threading.Lock()

    # ── 수명주기 ─────────────────────────────────────────────

    def load(self) -> None:
        """임베딩 모델 + Chroma 컬렉션을 로드한다 (기동 시 1회, /reload 시 재호출).

        인덱싱 재실행 후 재호출하면 재기동 없이 새 인덱스가 반영된다 — 단, 임베딩
        모델 자체가 바뀐 경우는 프로세스 재기동이 필요하다 (config 상수가 고정되므로).
        """
        self.retriever.load()
        if self.reranker and not self.reranker.ready:
            self.reranker.load()

    @property
    def ready(self) -> bool:
        return self.retriever.ready

    async def health(self) -> dict:
        """검색기·생성기 상태를 묶어 반환한다 (/health 응답 본문)."""
        return {
            "retriever": {"status": "ok" if self.retriever.ready else "not_loaded"},
            "generator": await self.generator.health(),
        }

    # ── 파이프라인 ───────────────────────────────────────────

    def retrieve_context(
        self,
        question: str,
        top_k: Optional[int] = None,
        institution_id: Optional[str] = None,
    ) -> list[dict]:
        """검색 → 임계값 필터 → 페이지 병합까지 수행해 LLM에 줄 근거를 만든다.

        빈 리스트 반환 = 관련 근거 없음(무관 질문). 호출 측은 LLM을 태우지 않고
        NO_CONTEXT_ANSWER를 내보내야 한다 — 억지 답변(환각) 방지.
        """
        if not self.retriever.ready:
            raise RetrieverNotReady("검색기가 준비되지 않았습니다 (인덱싱 chroma_db 확인 필요)")

        if self.reranker and self.reranker.ready:
            # reranker 경로: 임베딩으로 후보를 넓게(RERANK_CANDIDATES) 추린 뒤
            # cross-encoder가 재채점·무관 차단·상위 선택까지 담당한다
            from .config import DEFAULT_TOP_K

            search_text = self.retriever.search_text(question, institution_id)
            candidates = self.retriever.retrieve(
                question, top_k=RERANK_CANDIDATES, institution_id=institution_id, search_text=search_text
            )
            chunks = self.reranker.select(question, candidates, top_k or DEFAULT_TOP_K, search_text=search_text)
        else:
            # 폴백 경로(RERANK=0): 임베딩 거리 임계값으로만 필터
            kwargs = {"institution_id": institution_id}
            if top_k and top_k > 0:
                kwargs["top_k"] = top_k
            chunks = filter_relevant(self.retriever.retrieve(question, **kwargs))

        return self.retriever.expand_pages(chunks, question, self.reranker)

    async def retrieve_context_async(
        self,
        question: str,
        top_k: Optional[int] = None,
        institution_id: Optional[str] = None,
    ) -> list[dict]:
        """retrieve_context를 워커 스레드에서 실행한다 (HTTP 핸들러는 이쪽을 쓴다)."""

        def run() -> list[dict]:
            with self._retrieve_lock:
                return self.retrieve_context(question, top_k, institution_id)

        return await asyncio.to_thread(run)

    async def answer(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        top_k: Optional[int] = None,
        institution_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> dict:
        """질문 → 근거 기반 답변 (비스트리밍 전체 파이프라인).

        반환: {"answer": 답변 텍스트, "chunks": 근거 조각들, "grounded": 근거 기반 여부}
        - 빈 질문: answer="" / grounded=False
        - 근거 없음: answer=NO_CONTEXT_ANSWER / grounded=False (LLM 미호출)
        """
        question = question.strip()
        if not question:
            return {"answer": "", "chunks": [], "grounded": False}

        chunks = await self.retrieve_context_async(question, top_k, institution_id)
        if not chunks:
            logger.info("근거 없음 (임계값 초과) — 고정 안내문 반환: %r", question)
            return {"answer": NO_CONTEXT_ANSWER, "chunks": [], "grounded": False}

        text, quote = await self.generator.generate_with_quote(question, chunks, history, mode)
        # 2.5단계: 답변의 숫자·날짜가 근거(인용)에 있는지 대조 — 없으면 지적하고 재생성, 그래도 없으면 인용/차단
        text, ok, bad = await self.verify_with_retry(question, chunks, history, mode, text, quote)
        return {"answer": text, "chunks": chunks, "grounded": ok, "unverified": bad}

    async def verify_with_retry(
        self,
        question: str,
        chunks: list[dict],
        history: Optional[list[dict]],
        mode: Optional[str],
        text: str,
        quote: str = "",
    ) -> tuple[str, bool, list[str]]:
        """숫자(컨텍스트 대조)·날짜(인용 구절 대조) 검증 → 실패 시 VERIFY_RETRY 회 재생성 → 재검증.
        끝까지 실패하면: 인용 구절이 [문서]에 실제로 있는 줄이면 그 구절을 그대로 읽어 주고(모델 문장보다 안전),
        아니면 확인 안내문(UNVERIFIED_ANSWER). 스트리밍 경로(main.py)도 이걸 쓴다."""
        from .config import NO_CONTEXT_ANSWER, QUOTE_ANSWER_PREFIX, VERIFY_RETRY
        from .textutil import clean_answer, has_value_digit, is_value_question, pick_evidence_line, prose_sentences, question_overlap, question_tokens, split_compound_question, strip_paren_numbers, whole_word_hits

        ctx = verify.context_text(chunks) if chunks else ""
        # 복합 질문의 부분들 — 검증(부분별 행 대조·항목 누락)과 폴백(항목별 줄 고르기)이 같이 쓴다
        c_parts = split_compound_question(question)
        hl_lines = [h for c in chunks for h in (c.get("highlights") or [])]
        hl = "\n".join(hl_lines)
        ctx_lines = hl_lines + [l.strip() for c in chunks for l in c.get("text", "").split("\n") if l.strip()]

        def real_quote(q: str) -> str:
            """모델이 '근거:' 에 쓴 것이 [문서]에 실제로 있는 구절일 때만 인정한다.
            실측(09-17): 재생성에서 모델이 근거 자리에 자기 문장("1월 31일부터 1월 2일까지")을 쓰고 답도 그대로
            써서, 가짜 인용이 날짜 허용 목록이 되어 틀린 답이 통과했다. 지어낸 인용은 검증에도 폴백에도 쓰지 않는다."""
            # 프롬프트에는 괄호 부가 숫자를 지운 본문이 갔으므로(prompt.build_prompt) 그 형태로도 대조한다
            if q and (verify.quote_in_context(q, ctx) or verify.quote_in_context(q, strip_paren_numbers(ctx))):
                return q
            if q:
                logger.info("인용이 [문서]에 없음(모델 문장) — 검증에서 제외: %r", q[:100])
            return ""

        async def say_line(line: str) -> str:
            """문서 한 줄을 그대로 읽어 주기 전에 그 줄만 주고 문장으로 바꿔 본다 — 값은 그 줄에 있는 것만 허용.
            검증(숫자·날짜·행 대조·동문서답)을 통과하면 그 문장을, 아니면 '' (호출 측이 줄을 그대로 읽어 준다)."""
            try:
                t = await self.generator.answer_from_line(question, line)
            except Exception as exc:
                logger.warning("한 줄 재답 실패 — 줄을 그대로 읽어 줌: %s", exc)
                return ""
            if not t or verify.is_refusal(t):
                return ""
            if not re.search(r"[다요죠]\s*[.!]?\s*$", t.strip()):
                # 문장이 아닌 조각("계: 255")은 재답이 아니다 — 실측(09-21 키오스크): "전체 학생 수는?" 이 "계: 255" 로 나갔다
                logger.info("한 줄 재답이 문장이 아님 — 줄을 그대로 읽어 줌: %r", t[:60])
                return ""
            # 동문서답 검사는 재답의 원천 줄에 질문 낱말이 없을 때만 — 줄이 이미 질문 낱말을 담고 있으면(폴백 줄은 그렇게
            # 골랐다) 값만 말한 짧은 문장도 그 질문의 답이다. 실측(09-21 키오스크): "개교는 언제야?" 재답 "1982년 3월 1일입니다."
            # 가 '개교' 가 없다고 탈락해 "자료에는 다음과 같이 적혀 있습니다. 1982.03.01. 망미여자중학교 개교 초대 …" 를 읽어 줬다.
            bad2 = (verify.unsupported_numbers(t, line) + verify.unsupported_dates(t, line, "", line)
                    + verify.unsupported_row_dates(question, t, [line])
                    + ([] if question_overlap(question, line) > 0 else verify.off_topic(question, t)))
            if bad2:
                logger.info("한 줄 재답 검증 실패 %s — 줄을 그대로 읽어 줌: %r", bad2, t[:100])
                return ""
            return clean_answer(t)

        def problems(t: str, q: str) -> list[str]:
            # 이름 검사의 근거는 모델이 실제로 인용한 줄이다 — 인용이 없을 때만 핵심 근거 전체.
            # 실측(09-18): "담임 다음에 누구에게 보고?" 정답이 역할(보건교사)인 보고체계 줄을 인용했는데,
            # 다른 핵심 근거 줄의 담당자 이름을 답에 요구해 정답을 막았다.
            # 인용이 없으면 질문 낱말이 든 근거 줄만 이름 검사의 기준으로 — 실측(09-21 키오스크): "학생선도위원회 담당자는?"
            # 정답 행("… 담당자 선도위원회 담당", 역할이 답)을 두고 다른 부서 행의 "담당자 김OO" 을 답에 요구해 정답을 막았다.
            # 기준 줄은 사람을 묻는 낱말(담당자·누구·성함…)을 뺀 **주어 낱말**로 고른다 — "담당자" 는 어느 부서 행에나 있어
            # 그것까지 세면 엉뚱한 행의 이름이 또 기준이 된다.
            subj_toks = [t for t in question_tokens(question) if not verify.is_who_question(t)]
            # 접두 일치(question_overlap)는 "학생선도위원회" 가 2자 "학생" 까지 내려가 "장학생 선정 위원회" 행도 잡는다 — 낱말 그대로만.
            subj = [l for l in hl_lines if subj_toks and whole_word_hits(l, subj_toks) > 0]
            # 행 대조는 복합 질문이면 **부분마다** 한다 — 질문 낱말을 한 뭉치로 묶으면 한 항목만 맞아도 통과한다.
            # 시설별 배치도 행을 각 부분 질문에 대조한다.
            rows = verify.unsupported_row_numbers(question, t, ctx_lines)
            for part in c_parts:
                rows += [b for b in verify.unsupported_row_numbers(part, t, ctx_lines) if b not in rows]
            return (
                verify.unsupported_numbers(t, ctx)
                + [f"{verify.UNSUPPORTED_EN} {w}" for w in verify.unsupported_english(t, ctx)]
                + [f"{verify.UNSUPPORTED_URL} {u}" for u in verify.unsupported_urls(t, ctx)]
                + verify.unsupported_dates(t, q, hl, ctx)
                + verify.missing_person(question, t, q or ("\n".join(subj) if subj else hl))
                + rows
                + verify.unsupported_row_dates(question, t, ctx_lines)
                + verify.ordinal_mismatch(question, q)
                + verify.year_mismatch(question, t)
                + verify.off_topic(question, t)
            )

        # 모델이 쓴 문장 전부(인용·답·재생성) — [문서]에 없는 가짜라도 "모델이 어느 줄을 봤는지" 는 담고 있다 (폴백 후보 힌트).
        # 실측(09-18): 첫 인용에는 정답 줄의 낱말(기존·고도화·실증·국내외)이 있었는데 재생성 인용은 값만 남겨, 마지막 인용만
        # 힌트로 쓰니 옆 행이 다시 이겼다. 전부 모은다.
        model_words = [quote, text]
        quote = real_quote(quote)
        if verify.is_refusal(text) and VERIFY_RETRY:
            # 근거에 질문 낱말이 충분히 겹치는 줄이 있는데 거절했다 — 그 줄을 짚어 주며 한 번 더 묻는다 (verify.refusal_evidence 참고)
            # 값 질문이면 숫자가 든 줄만 짚어 준다 — 실측(09-21 키오스크): "망미중학교 학생 수는?" 에 소제목 줄
            # "일반 현황 가. 학생 현황 1) 학급 편제 및 학생수" 를 짚어 주고, 재질문에도 거절하자 그 소제목을 읽어 줬다.
            ev_lines = [l for l in hl_lines if has_value_digit(l, question)] if is_value_question(question) else hl_lines
            evidence = verify.refusal_evidence(question, ev_lines) or ("" if ev_lines else verify.refusal_evidence(question, hl_lines))
            if evidence:
                logger.info("거절했지만 근거 줄이 있음 — 재질문: %r", evidence[:100])
                try:
                    text2, quote2 = await self.generator.regenerate(
                        question, chunks, history, mode, text, [f"{verify.REFUSED_WITH_EVIDENCE}: {evidence}"])
                except Exception as exc:
                    logger.warning("재질문 실패 — 거절 답변 유지: %s", exc)
                    text2 = ""
                if text2 and not verify.is_refusal(text2):
                    text, quote = text2, real_quote(quote2)
                    model_words += [quote2, text2]
                    logger.info("재질문 답변: %r", text[:120])
                else:
                    # 실측(09-18): 값이 괄호 안에 있는 줄("…개편(총사업비 약 105억원)되어…")은 짚어 줘도 거절했다 —
                    # 프롬프트의 "괄호 안 숫자를 답에 넣지 마라" 를 넓게 적용한 것. 질문 낱말이 2개 이상 겹치는 문서 줄이니
                    # 모델 문장 대신 그 줄을 그대로 읽어 준다 (거절보다 낫고, 지어낸 값은 아니다)
                    said = await say_line(evidence)
                    if said:
                        logger.info("재질문에도 거절 — 근거 줄로 한 줄 재답: %r", said[:120])
                        return said, True, []
                    logger.warning("재질문에도 거절 — 근거 줄을 그대로 읽어 줌: %r", evidence[:120])
                    return QUOTE_ANSWER_PREFIX + evidence, True, []
        bad = problems(text, quote)
        for attempt in range(VERIFY_RETRY if bad else 0):
            logger.info("검증 실패(근거에 없는 값 %s) — 재생성 %d회차", bad, attempt + 1)
            try:
                text, quote2 = await self.generator.regenerate(question, chunks, history, mode, text, bad)
                quote = real_quote(quote2) or quote
                model_words += [quote2, text]
            except Exception as exc:
                logger.warning("재생성 실패 — 1차 답변으로 검증 진행: %s", exc)
                break
            bad = problems(text, quote)
            if not bad:
                logger.info("재생성 답변 검증 통과: %r", text[:120])
                break
        if not bad and c_parts:
            # 검증은 통과했지만 복합 질문의 한 항목만 답한 경우 — 빠진 항목을 짚어 한 번만 재생성한다.
            # 나아지지 않으면 1차 답변을 그대로 내보낸다(막지 않는다) — 한 항목이라도 맞은 답이 안내문보다 낫고,
            # 표현이 문서와 달라 누락으로 잘못 잡힐 수도 있다.
            miss = verify.missing_part(question, text, c_parts)
            if miss:
                logger.info("복합 질문의 항목 누락 %s — 재생성 1회", miss)
                try:
                    text2, quote2 = await self.generator.regenerate(question, chunks, history, mode, text, miss)
                except Exception as exc:
                    logger.warning("항목 누락 재생성 실패 — 1차 답변 유지: %s", exc)
                    text2 = ""
                if text2 and not verify.is_refusal(text2):
                    q2 = real_quote(quote2)
                    if not problems(text2, q2) and not verify.missing_part(question, text2, c_parts):
                        text, quote = text2, q2 or quote
                        model_words += [quote2, text2]
                        logger.info("항목 누락 재생성 답변: %r", text[:120])
                    else:
                        logger.info("항목 누락 재생성이 나아지지 않음 — 1차 답변 유지: %r", text2[:80])
        if bad and verify.OFF_TOPIC in bad:
            # 질문과 무관한 답 — 인용 줄도 질문과 무관하게 잡힌 것이라 읽어 주면 안 된다. 다만 핵심 근거에 질문 낱말이 충분히
            # 겹치는 줄이 따로 있으면(refusal_evidence 기준) 모델이 옆 행을 잘못 고른 것이므로 그 줄로 답한다 — 실측(09-21
            # 키오스크): "밴드 자체 연주회는 언제?" 근거에 "11월 … 자체 연주회 실시" 행이 있는데 모델이 1월 졸업 공연 행을 인용해
            # 안내문으로 끝났다. 무관 질문("테스트")은 그런 줄이 없어 예전처럼 안내문.
            ev_lines = [l for l in hl_lines if has_value_digit(l, question)] if is_value_question(question) else hl_lines
            evidence = verify.refusal_evidence(question, ev_lines)
            if evidence:
                said = await say_line(evidence)
                if said:
                    logger.info("검증 실패(질문과 무관한 답) — 질문 낱말이 든 근거 줄로 한 줄 재답: %r", said[:120])
                    return said, True, bad
                logger.info("검증 실패(질문과 무관한 답) — 질문 낱말이 든 근거 줄을 그대로 읽어 줌: %r", evidence[:120])
                return QUOTE_ANSWER_PREFIX + evidence, True, bad
            logger.warning("검증 실패(질문과 무관한 답) — 관련 내용 없음 처리: %r", text[:120])
            return NO_CONTEXT_ANSWER, False, bad
        if bad and any(verify.ROW_MISMATCH in b for b in bad):
            # 값이 다른 줄에서 왔다 — 모델이 인용한 줄보다 질문 대상이 적힌 줄을 읽어 준다
            quote = ""
        if bad:
            # 모델의 문장은 못 믿는다 — [문서] 그대로인 구절을 읽어 준다: 모델의 진짜 인용 > 핵심 근거 줄(표 행)과
            # 본문 문장을 한 풀에 놓고 질문 낱말이 가장 많이 겹치는 것 (실측 09-18: 표 행을 먼저 보니 2점짜리 엉뚱한 행이
            # 9점짜리 본문 문장 "…직선거리 200미터…" 을 가렸다)
            from .textutil import filter_survey_lines
            pool = filter_survey_lines(question, hl_lines + [st for c in chunks for st in prose_sentences(c.get("text", ""))])
            # 줄 → 페이지 순위(chunks 는 확장된 페이지 단위, 검색 순위순). 동점일 때 앞선 페이지의 줄을 고른다 — 실측(09-19):
            # "입학식은 언제야?" 에 3위 페이지(연혁)의 1982년 입학식 줄이 1위 페이지(학사일정)의 2026.3.3 줄을 길이로 이겼다
            rank = {}
            for ci, c in enumerate(chunks):
                for l in (c.get("highlights") or []) + prose_sentences(c.get("text", "")):
                    rank.setdefault(l, ci)
            # 후보 점수의 기준 낱말: 질문 + (진짜 인용이 없을 때) 가짜 인용. 실측(09-18): "상용화 지원 규모?" 에 모델이
            # "…1000만원에서 2500백만원… 기존 제품 고도화·실증·국내외 사업화…" 라고 값을 ×10 해 쓴 가짜 인용을 남겼는데,
            # 질문 낱말만으로는 표 머리말(지원 규모·기업당)이 든 옆 행(시제품 50~150)이 동점으로 이겼다. 가짜 인용이 베낀
            # 낱말(기존·고도화·실증·국내외·사업화)을 더하면 모델이 실제로 본 줄(상용화 100~250)이 압도한다
            hint = "" if quote else " ".join(w for w in model_words if w)

            def pick_for(q: str, use_hint: bool = True) -> str:
                """질문(또는 복합 질문의 한 부분)에 맞는 폴백 줄 하나 — 질문 유형별 후보 제한.

                복합 질문의 부분에는 hint(모델이 쓴 문장들)를 쓰지 않는다(use_hint=False) — 부분 두 개에 같은 힌트를
                더하면 모델이 인용했던 한 줄이 두 부분 모두에서 이겨 같은 줄이 뽑히거나 옆 항목 줄이 뽑힌다.
                실측(09-23 골든 m33·a35): "교육과정부와 생활안전부 부장" 이 두 부분 다 안전대책위 줄로, "시제품과 상용화
                지원 규모" 가 두 부분 다 상용화 행으로 갔다. 부분 질문은 주어 낱말이 또렷해 힌트가 필요하지 않다."""
                hint_q = hint if use_hint else ""
                if verify.is_who_question(q):
                    # "누구?" 는 이름이 든 줄을 먼저, 없으면 역할(교사·부장·실장…)이 든 줄 — 둘 다 없으면 안내문
                    got = pick_evidence_line(q, [l for l in pool if verify.has_person_name(l)], hint_q, rank) or pick_evidence_line(
                        q, [l for l in pool if verify.has_person_or_role(l)], hint_q, rank)
                    if not got and not quote:
                        logger.info("폴백 후보에 이름·역할이 든 줄이 없어 사용하지 않음")
                    return got
                if is_value_question(q):
                    # 값(수·날짜·비율)을 묻는데 숫자 없는 줄(소제목·표 제목)을 읽어 주면 답이 아니다 — 실측(09-20):
                    # "망미중학교 전체 교직원 수는?" → "교직원 현황 1) 교직원 정원 및 현원 2) 교과별 교사 현황 다." 나열.
                    # 숫자 든 줄에서 먼저 고르고, 없으면 안내문으로 간다.
                    got = pick_evidence_line(q, [l for l in pool if has_value_digit(l, q)], hint_q, rank)
                    if not got and not quote:
                        logger.info("값 질문인데 숫자가 든 폴백 줄이 없어 사용하지 않음")
                    return got
                return pick_evidence_line(q, pool, hint_q, rank)

            parts = c_parts
            if parts:
                # 복합 질문("A와 B는 각각 …?")은 항목마다 줄을 골라 이어 읽는다 — 실측(09-22 수동 #96·#97): 한 줄만 고르니
                # 두 항목 다 안 든 엉뚱한 줄(교과협의회 10월, 안전대책위원회)을 읽어 줬다. 항목 하나라도 못 찾으면 전체 질문으로.
                picked = []
                for part in parts:
                    line = pick_for(part, use_hint=False)
                    if line:
                        picked.append(line)
                best = " ".join(dict.fromkeys(picked)) if len(picked) == len(parts) else pick_for(question)
                if len(picked) == len(parts):
                    quote = ""   # 모델 인용은 한 항목뿐이라 두 줄을 이은 쪽을 쓴다
                    logger.info("복합 질문 폴백 — 항목별 줄 %d개: %s", len(picked), [l[:50] for l in picked])
            else:
                best = pick_for(question)
            # 모델의 진짜 인용이라도 질문 낱말이 덜 겹치면 옆 줄을 잘못 인용한 것 — 실측(09-18): "상용화 지원 규모?" 에
            # 시제품 행을 인용해 폴백이 그 행을 읽었다. 질문과 더 겹치는 줄(상용화 100~250백만원)이 후보에 있으면 그쪽
            if quote and best and question_overlap(question, best) > question_overlap(question, quote):
                logger.info("인용(%d점)보다 질문 낱말이 더 겹치는 줄(%d점)로 폴백", question_overlap(question, quote), question_overlap(question, best))
                source = best
            elif quote and is_value_question(question) and not has_value_digit(quote, question):
                source = best   # 모델 인용이 숫자 없는 제목 줄이면 값 줄을 우선 (없으면 안내문)
            else:
                source = quote or best
            if source:
                said = await say_line(source)
                if said:
                    logger.info("검증 실패(%s) — 문서 구절로 한 줄 재답: %r", bad, said[:120])
                    return said, True, []
                logger.warning("검증 실패(%s) — 모델 문장 대신 문서 구절로 답변: %r", bad, source[:120])
                return QUOTE_ANSWER_PREFIX + source, True, []
            logger.warning("검증 실패 — 근거에 없는 값 %s. 답변 차단: %r", bad, text[:120])
            return verify.UNVERIFIED_ANSWER, False, bad
        sents = [s for s in re.split(r"(?<=[.!?。])\s+", text.strip()) if s]
        if sents and all(verify.is_refusal(s) for s in sents):
            # 검색은 됐지만 모델이 '찾을 수 없다' 고만 답했다(거절 재질문까지 거친 뒤) — 근거 없는 답이므로 grounded=False 로,
            # 문구도 검색 0건과 같은 안내문으로 통일한다. 실측(09-22 골든 bx04·수동 #81·#184·#285~#290): 거절문이
            # grounded=True 로 나가 키오스크 로그·채점에서 '근거 있는 답' 으로 집계됐고, 안내 문구가 두 종류였다.
            # 복합 질문의 부분 답("기숙사는 우남학사입니다. 도서관은 찾을 수 없습니다.")은 답이 섞여 있으므로 그대로 둔다.
            logger.info("모델 거절 — 관련 내용 없음 처리: %r", text[:80])
            return NO_CONTEXT_ANSWER, False, []
        # 모델 문장은 괄호 부가 숫자·표 기호 문장을 뗀다 (폴백 인용은 원문 그대로 — "자료에는 이렇게 적혀 있다" 는 형태라서)
        answer = clean_answer(text)
        if ROW_VALUE_AUDIT_MODE in ("log", "retry") and c_parts and re.search(r"\d", answer):
            try:
                facts = self.retriever.fact_lines_on_pages(chunks)
                issues = verify.audit_answer_values(question, answer, facts)
                if issues and ROW_VALUE_AUDIT_MODE == "log":
                    logger.info("사실 행 값 대조(로그 전용) question=%r answer=%r issues=%s",
                                question, answer[:160], issues)
                elif issues:
                    # 실측(261102_4, m29·m30): 두 항목의 근거를 함께 주고 각각 검증한다.
                    paired = verify.audit_answer_values(question, answer, facts, with_evidence=True)
                    if not paired:
                        logger.info("사실 행 값 재생성 생략 — 두 항목의 근거 미확정: %r", question)
                        return answer, True, []
                    logger.info("사실 행 값 불일치 — 두 항목 재생성 1회 question=%r issues=%s",
                                question, issues)
                    text2, quote2 = await self.generator.regenerate(
                        question, chunks, history, mode, answer,
                        issues,
                        [f"부분 질문: {part}\n근거 행: {line}" for part, line, _ in paired],
                    )
                    candidate = clean_answer(text2)
                    q2 = real_quote(quote2)
                    reasons = (verify.paired_value_problems(candidate, paired)
                               + problems(candidate, q2)
                               + verify.missing_part(question, candidate, c_parts)
                               + verify.audit_answer_values(question, candidate, facts))
                    if not reasons:
                        logger.info("사실 행 값 재생성 채택: %r", candidate[:160])
                        return candidate, True, []
                    logger.info("사실 행 값 재생성 검증 미통과 — 기존 답변 유지 reasons=%s answer=%r",
                                reasons, candidate[:160])
            except Exception:
                logger.exception("사실 행 값 대조 실패 — 답변 유지: %r", question)
        return answer, True, []

    def answer_stream(
        self,
        question: str,
        chunks: list[dict],
        history: Optional[list[dict]] = None,
        mode: Optional[str] = None,
        raw_parts: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """근거가 이미 준비된 상태에서 답변을 토큰 단위로 스트리밍한다.

        스트리밍은 근거(sources)를 먼저 클라이언트에 알려야 하는 프로토콜 특성상
        retrieve_context()를 호출 측이 먼저 실행하고, 그 결과를 이 메서드에 넘긴다.
        raw_parts 를 주면 모델 원문(인용 포함)이 거기에 쌓인다 — 호출 측이 검증에 쓴다.
        """
        return self.generator.generate_stream(question, chunks, history, mode, raw_parts)
