"""1.2단계 — 재순위화(Reranker).

임베딩 검색(bi-encoder)은 질문과 문서를 따로 벡터화해 비교하므로 빠르지만 정밀도에
한계가 있다 — 실측 사례: "생활안전부 상담 담당자는 누구야?"에 조직표(p27)가 아니라
역할 설명 페이지(p93)가 1위로 뽑혀 담당자 이름이 답변에 못 들어갔다.

cross-encoder는 (질문, 조각) 쌍을 한 모델에 함께 넣어 상호작용을 직접 보므로 훨씬
정밀하다. 대신 쌍마다 추론이 필요해 느리다 — 그래서 임베딩 검색으로 후보를 넓게
(RERANK_CANDIDATES개) 추린 뒤, 그 후보들만 재채점하는 2단 구성을 쓴다.

무관 질문 차단도 reranker 점수로 한다(RERANK_MIN_SCORE, 골든셋 실측 캘리브레이션) —
임베딩 거리 임계값(MAX_RETRIEVAL_DISTANCE)은 reranker 비활성 시의 폴백 경로에서만 쓴다.
"""

import logging
from typing import Optional

from .config import (
    RERANK_CANDIDATES,
    RERANK_LEXICAL_RESCUE,
    RERANK_MIN_SCORE,
    RERANK_MODEL,
    RERANK_RELATIVE_MIN,
    RERANK_RESCUE_MARGIN,
)

logger = logging.getLogger("rag.reranker")

from .textutil import strip_title


def _passage(text: str) -> str:
    """재채점에 넣을 지문 — 문서 제목 접두를 뗀다.

    실측(2026-09-17, bge-reranker-v2-m3): 제목 줄이 붙으면 짧은 지문의 확률이 4~10배 떨어진다
    ("학생수: 1학년 83명 … 전체 255명" 0.63 → 0.12, "여름방학: 2026.7.22.~8.17." 0.95 → 0.38).
    제목은 임베딩 검색(문서 구분)과 LLM 프롬프트(출처 표시)에는 필요하므로 저장 텍스트는 그대로 두고
    cross-encoder 입력에서만 뗀다. 긴 페이지 조각은 영향이 작다(±0.01)."""
    return strip_title(text)


class Reranker:
    """(질문, 조각) 쌍 정밀 재채점기. 무거운 모델은 load()에서 한 번만 준비한다."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        """cross-encoder 모델을 로드한다. 서버 기동 시 한 번만 호출한다."""
        # import를 메서드 안에 둬서, reranker를 끈 구성에서는 로딩 비용을 치르지 않는다
        from sentence_transformers import CrossEncoder

        from .config import RERANK_DEVICE, fp16_model_kwargs

        extra = fp16_model_kwargs(RERANK_DEVICE)
        logger.info("reranker 모델 로드: %s (device=%s, fp16=%s)", RERANK_MODEL, RERANK_DEVICE or "auto", bool(extra))
        self._model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, **extra)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """기존 모델로 질문·지문 쌍을 일괄 평가한다. 별도 모델을 로드하지 않는다."""
        if not self.ready:
            raise RuntimeError("Reranker가 아직 load되지 않았습니다")
        if not pairs:
            return []
        import math
        scores = [float(s) for s in self._model.predict([(q, _passage(t)) for q, t in pairs])]
        if len(scores) != len(pairs) or not all(math.isfinite(s) for s in scores):
            raise ValueError("리랭커 점수 개수 또는 값이 잘못되었습니다")
        return scores

    def select(self, question: str, candidates: list[dict], top_k: int, search_text: Optional[str] = None) -> list[dict]:
        """후보 조각들을 재채점해 관련도 순 상위 top_k개를 돌려준다.

        - 각 조각에 rerank_score(0~1 확률)와 rank_key(작을수록 관련 — 페이지
          병합의 우선순위 정렬이 거리 기반이라 1-score로 맞춘다)를 붙인다.
        - RERANK_MIN_SCORE 미만은 무관으로 버린다. 전부 버려지면 빈 리스트 —
          호출 측은 근거 없음(고정 안내문) 경로를 탄다.
        """
        if not self.ready:
            raise RuntimeError("Reranker가 아직 load되지 않았습니다")
        if not candidates:
            return []

        from .textutil import search_question
        # search_text = 검색기가 범위에 맞춰 제목 낱말을 정리한 질문 (retriever.search_text). 없으면 전부 지운 것으로.
        q = search_text if search_text is not None else search_question(question)
        pairs = [(q, c["text"]) for c in candidates]
        # sentence-transformers 5.x 의 CrossEncoder.predict 는 단일 라벨 모델에 기본으로 Sigmoid 를
        # 적용해 이미 0~1 확률을 돌려준다 (activation_fn 기본값). 예전 코드는 "로짓을 반환한다" 고
        # 가정하고 sigmoid 를 한 번 더 씌웠고, 그 결과 모든 점수가 sigmoid([0,1]) = 0.500~0.731 로
        # 뭉개졌다 — 실측(2026-09-17 서버): 관련 0.502~0.730, 무관 0.500~0.503. 실제 확률은
        # 관련 0.008~0.99, 무관 0.000~0.012 였다. 이중 적용을 제거하고 실제 확률을 그대로 쓴다.
        scores = self.score_pairs(pairs)

        scored = []
        for c, s in zip(candidates, scores):
            c = dict(c)
            c["rerank_score"] = round(float(s), 4)
            c["rank_key"] = 1.0 - float(s)
            scored.append(c)
        scored.sort(key=lambda c: c["rank_key"])

        # 하한 = max(절대 임계, 1위 × 상대 비율). 절대 임계는 무관 질문 차단, 상대 비율은 곁다리 페이지 제거
        # (config.RERANK_RELATIVE_MIN 주석 참고 — 7B 모델이 관련 낮은 페이지를 섞어 읽는 실측 사례)
        floor = max(RERANK_MIN_SCORE, scored[0]["rerank_score"] * RERANK_RELATIVE_MIN)
        kept = [c for c in scored if c["rerank_score"] >= floor][:top_k]
        if RERANK_LEXICAL_RESCUE > 0:
            # 낱말 구제 — config.RERANK_LEXICAL_RESCUE 주석 참고. 질문 낱말(군더더기·기관명 제외)이 **낱말 그대로** 든
            # 낱말 일치 후보 중 리랭커 점수가 가장 높은 하나는, 이미 통과한 페이지에 없으면 항상 더한다(임계 미달이라도).
            # 페이지 확장(expansion)이 그 페이지 전문을 붙인다.
            # 실측(09-21 키오스크): "개교는 언제야?" — 달력 쪽(개교기념일)만 통과하고 연혁 행("1982.03.01. … 개교")은 탈락해
            # 모델이 개학·개교기념일 날짜로 답했다. 질문 낱말이 그대로 든 행은 답의 후보이므로 리랭커 점수와 무관하게 보여 준다.
            from . import verify
            from .textutil import question_tokens, strip_title, subject_tokens, whole_word_hits
            toks = question_tokens(question)
            kept_pages_now = {(c["source_file"], c["page_number"]) for c in kept}
            # 사람을 묻는 질문("교장 선생님 성함이 뭐야?")은 주어 낱말(교장)만으로 세고, 이름이 든 줄만 구제한다 —
            # 실측(09-22 골든 m09·m10·수동 #54): 조직도 줄에 '선생님·성함' 이 없어 구제되지 않아 "찾을 수 없음", 보건 담당자 질문은
            # 업무 페이지만 잡혀 "보건 선생님은 보건교사입니다" 로 답했다. 이름 없는 줄(결재 규정의 '교장')은 구제하지 않는다.
            who = verify.is_who_question(question)
            if who:
                toks = subject_tokens(question)
            # 점수 하한(RERANK_LEXICAL_RESCUE)은 통과한 페이지가 하나도 없을 때만 본다 — 무관 질문에 낱말만 겹치는 줄로
            # 답하는 것을 막는 용도. 이미 통과한 페이지가 있으면 질문은 문서 범위 안이므로 낱말 일치 행 하나를 더 보여 주는
            # 위험은 작다. 실측(09-21 키오스크): "개교는 언제야?" — 리랭커가 달력의 "2학기 개학" 줄에 0.344 를 주고 연혁 행
            # ("1982.03.01. … 개교")에는 0.003 미만을 줘 구제가 안 됐고, 모델은 개학 날짜를 개교라고 답했다(5회 연속).
            # 통과 페이지가 있을 때는 질문 낱말 **전부**가 낱말 그대로 든 행만 — 실측(09-21 골든 회귀 e91·m21·e14·e08): "미술실은
            # 몇 층?" 에 '미술실' 하나로 교실 수 표("미술실 1, 도서실 1")가 구제돼 모델이 1 을 층으로 읽었다. "개교는 언제야?"(낱말 1개)는
            # 그대로 구제된다. 통과 페이지가 없을 때는 예전대로 점수 하한 + 낱말 하나.
            need = len(toks) if kept else 1
            lex = [c for c in scored if c.get("lexical", 0) > 0
                   and (kept or c["rerank_score"] >= RERANK_LEXICAL_RESCUE)
                   and (c["source_file"], c["page_number"]) not in kept_pages_now
                   and whole_word_hits(strip_title(c["text"]), toks) >= need
                   and (not who or verify.has_person_name(strip_title(c["text"])))]
            if lex:
                kept.append(lex[0])
                logger.info("낱말 구제: 질문 낱말이 그대로 든 행을 %s (%.3f, p%s %r)", "살림" if len(kept) == 1 else "추가",
                            lex[0]["rerank_score"], lex[0]["page_number"], strip_title(lex[0]["text"])[:50])

        # 인접 페이지 구제: 표가 페이지를 넘어 이어지는 경우(실측: 업무조직 표 p27~28)
        # 뒷장 조각은 앞장보다 채점이 조금 낮아 임계·top_k에서 아깝게 탈락할 수 있다.
        # 그러면 페이지 병합(expansion)이 뒷장을 이어붙일 근거가 사라져 반쪽 답변이
        # 재발한다 — 선택된 페이지의 **바로 다음 페이지** 후보가 근소 차(RESCUE_MARGIN)로
        # 탈락한 상태라면 함께 살린다.
        # 다음(+1) 페이지만 구제하고 앞(-1) 페이지는 살리지 않는다 — 표의 연속은 항상
        # 뒤로 이어지고, 앞 페이지 구제는 실측에서 역효과였다: 표 시작 전의 요약 도표
        # (p26 경영조직)가 컨텍스트 맨 앞에 끼면서 답변이 4,855자→3,340자로 위축되고
        # 행 누락(1학년 부장)이 생겼다.
        kept_pages = {(c["source_file"], c["page_number"]) for c in kept}
        rescue_floor = RERANK_MIN_SCORE - RERANK_RESCUE_MARGIN
        for c in scored:
            key = (c["source_file"], c["page_number"])
            if key in kept_pages or c["rerank_score"] < rescue_floor:
                continue
            if not isinstance(c["page_number"], int):
                continue
            if (c["source_file"], c["page_number"] - 1) in kept_pages:
                kept.append(c)
                kept_pages.add(key)

        logger.info(
            "rerank: 후보 %d → 통과 %d (최고 %.3f, 임계 %.2f, 상대 하한 %.3f) 페이지 %s",
            len(candidates), len(kept), scored[0]["rerank_score"], RERANK_MIN_SCORE, floor,
            sorted({c["page_number"] for c in kept}),
        )
        return kept


__all__ = ["Reranker", "RERANK_CANDIDATES"]
